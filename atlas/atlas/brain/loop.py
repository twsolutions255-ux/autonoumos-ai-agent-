"""The autonomous cycle.

One `Runtime` owns everything with a lifetime: the app client, the store, the
policy, the memory, and the model. It exposes three entry points —
`run_cycle` for scheduled autonomous work, `chat` for the owner talking to
Atlas, and `serve_forever` for the scheduler that drives the first one.

The cycle is deliberately the same shape every time, because a variable
process cannot be reasoned about after the fact:

    open the cycle record  ->  gather context  ->  reason and act
    ->  close the record with what happened

Everything that happens inside is written to `atlas_cycles` and
`atlas_actions` as it happens, not at the end. A cycle that crashes halfway
still leaves a complete account of what it did up to that point, which is the
only way to debug an agent that ran at 3am.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from ..config import Settings
from ..db import Store, iso, now
from ..guardrails.policy import Policy, RateLimiter
from ..llm.engine import BudgetExceeded, Engine
from ..memory.store import MemoryStore
from ..tools.registry import ToolContext, registry
from ..tws.client import TWSClient, TWSError
from . import prompts

log = logging.getLogger("atlas.loop")

# Importing the tool modules is what registers them. Without this the registry
# is empty and Atlas has no hands.
from ..tools import comms, growth, money, observe, reflect  # noqa: E402,F401


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = Store(settings.mongo_url, settings.mongo_db,
                           settings.collection_prefix) if settings.can_remember else None
        self.memory = MemoryStore(self.store) if self.store else None
        self.policy = Policy(settings, RateLimiter())
        self.client: Optional[TWSClient] = None
        self.engine: Optional[Engine] = None
        self._spend_today = 0.0
        self._spend_day = ""
        self._running = False
        self._cycle_lock = asyncio.Lock()
        self.last_cycle: Optional[dict] = None
        self.identity: Optional[dict] = None
        # Which cycle a tool call belongs to, and whether a person is waiting
        # on it. Set around each turn so dispatch can tag the audit trail.
        self._current_cycle_id = ""
        self._interactive = False

    # ---------------------------------------------------------------- boot

    async def start(self) -> dict:
        """Bring Atlas up, and say honestly what it can and cannot do."""
        report: dict[str, Any] = {"started_at": iso()}

        if self.store:
            await self.store.ping()
            report["indexes"] = await self.store.ensure_indexes()
            await self._hydrate_rate_limits()
            await self._load_spend()

        if self.settings.can_reach_app:
            self.client = TWSClient(
                self.settings.tws_api_url,
                email=self.settings.tws_email,
                password=self.settings.tws_password,
                token=self.settings.tws_token,
                cron_secret=self.settings.tws_cron_secret,
                timeout=self.settings.tws_timeout_secs,
                audit=self._audit_http,
            )
            try:
                self.identity = await self.client.verify_access()
                report["identity"] = self.identity
            except TWSError as e:
                # Not fatal: Atlas should boot and report that it is blind,
                # rather than crash-loop and tell nobody why.
                report["identity_error"] = str(e)
                log.error("atlas: cannot verify app access: %s", e)

        if self.settings.can_reason:
            self.engine = Engine(self.settings, dispatch=self._dispatch,
                                 spend_today=lambda: self._spend_today)

        report["readiness"] = self.settings.readiness()
        report["tools"] = len(registry)
        report["policy"] = self.policy.snapshot()
        return report

    async def close(self) -> None:
        self._running = False
        if self.client:
            await self.client.close()
        if self.store:
            await self.store.close()

    async def _hydrate_rate_limits(self) -> None:
        """Reload today's counts so a restart cannot reset a daily cap."""
        cutoff = (now() - timedelta(days=1)).isoformat()
        rows = await self.store["counters"].find(
            {"at": {"$gte": cutoff}}, {"_id": 0}).to_list(5000)
        by_bucket: dict[str, list] = {}
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r["at"]).replace("Z", "+00:00"))
            except Exception:
                continue
            by_bucket.setdefault(r["bucket"], []).append(ts.timestamp())
        for bucket, stamps in by_bucket.items():
            self.policy.limiter.hydrate(bucket, stamps)
        if by_bucket:
            log.info("atlas: restored rate-limit counts for %s", ", ".join(by_bucket))

    async def _load_spend(self) -> None:
        today = now().date().isoformat()
        self._spend_day = today
        rows = await self.store["cycles"].find(
            {"day": today}, {"_id": 0, "usage": 1}).to_list(500)
        self._spend_today = sum(float((r.get("usage") or {}).get("cost_usd") or 0)
                                for r in rows)

    def _roll_day(self) -> None:
        today = now().date().isoformat()
        if today != self._spend_day:
            self._spend_day, self._spend_today = today, 0.0

    # ------------------------------------------------------------- plumbing

    async def _audit_http(self, entry: dict) -> None:
        if self.store and entry.get("status") and entry["status"] >= 400:
            await self.store["actions"].insert_one({
                "id": str(uuid.uuid4()), "at": iso(), "kind": "http_error", **entry})

    async def _record_action(self, entry: dict) -> None:
        if not self.store:
            return
        # Counters are persisted by ToolContext.record_action, so every route
        # into dispatch records them, not just this one.
        doc = {"id": str(uuid.uuid4()), "at": iso(), "kind": "tool", **entry}
        await self.store["actions"].insert_one(doc)

    async def _queue_approval(self, entry: dict) -> str:
        if not self.store:
            return "unqueued"
        approval_id = str(uuid.uuid4())
        await self.store["approvals"].insert_one({
            "id": approval_id, "status": "pending", "created_at": iso(),
            "decided_at": None, "decided_by": None, **entry})
        return approval_id

    def _context(self, cycle_id: str, interactive: bool = False) -> ToolContext:
        return ToolContext(
            client=self.client, store=self.store, memory=self.memory,
            policy=self.policy, settings=self.settings, cycle_id=cycle_id,
            interactive=interactive,
            _recorder=self._record_action, _approver=self._queue_approval,
        )

    async def _dispatch(self, name: str, args: dict) -> Any:
        ctx = self._context(self._current_cycle_id, interactive=self._interactive)
        tool = registry.get(name)
        # Offline, only the self-management tools mean anything. Saying so
        # plainly lets the model fall back to planning instead of retrying a
        # call that cannot succeed.
        if self.client is None and tool is not None and tool.group != "self":
            return ("Atlas has no connection to the app right now, so this cannot run. "
                    "Only your own memory and planning tools work. Say so in your summary.")
        # registry.dispatch already converts an app refusal into a readable
        # result, so nothing is caught here. Anything that still escapes is a
        # genuine bug, and the engine records it as a failed tool call rather
        # than losing the whole turn.
        return await registry.dispatch(ctx, name, args)

    # ---------------------------------------------------------------- cycles

    async def run_cycle(self, kind: str = "work", *, note: str = "") -> dict:
        """One autonomous cycle, start to finish."""
        if self._cycle_lock.locked():
            return {"skipped": "a cycle is already running"}
        async with self._cycle_lock:
            return await self._run_cycle(kind, note)

    async def _run_cycle(self, kind: str, note: str) -> dict:
        self._roll_day()
        cycle_id = str(uuid.uuid4())
        self._current_cycle_id = cycle_id
        self._interactive = False
        started = now()

        record = {
            "id": cycle_id, "kind": kind, "status": "running",
            "started_at": started.isoformat(), "day": started.date().isoformat(),
            "autonomy": self.policy.autonomy, "sandbox": self.policy.sandbox,
            "note": note or None, "ended_at": None, "summary": None,
            "usage": None, "actions": 0, "error": None,
        }
        if self.store:
            await self.store["cycles"].insert_one(dict(record))

        if self.engine is None:
            return await self._close_cycle(record, error=(
                "ANTHROPIC_API_KEY is not set, so Atlas cannot think. The cycle did "
                "nothing."))
        if self.policy.kill_switch:
            return await self._close_cycle(record, summary=(
                "The kill switch is on. Atlas looked at nothing and did nothing."))

        try:
            system = prompts.build_system_prompt(
                autonomy=self.policy.autonomy,
                sandbox=self.policy.sandbox,
                kill_switch=self.policy.kill_switch,
                tool_doctrine=registry.doctrine(self.policy),
                cycle=kind,
            )
            opener = await self._opening_message(kind, note)
            result = await self.engine.run(
                system=system,
                messages=[{"role": "user", "content": opener}],
                tools=registry.specs(self.policy),
            )
            self._spend_today += result.usage.cost_usd
            record["usage"] = result.usage.as_dict()
            record["actions"] = len(result.actions)
            if result.refusal:
                record["refusal"] = result.refusal
            if result.truncated:
                record["truncated"] = True
            return await self._close_cycle(record, summary=result.text)
        except BudgetExceeded as e:
            return await self._close_cycle(record, error=str(e))
        except Exception as e:
            log.exception("atlas: cycle failed")
            return await self._close_cycle(record, error=f"{type(e).__name__}: {e}")

    async def _opening_message(self, kind: str, note: str) -> str:
        """The volatile half of the prompt: time, memory, and recent history.

        Deliberately in the message rather than the system prompt, so the long
        stable half stays cacheable across the day's cycles.
        """
        parts = [f"It is {now().strftime('%A %d %B %Y, %H:%M UTC')}."]

        if self.memory:
            remembered = await self.memory.render_for_prompt(
                query=note or kind, limit=25)
            parts.append("WHAT YOU REMEMBER\n\n" + remembered)

        if self.store:
            recent = await self.store["cycles"].find(
                {"status": "done", "summary": {"$ne": None}},
                {"_id": 0, "kind": 1, "started_at": 1, "summary": 1},
            ).sort("started_at", -1).to_list(3)
            if recent:
                lines = [f"  [{r['kind']} @ {r['started_at'][:16]}] "
                         f"{(r.get('summary') or '')[:400]}" for r in recent]
                parts.append("YOUR LAST FEW CYCLES\n\n" + "\n".join(lines))

            pending = await self.store["approvals"].count_documents({"status": "pending"})
            if pending:
                parts.append(
                    f"You have {pending} action(s) waiting on the owner's approval. "
                    f"Do not queue them again; chase them in your briefing if they matter.")

        if note:
            parts.append(f"THE OWNER ADDED\n\n{note}")

        parts.append("Begin. Look at the real numbers first.")
        return "\n\n".join(parts)

    async def _close_cycle(self, record: dict, *, summary: str = "",
                           error: str = "") -> dict:
        record["ended_at"] = iso()
        record["status"] = "failed" if error else "done"
        record["summary"] = summary or None
        record["error"] = error or None
        started = datetime.fromisoformat(record["started_at"])
        record["seconds"] = round((now() - started).total_seconds(), 1)
        if self.store:
            await self.store["cycles"].update_one(
                {"id": record["id"]}, {"$set": {k: v for k, v in record.items() if k != "id"}})
        self.last_cycle = record
        self._current_cycle_id = ""
        log.info("atlas: cycle %s %s in %ss (%d actions)", record["kind"],
                 record["status"], record["seconds"], record.get("actions") or 0)
        return record

    # ------------------------------------------------------------------ chat

    async def chat(self, message: str, *, history_limit: int = 12) -> dict:
        """The owner talking to Atlas. Same brain, same tools, same guardrails."""
        if self.engine is None:
            return {"reply": "ANTHROPIC_API_KEY is not set, so I cannot think yet.",
                    "actions": []}
        self._roll_day()
        cycle_id = str(uuid.uuid4())
        self._current_cycle_id = cycle_id
        self._interactive = True

        history: list = []
        if self.store:
            await self.store["chat"].insert_one({
                "id": str(uuid.uuid4()), "role": "user", "text": message,
                "created_at": iso()})
            rows = await self.store["chat"].find({}, {"_id": 0}) \
                .sort("created_at", -1).to_list(history_limit)
            rows.reverse()
            history = [{"role": r["role"], "content": r["text"]}
                       for r in rows if r.get("text")]

        system = prompts.build_system_prompt(
            autonomy=self.policy.autonomy, sandbox=self.policy.sandbox,
            kill_switch=self.policy.kill_switch,
            tool_doctrine=registry.doctrine(self.policy), cycle="chat")

        preface = f"It is {now().strftime('%A %d %B %Y, %H:%M UTC')}."
        if self.memory:
            preface += "\n\nWHAT YOU REMEMBER\n\n" + \
                await self.memory.render_for_prompt(query=message, limit=20)

        messages = history or [{"role": "user", "content": message}]
        # Prepend the context to the FIRST user turn rather than sending it as
        # its own message: two consecutive user turns is a shape worth avoiding,
        # and this keeps the conversation strictly alternating.
        messages = [dict(m) for m in messages]
        for m in messages:
            if m["role"] == "user":
                m["content"] = f"{preface}\n\n---\n\n{m['content']}"
                break

        try:
            result = await self.engine.run(system=system, messages=messages,
                                           tools=registry.specs(self.policy))
        except BudgetExceeded as e:
            return {"reply": str(e), "actions": []}

        self._spend_today += result.usage.cost_usd
        reply = result.text or "(no reply)"
        if self.store:
            await self.store["chat"].insert_one({
                "id": str(uuid.uuid4()), "role": "assistant", "text": reply,
                "created_at": iso(), "usage": result.usage.as_dict()})
        self._interactive = False
        return {"reply": reply, "actions": result.actions,
                "usage": result.usage.as_dict()}

    # ------------------------------------------------------------- scheduler

    def _due_kind(self, last_morning: str, last_evening: str) -> str:
        """Which cycle is owed right now.

        Morning and evening are once-a-day events; everything else is a work
        tick. Comparing on the date string means a restart cannot cause the
        morning plan to run twice.
        """
        n = now()
        today = n.date().isoformat()
        if n.hour >= self.settings.morning_brief_hour and last_morning != today:
            return "morning"
        if n.hour >= self.settings.evening_brief_hour and last_evening != today:
            return "evening"
        return "work"

    async def serve_forever(self) -> None:
        """Drive cycles on a timer until told to stop."""
        self._running = True
        last_morning = last_evening = ""
        log.info("atlas: scheduler running, tick every %ss", self.settings.tick_seconds)
        while self._running:
            try:
                if self.policy.kill_switch:
                    log.info("atlas: kill switch on, skipping tick")
                else:
                    kind = self._due_kind(last_morning, last_evening)
                    result = await self.run_cycle(kind)
                    if not result.get("skipped"):
                        today = now().date().isoformat()
                        if kind == "morning":
                            last_morning = today
                        elif kind == "evening":
                            last_evening = today
            except Exception:
                log.exception("atlas: scheduler tick failed")
            await asyncio.sleep(max(60, self.settings.tick_seconds))

    def stop(self) -> None:
        self._running = False
