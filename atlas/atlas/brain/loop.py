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
from datetime import datetime, timedelta, timezone
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
from ..tools import clientcare, comms, growth, money, observe, reflect  # noqa: E402,F401


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
        spent = sum(float((r.get("usage") or {}).get("cost_usd") or 0) for r in rows)
        # Chat costs money too, and it is not a cycle. Leaving it out would let
        # a redeploy hand Atlas a fresh budget it had already spent talking.
        chats = await self.store["chat"].find(
            {"created_at": {"$gte": today}}, {"_id": 0, "usage": 1}).to_list(500)
        spent += sum(float((r.get("usage") or {}).get("cost_usd") or 0) for r in chats)
        self._spend_today = spent

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
            # Morning and evening plan and review on the strong model. A work
            # cycle takes one step of an existing plan, and the cheap model
            # is enough for that -- the plan is the thinking, done already.
            model, cap = self._model_for(kind)
            result = await self.engine.run(
                system=system,
                messages=[{"role": "user", "content": opener}],
                tools=registry.specs(self.policy),
                model=model, max_iterations=cap,
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
            # Loud on purpose. This was caught and closed with an error string
            # nobody reads, so an Atlas that had spent its day's allowance at
            # 9am was indistinguishable in the logs from an Atlas with nothing
            # to do -- "cycle work failed in 1.3s (0 actions)" and no more.
            # The owner then asks why it is quiet and there is nothing to find.
            log.error("atlas: STOPPED BY THE DAILY BUDGET -- %s. No further "
                      "cycle will do anything until midnight %s, or until "
                      "ATLAS_DAILY_LLM_BUDGET_USD is raised above what has "
                      "already been spent today.",
                      e, getattr(self.settings, "timezone", "local time"))
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

            refused = await self._recent_refusals()
            if refused:
                parts.append("ALREADY REFUSED -- DO NOT TRY THESE AGAIN\n\n" + refused)

        if note:
            parts.append(f"THE OWNER ADDED\n\n{note}")

        parts.append("Begin. Look at the real numbers first.")
        return "\n\n".join(parts)

    async def _recent_refusals(self, hours: int = 48) -> str:
        """Policy denials from the last two days, one line per tool.

        The approvals queue already gets a "do not queue them again" line.
        Denials did not, and at the `recommend` rung every action is a
        denial -- so each work cycle read the same numbers, found the same
        gap, called the same tool, was refused the same way, and recommended
        the same thing. Not a memory fault: the world cannot change until the
        owner acts, and nobody was telling the model that it had already
        asked. The three-cycle summaries above are prose, and a model does
        not reliably notice its own repetition in prose. This is a list.
        """
        cutoff = now() - timedelta(hours=hours)
        rows = await self.store["actions"].find(
            {"kind": "tool", "outcome": "deny"},
            {"_id": 0, "tool": 1, "gate": 1, "reason": 1, "at": 1},
        ).sort("at", -1).to_list(60)
        seen: dict = {}
        for r in rows:
            try:
                when = datetime.fromisoformat(r["at"])
            except (KeyError, TypeError, ValueError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when < cutoff or r.get("tool") in seen:
                continue
            seen[r["tool"]] = r
        if not seen:
            return ""
        lines = [f"  - {t} (blocked by {r.get('gate') or 'policy'}): "
                 f"{(r.get('reason') or '')[:200]}" for t, r in seen.items()]
        autonomy = str(getattr(self.settings, "autonomy", "") or "")
        tail = ""
        if autonomy == "recommend":
            tail = ("\n\nYour autonomy is 'recommend': you cannot act, only "
                    "recommend. If your last cycles already recommended the same "
                    "thing and the numbers have not moved, do not recommend it "
                    "again. Say what is NEW since then, or say that nothing has "
                    "changed and stop -- a short cycle is the right answer.")
        return ("Nothing about your authority has changed since these were "
                "refused:\n" + "\n".join(lines) + tail)

    #: The one cycle a day that genuinely plans. Everything else either takes
    #: one step of a plan that already exists or writes up what happened, and
    #: neither needs the expensive model.
    #:
    #: Measured on 2026-09-02, the day the owner watched $5 of DeepSeek credit
    #: become $2: four cycles, 11 to 19 actions each, roughly $0.75 a cycle.
    #: The driver is not how often cycles run -- it is that every tool
    #: iteration re-sends the whole conversation plus 87 tool specifications,
    #: so cost grows with the SQUARE of the iteration cap and linearly with
    #: the model's input rate. The evening review was on the pro model for no
    #: reason anybody could name: it reads numbers and writes a summary.
    PLANNING_CYCLES = ("morning",)

    def _model_for(self, kind: str) -> tuple:
        """Which model and iteration cap a cycle kind gets.

        Only the morning plan gets the strong model. Work and evening cycles
        get the fast one at half the cap: a work cycle advances a plan that
        already exists, and an evening review reports what happened.
        """
        if kind in self.PLANNING_CYCLES:
            return self.settings.model, self.settings.max_tool_iterations
        return (self.settings.fast_model,
                max(6, self.settings.max_tool_iterations // 2))

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
            # Names the key that would ACTUALLY fix it. This said
            # ANTHROPIC_API_KEY unconditionally, which on a DeepSeek
            # deployment is a variable the owner could set correctly and
            # still get the same refusal. The same bug was fixed once in
            # reasoning_key_needed and left standing here, which is why the
            # helper is reused rather than the string rewritten.
            return {"reply": "I cannot think yet — %s."
                             % self.settings.reasoning_key_needed,
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

    def _due_kind(self, last_morning: str, last_evening: str,
                  last_work_at: Optional[datetime] = None) -> str:
        """Which cycle is owed right now.

        Morning and evening are once-a-day events; everything else is a work
        tick. Two things this has to get right, both learned the hard way:

        * **Morning is a window, not a threshold.** An unbounded `hour >=
          morning_hour` means a process that starts at 22:00 writes a "morning
          plan" for a day that is over, and then never runs the evening review
          because morning matched first.
        * **The "already ran today" marks must survive a restart.** Held only
          in memory, every redeploy re-runs the morning plan and re-briefs the
          owner. They are therefore read back from the cycle history at boot
          (see `_last_cycle_days`), not just tracked in the loop.
        """
        n = now()
        today = n.date().isoformat()
        morning_h = self.settings.morning_brief_hour
        evening_h = self.settings.evening_brief_hour
        if morning_h <= n.hour < evening_h and last_morning != today:
            return "morning"
        if n.hour >= evening_h and last_evening != today:
            return "evening"
        # Work is rationed. Morning and evening are once a day by nature;
        # without this, every tick between them was a full planning turn on
        # the expensive model, and the numbers rarely move hour to hour.
        # "" means nothing is owed and the loop just sleeps.
        if last_work_at is not None:
            since = (n - last_work_at).total_seconds() / 3600.0
            if since < self.settings.work_every_hours:
                return ""
        return "work"

    async def _last_cycle_days(self) -> tuple:
        """The days the last morning and evening cycles actually ran.

        Read from storage so a restart does not repeat either of them.
        """
        if not self.store:
            return "", ""
        out = []
        for kind in ("morning", "evening"):
            row = await self.store["cycles"].find_one(
                {"kind": kind, "status": "done"}, {"_id": 0, "day": 1},
                sort=[("started_at", -1)])
            out.append((row or {}).get("day") or "")
        return tuple(out)

    async def _last_work_at(self) -> Optional[datetime]:
        """When the last work cycle STARTED, from storage, so a redeploy does
        not hand Atlas a fresh work cycle it has just had."""
        if not self.store:
            return None
        row = await self.store["cycles"].find_one(
            {"kind": "work", "status": {"$in": ["done", "failed"]}},
            {"_id": 0, "started_at": 1}, sort=[("started_at", -1)])
        try:
            when = datetime.fromisoformat(row["started_at"])
        except (TypeError, KeyError, ValueError):
            return None
        return when if when.tzinfo else when.replace(tzinfo=timezone.utc)

    async def serve_forever(self) -> None:
        """Drive cycles on a timer until told to stop."""
        self._running = True
        last_morning, last_evening = await self._last_cycle_days()
        last_work_at = await self._last_work_at()
        log.info("atlas: scheduler running, tick every %ss "
                 "(last morning: %s, last evening: %s)",
                 self.settings.tick_seconds, last_morning or "never",
                 last_evening or "never")
        while self._running:
            try:
                if self.policy.kill_switch:
                    log.info("atlas: kill switch on, skipping tick")
                else:
                    kind = self._due_kind(last_morning, last_evening, last_work_at)
                    if not kind:
                        log.info("atlas: nothing owed this tick; next work cycle "
                                 "in %.1fh", self.settings.work_every_hours
                                 - (now() - last_work_at).total_seconds() / 3600.0)
                    else:
                        result = await self.run_cycle(kind)
                        if not result.get("skipped"):
                            today = now().date().isoformat()
                            if kind == "morning":
                                last_morning = today
                            elif kind == "evening":
                                last_evening = today
                            else:
                                last_work_at = now()
            except Exception:
                log.exception("atlas: scheduler tick failed")
            await asyncio.sleep(max(60, self.settings.tick_seconds))

    def may_drive_app_jobs(self) -> tuple:
        """May the job driver fire right now, and if not, why not?

        Split out from the loop so it is testable without waiting out a real
        timer — a guard that is only exercised by sleeping is a guard nobody
        ever actually tests.
        """
        if self.policy.kill_switch:
            return False, "the kill switch is on"
        if self.policy.sandbox:
            return False, "sandbox mode is on"
        # Same rung the equivalent tool needs: these jobs call and email real
        # people, so they must never outrun Atlas's own authority.
        if not self.policy._allows("operate"):
            return False, f"autonomy is '{self.policy.autonomy}', below 'operate'"
        if self.client is None:
            return False, "there is no connection to the app"
        return True, ""

    async def drive_app_jobs_forever(self) -> None:
        """Be the clock the app does not have.

        TWS has no task queue. Its in-process scheduler is off by default, and
        on a free instance it sleeps overnight — which is exactly when the
        nightly digest and the audit window need it. Its five cron jobs run
        from GitHub Actions against a hardcoded hostname that breaks the day
        the service is renamed.

        Atlas is already a long-lived process that holds the cron secret, so it
        can be the reliable driver. Two jobs are worth a tight loop:
        speed-to-lead, because a lead called in minutes converts far better
        than one called tomorrow, and workflows, because that is where
        scheduled follow-ups fire.

        Deliberately NOT model-gated. This is plumbing that should run on time
        every time, not a decision to be re-reasoned every five minutes. It is
        still gated by the kill switch, sandbox and autonomy, because it does
        reach the outside world.
        """
        if not self.settings.can_run_jobs:
            log.warning("atlas: asked to drive the app's jobs but no cron secret is "
                        "set; not starting")
            return
        log.info("atlas: driving the app's speed-to-lead and workflow jobs every %ss",
                 self.settings.app_job_seconds)
        while self._running:
            await asyncio.sleep(max(60, self.settings.app_job_seconds))
            allowed, _why = self.may_drive_app_jobs()
            if not allowed:
                continue
            for path in ("/internal/speed-to-lead/drain", "/internal/workflows/run"):
                try:
                    await self.client.run_internal_job(path)
                except TWSError as e:
                    log.warning("atlas: job %s failed: %s", path, e)
                except Exception:
                    log.exception("atlas: job %s raised", path)

    def stop(self) -> None:
        self._running = False
