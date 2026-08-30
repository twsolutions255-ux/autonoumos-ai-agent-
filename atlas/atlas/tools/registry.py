"""The tool surface: what Atlas can actually do, and the gate it does it through.

A tool here is four things bound together, and the binding is the point:

    name + schema     what the model sees and how it must call it
    policy            the risk class, autonomy rung and rate bucket
    handler           the code that talks to the app
    doc               a description written for a model that must choose well

Declaring the policy next to the handler means it is impossible to add a
capability without stating what it can cost you. There is no path from "new
endpoint" to "the agent can call it" that skips the risk decision.

Dispatch is the only way a tool runs. It checks policy, executes, records,
and rate-limits — in that order — so the audit log contains blocked attempts
as well as successful ones. What an autonomous agent *tried* to do is at
least as interesting as what it did.
"""
from __future__ import annotations

import inspect
import logging
import re
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from ..guardrails.policy import Decision, Outcome, Policy, Risk, ToolPolicy

log = logging.getLogger("atlas.tools")


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    policy: ToolPolicy
    handler: Callable[..., Awaitable[Any]]
    #: Grouping for the console and for the system prompt's tool doctrine.
    group: str = "general"

    def spec(self) -> dict:
        """The Anthropic tool definition.

        `strict` is on: these tools drive a live business, and a hallucinated
        extra field silently ignored is a worse outcome than a validation
        error the model can see and correct.
        """
        schema = dict(self.input_schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema.setdefault("required", [])
        schema["additionalProperties"] = False
        return {
            "name": self.name,
            "description": self.description.strip(),
            "input_schema": schema,
            "strict": True,
        }


class Registry:
    """Every tool Atlas has, and the one path by which they run."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ---------- declaration ----------

    def tool(self, name: str, *, description: str, schema: Optional[dict] = None,
             risk: Risk = Risk.READ, requires: Optional[str] = None,
             rate_bucket: Optional[str] = None, group: str = "general",
             estimate_cost: Optional[Callable[[dict], float]] = None,
             always_approve: bool = False):
        """Decorator registering one tool."""
        def wrap(fn: Callable[..., Awaitable[Any]]):
            if name in self._tools:
                raise ValueError(f"tool {name!r} is already registered")
            self._tools[name] = Tool(
                name=name,
                description=description,
                input_schema=schema or {"type": "object", "properties": {}, "required": []},
                policy=ToolPolicy(risk=risk, requires=requires, rate_bucket=rate_bucket,
                                  estimate_cost=estimate_cost, always_approve=always_approve),
                handler=fn,
                group=group,
            )
            return fn
        return wrap

    # ---------- inspection ----------

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list:
        return sorted(self._tools)

    def specs(self, policy: Optional[Policy] = None,
              groups: Optional[list] = None) -> list:
        """Tool definitions to send to the model.

        When a policy is supplied, tools the current autonomy rung can never
        run are omitted entirely rather than offered and refused. Showing a
        model a tool it is not allowed to use wastes a turn every time it
        reaches for it, and reads to the model as an unexplained failure.
        """
        out = []
        for t in self._tools.values():
            if groups and t.group not in groups:
                continue
            if policy is not None:
                probe = policy.check(t.name, t.policy, {})
                # NEEDS_APPROVAL still gets offered: the action is permitted,
                # it just routes through a human first, and the model should
                # know it can propose it.
                if probe.outcome is Outcome.DENY and probe.gate in ("autonomy", "sandbox"):
                    continue
            out.append(t.spec())
        return out

    def doctrine(self, policy: Optional[Policy] = None) -> str:
        """A grouped summary of the tool surface for the system prompt."""
        by_group: dict[str, list] = {}
        for t in self._tools.values():
            if policy is not None:
                probe = policy.check(t.name, t.policy, {})
                if probe.outcome is Outcome.DENY and probe.gate in ("autonomy", "sandbox"):
                    continue
            by_group.setdefault(t.group, []).append(t)
        lines = []
        for group in sorted(by_group):
            lines.append(f"\n{group.upper()}")
            for t in sorted(by_group[group], key=lambda x: x.name):
                first = t.description.strip().split("\n")[0]
                lines.append(f"  {t.name} — {first}")
        return "\n".join(lines).strip()

    # ---------- the one execution path ----------

    async def dispatch(self, ctx: "ToolContext", name: str, args: dict,
                       *, approved: bool = False) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            # Phrased so the model corrects itself rather than retrying.
            return (f"There is no tool called '{name}'. Available tools: "
                    f"{', '.join(self.names())}.")

        decision = ctx.policy.check(name, tool.policy, args, approved=approved)

        if decision.outcome is Outcome.NEEDS_APPROVAL:
            approval_id = await ctx.queue_approval(name, args, decision)
            await ctx.record_action(name, args, decision,
                                    result=f"queued as {approval_id}", tool=tool)
            return decision.as_tool_result() + f" (approval id: {approval_id})"

        if decision.outcome is Outcome.DENY:
            await ctx.record_action(name, args, decision, result="blocked", tool=tool)
            return decision.as_tool_result()

        try:
            result = await _invoke(tool.handler, ctx, args)
        except Exception as e:
            await ctx.record_action(name, args, decision, result=f"error: {e}",
                                    error=True, tool=tool)
            # A refusal or failure from the app is information, not a crash.
            # Converting it here rather than in the caller means EVERY route
            # into dispatch gets it — the autonomous loop, the owner's chat,
            # and the approval endpoint alike. The approval endpoint in
            # particular would otherwise return a 500 to the owner for what is
            # really just "the app said no".
            as_result = getattr(e, "as_tool_result", None)
            if callable(as_result):
                return as_result()
            raise

        # Counted only after it really happened, so a failed send does not
        # consume the hour's outreach allowance.
        ctx.policy.record(tool.policy)
        await ctx.record_action(name, args, decision, result=result, tool=tool)
        return result


async def _invoke(handler: Callable[..., Awaitable[Any]], ctx: "ToolContext",
                  args: dict) -> Any:
    """Call a handler with only the arguments it declares.

    The model occasionally sends a field the handler does not take. Dropping
    unknown keys here turns a TypeError into a successful call, which is the
    right trade for a system that is supposed to keep running unattended.
    """
    sig = inspect.signature(handler)
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
    if accepts_kwargs:
        clean = dict(args)
    else:
        allowed = {n for n, p in sig.parameters.items() if n != "ctx"}
        clean = {k: v for k, v in args.items() if k in allowed}
    return await handler(ctx, **clean)


@dataclass
class ToolContext:
    """Everything a tool needs, handed to it rather than imported by it.

    Passing this explicitly is what makes the tools testable without a live
    app, a live database or a live model behind them.
    """
    client: Any
    store: Any
    memory: Any
    policy: Policy
    settings: Any
    cycle_id: str = ""
    #: Set when the caller is the owner in chat rather than the autonomous
    #: cycle. Some tools phrase their output differently for a person.
    interactive: bool = False
    _recorder: Optional[Callable[[dict], Awaitable[None]]] = None
    _approver: Optional[Callable[[dict], Awaitable[str]]] = None

    async def record_action(self, name: str, args: dict, decision: Decision,
                            result: Any = None, error: bool = False,
                            tool: Optional[Tool] = None) -> None:
        # Rate-limit events are persisted here rather than by the caller, so a
        # restart cannot hand back a fresh daily allowance no matter which
        # route into dispatch was used.
        if (self.store is not None and tool is not None
                and tool.policy.rate_bucket
                and decision.outcome is Outcome.ALLOW and not error):
            try:
                await self.store["counters"].insert_one({
                    "id": str(uuid.uuid4()), "bucket": tool.policy.rate_bucket,
                    "tool": name, "at": _now_iso()})
            except Exception:
                log.exception("atlas: could not persist a rate-limit counter")

        if self._recorder is None:
            return
        preview = redact_text(result if isinstance(result, str) else str(result))
        await self._recorder({
            "cycle_id": self.cycle_id,
            "tool": name,
            "args": _redact(args),
            "outcome": decision.outcome.value,
            "gate": decision.gate,
            "reason": decision.reason,
            "error": error,
            "result_preview": (preview or "")[:1000],
        })

    async def queue_approval(self, tool: str, args: dict, decision: Decision) -> str:
        if self._approver is None:
            return "unqueued"
        return await self._approver({
            "cycle_id": self.cycle_id,
            "tool": tool,
            "args": args,
            "reason": decision.reason,
        })


#: Names whose values must never reach a log. Small on purpose: the audit trail
#: is worth much more than the marginal privacy of a lead's phone number, and
#: over-redaction makes an audit log you cannot reason from.
_SECRET_KEYS = {"password", "api_key", "apikey", "token", "access_token",
                "secret", "authorization", "site_key", "agent_key"}

#: Credential shapes that appear in RESULT bodies rather than in arguments.
#: This is not hypothetical: GET /admin/clients returns every tenant's
#: plaintext receptionist api_key in its response, so an agent that logged raw
#: results would quietly build a file of live customer credentials inside its
#: own audit collection. Redaction therefore runs over results too.
_SECRET_VALUE_RE = re.compile(
    r"(twsagent_[A-Za-z0-9_\-]+)"          # agent API keys
    r"|(\bsk-[A-Za-z0-9_\-]{16,})"          # provider keys
    r"|(\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+)"  # JWTs
)

#: Matches "api_key": "<value>" inside a serialised result body.
_SECRET_FIELD_RE = re.compile(
    r'("(?:api_key|apiKey|site_key|password|access_token|token|secret)"\s*:\s*)"[^"]*"',
    re.IGNORECASE)


def _redact(args: dict) -> dict:
    return {k: ("***" if k.lower() in _SECRET_KEYS else v) for k, v in (args or {}).items()}


def redact_text(text: str) -> str:
    """Strip credentials out of anything on its way to storage."""
    if not text:
        return text
    text = _SECRET_FIELD_RE.sub(r'\1"***"', text)
    return _SECRET_VALUE_RE.sub("***", text)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


registry = Registry()
