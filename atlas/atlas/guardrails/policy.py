"""What Atlas is allowed to do, decided before it does it.

An autonomous agent is only as trustworthy as the layer that sits between its
intention and the outside world. That layer is this file. It answers one
question — *may this specific call, with these specific arguments, happen right
now?* — and it answers it the same way whether the caller is the autonomous
cycle, the owner's chat, or a replayed approval.

Four independent gates, checked in this order. Order matters: the cheapest and
most absolute checks come first, so a killed agent burns no budget deciding it
is killed.

  1. KILL SWITCH   — one flag stops everything, mid-turn.
  2. AUTONOMY      — is this class of action unlocked at the configured rung?
  3. RATE LIMIT    — has this channel already done enough this hour/day?
  4. APPROVAL      — is this individually large enough to need a human?

A tool that passes all four executes. Anything else returns a decision the
model can read and reason about, because a refusal the agent understands
("staged instead of sent, awaiting release") produces better behaviour than a
raw exception it will simply retry.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Risk(str, Enum):
    """What a tool can cost you if it is wrong.

    Not a severity score — a description of who absorbs the mistake.
    """
    #: Nobody outside sees anything. Worst case is a wasted call.
    READ = "read"
    #: Writes only into Atlas's own memory/plan. Reversible, invisible outside.
    INTERNAL = "internal"
    #: A human on the team reads something Atlas wrote. Embarrassing if wrong.
    INTERNAL_COMMS = "internal_comms"
    #: Work is queued for a person to release. Nothing leaves the building yet.
    STAGE = "stage"
    #: A person outside the company is contacted. Cannot be unsent.
    EXTERNAL_COMMS = "external_comms"
    #: Money moves, or a price/obligation changes.
    MONEY = "money"
    #: Destroys or replaces something with no undo.
    IRREVERSIBLE = "irreversible"


#: The autonomy rung each risk class requires. This mapping is the whole
#: safety model in six lines, which is deliberate: it should be readable in
#: one sitting by someone deciding whether to turn autopilot on.
RISK_REQUIRES = {
    Risk.READ: "observe",
    Risk.INTERNAL: "recommend",
    Risk.INTERNAL_COMMS: "assist",
    Risk.STAGE: "assist",
    Risk.EXTERNAL_COMMS: "operate",
    Risk.MONEY: "autopilot",
    Risk.IRREVERSIBLE: "autopilot",
}

#: Risk classes that do nothing at all while SANDBOX is on. Reads and internal
#: bookkeeping still run — a sandboxed Atlas should still think, plan and
#: brief, otherwise you cannot evaluate it before trusting it.
SANDBOX_BLOCKS = {Risk.EXTERNAL_COMMS, Risk.MONEY, Risk.IRREVERSIBLE, Risk.STAGE}


class Outcome(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class Decision:
    outcome: Outcome
    reason: str
    #: Which gate produced it — for the audit log and the console.
    gate: str = ""
    #: What the owner would have to change to make this allowed.
    remedy: str = ""

    @property
    def allowed(self) -> bool:
        return self.outcome is Outcome.ALLOW

    def as_tool_result(self) -> str:
        """What the model sees. Written to redirect, not just to refuse."""
        if self.outcome is Outcome.NEEDS_APPROVAL:
            return (f"NOT DONE — held for the owner's approval. {self.reason} "
                    f"It is now in the approvals queue; say so in your summary and "
                    f"carry on with the rest of your plan rather than retrying.")
        return (f"NOT DONE — nothing happened. Blocked by policy ({self.gate}). {self.reason}"
                + (f" To allow it: {self.remedy}" if self.remedy else "")
                + " Do not retry this call. Choose a different action, and say in your "
                  "summary that this step did not happen.")


@dataclass
class RateLimit:
    """A rolling-window cap on one named channel."""
    name: str
    limit: int
    window_secs: int

    def describe(self) -> str:
        unit = "hour" if self.window_secs == 3600 else (
            "day" if self.window_secs == 86400 else f"{self.window_secs}s")
        return f"{self.limit} per {unit}"


@dataclass
class ToolPolicy:
    """The policy half of a tool definition."""
    risk: Risk
    #: Overrides RISK_REQUIRES when a specific tool is safer or more dangerous
    #: than its class (e.g. staging a call batch is STAGE, but *releasing* one
    #: is EXTERNAL_COMMS even though both live in the cold-call module).
    requires: Optional[str] = None
    rate_bucket: Optional[str] = None
    #: Returns the dollar impact of a specific call, so the approval threshold
    #: can be applied to arguments rather than to the tool as a whole.
    estimate_cost: Optional[Callable[[dict], float]] = None
    #: Always ask, whatever the numbers say.
    always_approve: bool = False

    def required_level(self) -> str:
        return self.requires or RISK_REQUIRES[self.risk]


class RateLimiter:
    """In-memory rolling-window counter, with a pluggable persistent backing.

    Kept in memory because the check runs on every tool dispatch and must be
    fast; `hydrate()` reloads counts from storage at boot so a restart cannot
    be used — accidentally or by a confused agent — to reset a daily cap.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[float]] = {}

    def hydrate(self, bucket: str, timestamps: list[float]) -> None:
        self._events.setdefault(bucket, []).extend(timestamps)

    def _prune(self, bucket: str, window: int, now: float) -> list[float]:
        cutoff = now - window
        kept = [t for t in self._events.get(bucket, []) if t >= cutoff]
        self._events[bucket] = kept
        return kept

    def would_exceed(self, rl: RateLimit, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return len(self._prune(rl.name, rl.window_secs, now)) >= rl.limit

    def used(self, rl: RateLimit, now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        return len(self._prune(rl.name, rl.window_secs, now))

    def record(self, bucket: str, now: Optional[float] = None) -> None:
        self._events.setdefault(bucket, []).append(now if now is not None else time.time())

    def snapshot(self, limits: dict[str, RateLimit]) -> dict:
        return {name: {"used": self.used(rl), "limit": rl.limit,
                       "window": rl.describe()}
                for name, rl in limits.items()}


class Policy:
    """The gate. One instance, consulted before every tool dispatch."""

    def __init__(self, settings, limiter: Optional[RateLimiter] = None):
        self.settings = settings
        self.limiter = limiter or RateLimiter()
        self.limits: dict[str, RateLimit] = {
            "outreach": RateLimit("outreach", settings.max_outreach_per_hour, 3600),
            "chat": RateLimit("chat", settings.max_chat_messages_per_hour, 3600),
            "calls": RateLimit("calls", settings.max_calls_staged_per_day, 86400),
        }
        #: Set by the runtime when the owner flips the switch at runtime, so a
        #: stop does not require a redeploy.
        self.kill_switch = bool(settings.kill_switch)
        self.sandbox = bool(settings.sandbox)
        self.autonomy = settings.autonomy

    # ---------- runtime overrides ----------

    def set_autonomy(self, level: str) -> None:
        from ..config import AUTONOMY_LEVELS
        if level not in AUTONOMY_LEVELS:
            raise ValueError(f"unknown autonomy level {level!r}")
        self.autonomy = level

    def _allows(self, required: str) -> bool:
        from ..config import AUTONOMY_LEVELS
        return AUTONOMY_LEVELS.index(self.autonomy) >= AUTONOMY_LEVELS.index(required)

    # ---------- the gate ----------

    def check(self, tool_name: str, pol: ToolPolicy, args: dict,
              *, approved: bool = False) -> Decision:
        # 1. Kill switch — absolute, and deliberately ahead of everything else.
        if self.kill_switch and pol.risk is not Risk.READ:
            return Decision(
                Outcome.DENY,
                "Atlas is stopped. The kill switch is on, so it is doing nothing "
                "that touches the business.",
                gate="kill_switch",
                remedy="turn the kill switch off in the Atlas console")

        # 2. Autonomy rung.
        required = pol.required_level()
        if not self._allows(required):
            from ..config import AUTONOMY_DESCRIPTIONS
            return Decision(
                Outcome.DENY,
                f"'{tool_name}' needs autonomy level '{required}' and Atlas is set to "
                f"'{self.autonomy}' ({AUTONOMY_DESCRIPTIONS[self.autonomy]})",
                gate="autonomy",
                remedy=f"raise ATLAS_AUTONOMY to '{required}' or higher")

        # 2b. Sandbox. Checked with autonomy because it is the same kind of
        # statement: a deliberate restriction on what may leave the building.
        if self.sandbox and pol.risk in SANDBOX_BLOCKS:
            return Decision(
                Outcome.DENY,
                f"Sandbox mode is on, so '{tool_name}' ({pol.risk.value}) is held back and "
                f"nothing reaches anyone outside the company.",
                gate="sandbox",
                remedy="set ATLAS_SANDBOX=false when you are ready for it to act for real")

        # 3. Rate limit.
        if pol.rate_bucket:
            rl = self.limits.get(pol.rate_bucket)
            if rl and self.limiter.would_exceed(rl):
                return Decision(
                    Outcome.DENY,
                    f"The {pol.rate_bucket} cap is spent: {rl.describe()} already used. "
                    f"This is the throttle working, not a failure.",
                    gate="rate_limit",
                    remedy=f"raise the {pol.rate_bucket} cap, or wait for the window to roll")

        # 4. Approval. Last, so an action that would be denied outright is
        # never queued for a human to approve pointlessly.
        if not approved:
            if pol.always_approve:
                return Decision(Outcome.NEEDS_APPROVAL,
                                f"'{tool_name}' always needs the owner to say yes.",
                                gate="approval")
            if pol.estimate_cost:
                try:
                    impact = float(pol.estimate_cost(args) or 0.0)
                except Exception:
                    # An estimator that throws must fail closed. Guessing "zero"
                    # on a money tool is exactly the wrong direction to be wrong in.
                    return Decision(Outcome.NEEDS_APPROVAL,
                                    f"Could not estimate the impact of '{tool_name}', "
                                    f"so it is being treated as large.",
                                    gate="approval")
                if impact >= self.settings.approval_threshold_usd:
                    return Decision(
                        Outcome.NEEDS_APPROVAL,
                        f"'{tool_name}' has an estimated impact of ${impact:,.0f}, at or over "
                        f"the ${self.settings.approval_threshold_usd:,.0f} approval threshold.",
                        gate="approval")

        return Decision(Outcome.ALLOW, "permitted", gate="")

    def record(self, pol: ToolPolicy) -> None:
        """Count a call that actually happened. Only ever called post-success."""
        if pol.rate_bucket:
            self.limiter.record(pol.rate_bucket)

    def snapshot(self) -> dict:
        from ..config import AUTONOMY_DESCRIPTIONS
        return {
            "autonomy": self.autonomy,
            "autonomy_means": AUTONOMY_DESCRIPTIONS[self.autonomy],
            "sandbox": self.sandbox,
            "kill_switch": self.kill_switch,
            "approval_threshold_usd": self.settings.approval_threshold_usd,
            "rate_limits": self.limiter.snapshot(self.limits),
        }
