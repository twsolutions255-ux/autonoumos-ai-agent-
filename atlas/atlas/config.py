"""Atlas configuration.

Every setting is read from the environment once, at import, and exposed as a
frozen `settings` object. Two rules govern this file:

1. **Nothing fakes a capability.** If a key is missing, the feature it gates is
   reported as off — it never degrades into a stub that looks like it worked.
   The TWS app learned this the hard way (see its README: an unset
   PUBLIC_BASE_URL reported provisioning success and silently delivered no
   calls). An autonomous agent makes that failure mode far worse, because
   nobody is watching when it happens.

2. **The dangerous defaults are the safe ones.** SANDBOX starts on, autonomy
   starts at the lowest tier, and every spend cap has a real number in it. A
   misconfigured deploy must be inert, not expensive.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        raise RuntimeError(
            f"{name}={raw!r} is not a number. Atlas refuses to boot with an "
            f"unreadable limit rather than fall back to a default you did not choose."
        )


def _float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"{name}={raw!r} is not a number.")


def _bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"{name}={raw!r} is not a boolean.")


# Autonomy is a ladder, not a switch. Each rung is a strictly larger set of
# permitted actions, and the guardrail layer resolves a tool's required rung
# against the one configured here. Named rather than numbered in the env so a
# typo cannot silently promote the agent.
AUTONOMY_LEVELS = ("observe", "recommend", "assist", "operate", "autopilot")

#: What each rung means, quoted verbatim into the agent's own system prompt so
#: its self-description can never drift from what the code enforces.
AUTONOMY_DESCRIPTIONS = {
    "observe": "Read everything. Change nothing. Write no messages to anyone.",
    "recommend": "Read everything, and write a plan and a briefing to the owner. Nothing in the business changes — the only writes are into Atlas's own memory.",
    "assist": "Everything in recommend, plus change data in the app (scan markets, build, analyse), talk to the team and the AI employees, and queue work for a human to release.",
    "operate": "Everything in assist, plus run the acquisition pipeline end to end: discover, score, build, and send outreach.",
    "autopilot": "Everything in operate, plus money-adjacent and irreversible actions, each still bounded by its own cap.",
}


@dataclass(frozen=True)
class Settings:
    # ---------- identity ----------
    #: The handle Atlas posts under in TWS team chat. Deliberately matches the
    #: existing AI-employee convention (`ai-viktor`, `ai-nadia`, ...) so the
    #: app's `is_ai` rendering and mention regex treat it like a colleague.
    handle: str = "atlas"
    display_name: str = "Atlas"
    title: str = "Chief of Staff"

    # ---------- the app Atlas runs ----------
    tws_api_url: str = ""
    tws_email: str = ""
    tws_password: str = ""
    #: Optional pre-minted bearer token. Skips the login round-trip; useful for
    #: a short-lived task runner that should not hold a password.
    tws_token: str = ""
    tws_timeout_secs: float = 45.0
    #: The app's INTERNAL_CRON_SECRET. Optional, and gated separately from
    #: everything else: it is the only credential that reaches the /internal/*
    #: endpoints, one of which (the AI-employee acting run) is the sole route
    #: by which this system can decide to phone a stranger. Unset simply means
    #: Atlas cannot drive the scheduled jobs on demand.
    tws_cron_secret: str = ""

    # ---------- reasoning ----------
    anthropic_api_key: str = ""
    #: Set when ATLAS_MODEL or ATLAS_FAST_MODEL is a DeepSeek model. Roughly
    #: two orders of magnitude cheaper than Claude, which is what makes an
    #: hourly agent affordable at all.
    deepseek_api_key: str = ""
    model: str = "claude-opus-5"
    #: Cheap model for high-volume mechanical calls (scoring one lead, drafting
    #: one line). The expensive model plans; this one grinds.
    fast_model: str = "claude-haiku-4-5"
    effort: str = "high"
    max_tokens: int = 16000
    #: Ceiling on tool-calling round trips inside a single reasoning turn. A
    #: runaway loop is the failure mode that costs real money unattended.
    max_tool_iterations: int = 40

    # ---------- storage ----------
    mongo_url: str = ""
    mongo_db: str = "tws"
    #: Atlas keeps its own collections inside the app's database so a single
    #: Atlas deploy needs no second piece of infrastructure. The prefix is what
    #: keeps them from ever colliding with the app's own.
    collection_prefix: str = "atlas_"

    # ---------- autonomy and safety ----------
    autonomy: str = "recommend"
    sandbox: bool = True
    #: Hard stop. Checked before every single tool dispatch, not just at the top
    #: of a cycle, so flipping it takes effect inside a running turn.
    kill_switch: bool = False

    daily_llm_budget_usd: float = 25.0
    #: Outbound actions per rolling hour, per channel. Independent of the LLM
    #: budget: a cheap model can still send a thousand emails.
    max_outreach_per_hour: int = 40
    max_calls_staged_per_day: int = 200
    max_chat_messages_per_hour: int = 30
    #: An action whose estimated dollar impact exceeds this always requires the
    #: owner to approve, at every autonomy level including autopilot.
    approval_threshold_usd: float = 250.0

    # ---------- cadence ----------
    tick_seconds: int = 900
    #: How often a WORK cycle may run. The tick decides how often Atlas
    #: checks whether anything is owed; this decides how often the expensive
    #: "advance the plan" reasoning actually happens. Hourly ticks with no
    #: cap meant ~22 planner calls a day that mostly re-read the same numbers.
    work_every_hours: float = 4.0
    #: Drive the app's own scheduled jobs on a timer, independently of the
    #: reasoning cycle. Off by default: these jobs call strangers and email
    #: clients, so switching them on is an outward-facing decision.
    drive_app_jobs: bool = False
    app_job_seconds: int = 300
    morning_brief_hour: int = 7
    evening_brief_hour: int = 19
    timezone: str = "America/New_York"

    # ---------- console ----------
    console_api_key: str = ""
    cors_origins: str = ""
    port: int = 8090
    log_level: str = "INFO"

    # ---------- derived ----------
    missing: tuple = field(default_factory=tuple)

    # -- capability probes: each answers "is this actually wired up?" --
    @property
    def can_reason(self) -> bool:
        """Whether the models Atlas is CONFIGURED to use have keys.

        Checked against the configured models rather than against Anthropic
        alone. A DeepSeek deployment with no ANTHROPIC_API_KEY reasons
        perfectly well, and reporting it as unable to think would be the same
        misleading-red as a green light on a broken integration.
        """
        models = {self.model, self.fast_model}
        if any(str(m).startswith("deepseek") for m in models) and not self.deepseek_api_key:
            return False
        if any(not str(m).startswith("deepseek") for m in models) and not self.anthropic_api_key:
            return False
        return True

    @property
    def reasoning_key_needed(self) -> str:
        """WHICH key is missing, not which key used to be the only option.

        can_reason was made vendor-aware and this hint was left hard-coded to
        ANTHROPIC_API_KEY, so a DeepSeek deployment with no DeepSeek key logged
        "reasoning is OFF (set ANTHROPIC_API_KEY)" -- naming a variable that
        would not have fixed it. The check was right and the instruction was
        wrong, and the instruction is the half a person acts on.
        """
        models = {self.model, self.fast_model}
        missing = []
        if any(str(m).startswith("deepseek") for m in models) and not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if any(not str(m).startswith("deepseek") for m in models) and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not missing:
            # Nothing is missing. Name what is in use, so a green line still
            # says which vendor is answering.
            return " + ".join(sorted(models))
        return " and ".join(missing) + " (for %s)" % ", ".join(sorted(models))

    @property
    def can_reach_app(self) -> bool:
        return bool(self.tws_api_url and (self.tws_token or (self.tws_email and self.tws_password)))

    @property
    def can_remember(self) -> bool:
        return bool(self.mongo_url)

    @property
    def can_run_jobs(self) -> bool:
        return bool(self.tws_cron_secret)

    @property
    def autonomy_rank(self) -> int:
        return AUTONOMY_LEVELS.index(self.autonomy)

    def allows(self, required_level: str) -> bool:
        """Is the configured rung at least `required_level`?"""
        if required_level not in AUTONOMY_LEVELS:
            raise ValueError(f"unknown autonomy level {required_level!r}")
        return self.autonomy_rank >= AUTONOMY_LEVELS.index(required_level)

    def readiness(self) -> dict:
        """What works, what does not, and what to set to fix it.

        Surfaced on /health and in the console. The point is that a half-
        configured Atlas says so in one place, rather than each subsystem
        failing separately at 3am.
        """
        checks = [
            ("reasoning", self.can_reason, self.reasoning_key_needed,
             "Atlas cannot think. Every cycle will no-op."),
            ("app_access", self.can_reach_app, "TWS_API_URL + TWS_EMAIL/TWS_PASSWORD (or TWS_TOKEN)",
             "Atlas cannot see or touch the business. It will run blind."),
            ("memory", self.can_remember, "MONGO_URL",
             "Atlas forgets everything between cycles: no plan, no history, no learning."),
            ("console_auth", bool(self.console_api_key), "ATLAS_CONSOLE_API_KEY",
             "The control API is unauthenticated and refuses to serve."),
        ]
        return {
            "ready": all(ok for _, ok, _, _ in checks),
            "autonomy": self.autonomy,
            "autonomy_means": AUTONOMY_DESCRIPTIONS[self.autonomy],
            "sandbox": self.sandbox,
            "kill_switch": self.kill_switch,
            "checks": [
                {"name": n, "ok": ok, "set": var, "off_means": consequence}
                for n, ok, var, consequence in checks
            ],
        }


def load() -> Settings:
    autonomy = _env("ATLAS_AUTONOMY", "recommend").lower()
    if autonomy not in AUTONOMY_LEVELS:
        raise RuntimeError(
            f"ATLAS_AUTONOMY={autonomy!r} is not one of {', '.join(AUTONOMY_LEVELS)}. "
            f"Refusing to boot: an unrecognised value must never be read as 'more'."
        )

    effort = _env("ATLAS_EFFORT", "high").lower()
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        raise RuntimeError(f"ATLAS_EFFORT={effort!r} is not a valid effort level.")

    return Settings(
        tws_api_url=_env("TWS_API_URL").rstrip("/"),
        tws_email=_env("TWS_EMAIL"),
        tws_password=_env("TWS_PASSWORD"),
        tws_token=_env("TWS_TOKEN"),
        tws_timeout_secs=_float("TWS_TIMEOUT_SECS", 45.0),
        tws_cron_secret=_env("TWS_CRON_SECRET") or _env("INTERNAL_CRON_SECRET"),

        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        deepseek_api_key=_env("DEEPSEEK_API_KEY"),
        model=_env("ATLAS_MODEL", "claude-opus-5"),
        fast_model=_env("ATLAS_FAST_MODEL", "claude-haiku-4-5"),
        effort=effort,
        max_tokens=_int("ATLAS_MAX_TOKENS", 16000),
        max_tool_iterations=_int("ATLAS_MAX_TOOL_ITERATIONS", 40),

        mongo_url=_env("MONGO_URL"),
        mongo_db=_env("ATLAS_DB_NAME", "tws"),
        collection_prefix=_env("ATLAS_COLLECTION_PREFIX", "atlas_"),

        autonomy=autonomy,
        sandbox=_bool("ATLAS_SANDBOX", True),
        kill_switch=_bool("ATLAS_KILL_SWITCH", False),

        daily_llm_budget_usd=_float("ATLAS_DAILY_LLM_BUDGET_USD", 25.0),
        max_outreach_per_hour=_int("ATLAS_MAX_OUTREACH_PER_HOUR", 40),
        max_calls_staged_per_day=_int("ATLAS_MAX_CALLS_STAGED_PER_DAY", 200),
        max_chat_messages_per_hour=_int("ATLAS_MAX_CHAT_PER_HOUR", 30),
        approval_threshold_usd=_float("ATLAS_APPROVAL_THRESHOLD_USD", 250.0),

        tick_seconds=_int("ATLAS_TICK_SECONDS", 900),
        work_every_hours=_float("ATLAS_WORK_EVERY_HOURS", 4.0),
        drive_app_jobs=_bool("ATLAS_DRIVE_APP_JOBS", False),
        app_job_seconds=_int("ATLAS_APP_JOB_SECONDS", 300),
        morning_brief_hour=_int("ATLAS_MORNING_HOUR", 7),
        evening_brief_hour=_int("ATLAS_EVENING_HOUR", 19),
        timezone=_env("ATLAS_TIMEZONE", "America/New_York"),

        console_api_key=_env("ATLAS_CONSOLE_API_KEY"),
        cors_origins=_env("ATLAS_CORS_ORIGINS"),
        port=_int("PORT", 8090),
        log_level=_env("ATLAS_LOG_LEVEL", "INFO").upper(),
    )


settings = load()
