"""The reasoning loop: Claude, tools, and the gate between them.

This is a manual agentic loop rather than the SDK's tool runner, for one
reason that matters: every tool call has to pass through the policy gate, be
written to the audit log, and be able to come back as "held for approval"
without the loop mistaking that for an error. Owning the loop makes those
three things explicit at the point they happen.

Cost and safety notes:

* **The system prompt is cached.** It is long (the whole tool doctrine and the
  business context) and identical across the cycles of a day, so it sits
  behind a cache breakpoint with volatile content — the live snapshot, the
  time — placed *after* it. Cache hits are asserted, not assumed: `usage`
  is recorded on every call and surfaced in the console.

* **Every turn is bounded.** `max_tool_iterations` caps tool round-trips, and
  the daily budget is checked before each model call. An agent that loops
  unattended is the failure mode that costs real money.

* **Refusals are handled.** Server-side fallback is enabled, and `stop_reason`
  is checked before content is read.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .deepseek import DeepSeekClient

log = logging.getLogger("atlas.llm")

#: Per-million-token pricing, used only for budget accounting and the console.
#: Wrong numbers here cost nothing but a mis-drawn chart; the hard stop is the
#: budget check, which is conservative by design.
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Roughly two orders of magnitude cheaper, which is the whole reason Atlas
    # can run hourly at all. Peak rates: DeepSeek bills at half outside peak,
    # so this over-reserves at worst -- the right direction for a budget guard.
    "deepseek-v4-flash": (0.28, 0.42),
    "deepseek-v4-pro": (0.55, 2.19),
}


def is_deepseek(model: str) -> bool:
    return (model or "").startswith("deepseek")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, model: str, u: Any) -> None:
        self.calls += 1
        i = getattr(u, "input_tokens", 0) or 0
        o = getattr(u, "output_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        self.input_tokens += i
        self.output_tokens += o
        self.cache_read += cr
        self.cache_write += cw
        in_rate, out_rate = PRICING.get(model, (5.0, 25.0))
        # Cache reads bill at a tenth of input; writes at 1.25x. Approximate
        # on purpose — this is a budget guard, not an invoice.
        self.cost_usd += (
            (i * in_rate) + (cr * in_rate * 0.1) + (cw * in_rate * 1.25) + (o * out_rate)
        ) / 1_000_000

    def merge(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read += other.cache_read
        self.cache_write += other.cache_write
        self.cost_usd += other.cost_usd
        self.calls += other.calls

    def as_dict(self) -> dict:
        return {
            "calls": self.calls, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "cache_read": self.cache_read,
            "cache_write": self.cache_write, "cost_usd": round(self.cost_usd, 4),
            "cache_hit_rate": round(
                self.cache_read / max(1, self.cache_read + self.input_tokens), 3),
        }


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TurnResult:
    text: str
    usage: Usage
    #: Every tool call attempted this turn, with its policy decision.
    actions: list = field(default_factory=list)
    stop_reason: str = ""
    iterations: int = 0
    #: True when the loop hit its iteration cap rather than finishing. The
    #: caller needs to know the answer may be half-formed.
    truncated: bool = False
    refusal: Optional[dict] = None


class Engine:
    """One Claude conversation, driven to completion, with tools gated."""

    def __init__(self, settings, *, dispatch: Callable[[str, dict], Awaitable[Any]],
                 spend_today: Callable[[], float] = lambda: 0.0):
        # WHICHEVER VENDOR THE CONFIGURED MODEL BELONGS TO.
        #
        # Both models are declared -- model and fast_model -- and either may be
        # DeepSeek, so a deployment running DeepSeek for the main loop and
        # Claude for one-shot asks needs both clients. Each is built only when
        # its key exists, and the check names the model that asked for it: a
        # deployment refusing to start should say which setting to fix.
        self.settings = settings
        self._anthropic = None
        self.client = None
        self.deepseek = None

        wants = {settings.model, settings.fast_model}
        needs_anthropic = any(not is_deepseek(m) for m in wants)
        needs_deepseek = any(is_deepseek(m) for m in wants)

        if needs_anthropic:
            if not settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set, and %s is a Claude model. "
                    "Set the key, or set ATLAS_MODEL and ATLAS_FAST_MODEL to "
                    "DeepSeek models."
                    % ", ".join(sorted(m for m in wants if not is_deepseek(m))))
            import anthropic
            self._anthropic = anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                # A tool-heavy turn is long. The SDK default is 10 minutes; the
                # loop's own iteration cap is the real bound.
                timeout=600.0, max_retries=3,
            )

        if needs_deepseek:
            key = getattr(settings, "deepseek_api_key", "") or ""
            if not key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY is not set, and %s is a DeepSeek model. "
                    "Atlas refuses to start a cycle that would silently do "
                    "nothing."
                    % ", ".join(sorted(m for m in wants if is_deepseek(m))))
            self.deepseek = DeepSeekClient(key)
        self._dispatch = dispatch
        self._spend_today = spend_today

    def _budget_check(self, usage: Usage) -> None:
        spent = self._spend_today() + usage.cost_usd
        if spent >= self.settings.daily_llm_budget_usd:
            raise BudgetExceeded(
                f"Atlas has spent ${spent:.2f} of its ${self.settings.daily_llm_budget_usd:.2f} "
                f"daily thinking budget and has stopped for the day. Raise "
                f"ATLAS_DAILY_LLM_BUDGET_USD to give it more room.")

    async def run(self, *, system: str, messages: list, tools: list,
                  model: Optional[str] = None, effort: Optional[str] = None,
                  max_iterations: Optional[int] = None,
                  on_event: Optional[Callable[[dict], Awaitable[None]]] = None) -> TurnResult:
        """Drive one reasoning turn to completion, executing tools as asked."""
        model = model or self.settings.model
        effort = effort or self.settings.effort
        cap = max_iterations or self.settings.max_tool_iterations
        usage = Usage()
        actions: list = []
        convo = list(messages)
        truncated = False
        stop_reason = ""
        refusal = None
        resp = None
        rounds = 0

        for _ in range(cap):
            rounds += 1
            self._budget_check(usage)
            resp = await self._call(system, convo, tools, model, effort)
            usage.add(model, resp.usage)
            stop_reason = resp.stop_reason or ""

            # Safety classifiers can decline. Check before reading content.
            if stop_reason == "refusal":
                details = getattr(resp, "stop_details", None)
                refusal = {
                    "category": getattr(details, "category", None),
                    "explanation": getattr(details, "explanation", None),
                }
                log.warning("atlas: model refused (%s)", refusal.get("category"))
                break

            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                convo.append({"role": "assistant", "content": resp.content})
                break

            convo.append({"role": "assistant", "content": resp.content})

            # Parallel tool calls arrive together and must be answered together,
            # in ONE user message. Splitting them teaches the model to stop
            # issuing parallel calls at all.
            results = await asyncio.gather(
                *[self._run_tool(tu, actions, on_event) for tu in tool_uses])
            convo.append({"role": "user", "content": list(results)})
        else:
            truncated = True
            log.warning("atlas: turn hit the %d-iteration cap", cap)

        text = "" if resp is None else "\n".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()

        return TurnResult(text=text, usage=usage, actions=actions,
                          stop_reason=stop_reason, iterations=rounds,
                          truncated=truncated, refusal=refusal)

    async def _call(self, system: str, convo: list, tools: list,
                    model: str, effort: str):
        """One request. Streamed for Claude, because turns here are long.

        DeepSeek is answered through an adapter that returns the same block
        shapes, so everything above this line is vendor-agnostic. Effort has no
        DeepSeek equivalent and is not faked -- model choice carries it there.
        """
        if is_deepseek(model):
            return await self.deepseek.call(
                system=system, convo=convo, tools=tools, model=model,
                max_tokens=self.settings.max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.settings.max_tokens,
            # The stable half of the prompt is cached; callers put volatile
            # context in the messages, after this breakpoint.
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": convo,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if tools:
            kwargs["tools"] = tools

        try:
            async with self.client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **kwargs,
            ) as stream:
                return await stream.get_final_message()
        except self._anthropic.BadRequestError:
            # A deployment whose account or model does not have the fallback
            # beta should degrade to a plain call rather than stop working.
            log.info("atlas: retrying without server-side fallback")
            async with self.client.messages.stream(**kwargs) as stream:
                return await stream.get_final_message()

    async def _run_tool(self, tu: Any, actions: list,
                        on_event: Optional[Callable[[dict], Awaitable[None]]]) -> dict:
        """Execute one tool call and shape its result block.

        A failure here is returned to the model as an error result, never
        raised: the model can read "that did not work, here is why" and pick
        another route, whereas an exception ends the whole turn.
        """
        name = tu.name
        # Inputs arrive already parsed by the SDK; guard anyway because a
        # malformed input must not take the loop down.
        args = tu.input if isinstance(tu.input, dict) else {}
        started = time.monotonic()
        is_error = False
        try:
            out = await self._dispatch(name, args)
            content = out if isinstance(out, str) else json.dumps(out, default=str)
        except Exception as e:
            log.exception("atlas: tool %s raised", name)
            content = f"{type(e).__name__}: {e}"
            is_error = True

        record = {
            "tool": name, "args": args,
            "ms": round((time.monotonic() - started) * 1000),
            "error": is_error,
            "result_preview": content[:400],
        }
        actions.append(record)
        if on_event:
            try:
                await on_event(record)
            except Exception:
                log.exception("atlas: on_event raised")

        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": content[:60000],
        }
        if is_error:
            block["is_error"] = True
        return block

    async def ask(self, *, system: str, prompt: str, model: Optional[str] = None,
                  effort: str = "low", max_tokens: int = 2000) -> str:
        """A single question with no tools. For scoring, drafting, classifying."""
        resp = await self.client.messages.create(
            model=model or self.settings.fast_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
        )
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
