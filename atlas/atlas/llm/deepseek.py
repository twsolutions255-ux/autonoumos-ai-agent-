"""DeepSeek, wearing the shape the reasoning loop already speaks.

WHY THIS EXISTS. Atlas ran on Claude Opus at high effort every fifteen minutes
and spent about ten dollars in an hour. Opus is $5/$25 per million tokens, the
turn carries eighty-two tool definitions, and each of up to forty tool
iterations re-sends the whole conversation. The reasoning was excellent and the
bill was not survivable for a company with no revenue yet.

DeepSeek's V4 models are roughly two orders of magnitude cheaper and the app
this agent drives already runs on them, so the operator and the agent now think
with the same models.

HOW IT IS BUILT, AND WHY THAT WAY.

The loop in engine.py is good code: it handles parallel tool calls correctly,
answers them in one message, caps iterations, and checks the budget before
every call. None of that is Anthropic-specific and none of it should be
rewritten to change vendor.

So this is an ADAPTER, not a second loop. It speaks OpenAI-compatible JSON to
DeepSeek and returns objects shaped the way the loop already reads them --
.content blocks with .type, .text, .id, .name, .input, plus .usage and
.stop_reason. The loop does not know which vendor answered.

WHAT DOES NOT TRANSLATE, said plainly rather than faked:

  thinking / effort   Anthropic's adaptive thinking has no DeepSeek equivalent.
                      Model choice carries that instead: flash for ordinary
                      cycles, pro for the hard ones.
  cache_control       No prompt caching. The system prompt is re-sent whole.
  server-side refusal There is no stop_reason == "refusal". A refusal from
                      DeepSeek arrives as ordinary text.

Each of those is absent, not emulated. A shim that pretends to support
something the vendor does not is how a guarantee quietly stops holding.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

import httpx

log = logging.getLogger("atlas.llm.deepseek")

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

#: Per-million-token pricing, for the budget guard and the console. DeepSeek
#: bills at half rate outside peak hours, so these over-reserve at worst --
#: which is the right direction for a cap.
DEEPSEEK_PRICING = {
    "deepseek-v4-flash": (0.28, 0.42),
    "deepseek-v4-pro": (0.55, 2.19),
}

# DeepSeek's V4 models sometimes emit a tool call as DSML markup inside the
# message CONTENT instead of in the tool_calls field. Their documentation says
# tool calls come back in the standard field, and mostly they do -- this is the
# exception, not the contract.
#
# It is handled here because the alternative is silent: Atlas would see a turn
# with no tool calls, treat it as "the model is done", and end a cycle having
# done nothing, while the log showed a perfectly healthy request. The same
# recovery already exists in the app this agent drives; it was found there the
# hard way and is not going to be rediscovered here.
# THE PIPES ARE FULL-WIDTH (U+FF5C) in DeepSeek's real output, the same
# character as in their other special tokens. ASCII pipes are accepted too
# because the two are indistinguishable on screen and one day it will be both.
#
# The first version of this file used a bare ASCII pipe as a literal, which in
# a regex is ALTERNATION -- so the pattern matched "<" or "DSML" or the rest,
# and stripped any tag it found while recovering nothing. It also could not
# have matched real DeepSeek output at all, which does not use ASCII pipes.
# These are copied from the app, where they were arrived at against real
# responses rather than guessed.
_DSML_PIPE = r"[|｜]"
_P = _DSML_PIPE
_Q = '["\']'          # either quote character

_DSML_BLOCK_RE = re.compile(
    "<" + _P + "DSML" + _P + "tool_calls>(.*?)</?" + _P + "DSML" + _P + "tool_calls>",
    re.DOTALL | re.IGNORECASE)
_DSML_INVOKE_RE = re.compile(
    "<" + _P + "DSML" + _P + "invoke\s+name=" + _Q + "([^" + _Q[1:-1] + "]+)" + _Q +
    ">(.*?)</?" + _P + "DSML" + _P + "invoke>",
    re.DOTALL | re.IGNORECASE)
_DSML_PARAM_RE = re.compile(
    "<" + _P + "DSML" + _P + "parameter\s+name=" + _Q + "([^" + _Q[1:-1] + "]+)" + _Q +
    ">(.*?)</?" + _P + "DSML" + _P + "parameter>",
    re.DOTALL | re.IGNORECASE)
# A partial block -- the model started the markup and stopped, or the response
# was truncated. Nothing to recover, but it must still not be displayed.
_DSML_ANY_RE = re.compile("<[/]?" + _P + "DSML" + _P + "[^>]*>", re.IGNORECASE)


# --------------------------------------------------------------------- shapes

class _Block:
    """One content block, read by the loop exactly like an Anthropic one."""

    def __init__(self, type_: str, *, text: str = "", id: str = "",
                 name: str = "", input: Optional[dict] = None):
        self.type = type_
        self.text = text
        self.id = id
        self.name = name
        self.input = input or {}


class _Usage:
    def __init__(self, prompt: int = 0, completion: int = 0):
        self.input_tokens = prompt
        self.output_tokens = completion
        # DeepSeek reports a cache hit count, but the loop prices cache reads
        # at a tenth of input and DeepSeek already discounts them in its own
        # billing. Reporting zero here keeps the guard conservative rather
        # than double-counting a discount.
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _Response:
    def __init__(self, content: list, usage: _Usage, stop_reason: str):
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason
        self.stop_details = None


# ---------------------------------------------------------------- translation

def tools_to_openai(tools: list) -> list:
    """Anthropic tool specs -> OpenAI function specs.

    `strict` is dropped: DeepSeek has no equivalent and would ignore it. The
    schema still forbids extra properties at every level, which is what
    actually describes the contract -- strict only decided who enforced it.
    """
    out = []
    for t in tools or []:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") or "",
                "parameters": t.get("input_schema")
                or {"type": "object", "properties": {}},
            },
        })
    return out


def messages_to_openai(system: str, convo: list) -> list:
    """The loop's Anthropic-shaped conversation -> OpenAI messages.

    Rebuilt whole on every call rather than maintained incrementally. The
    conversation is small, and two representations kept in step by hand is a
    bug waiting for the turn where they drift.
    """
    out: list = [{"role": "system", "content": system}]
    for msg in convo:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text = "".join(getattr(b, "text", "") or ""
                           for b in content if getattr(b, "type", "") == "text")
            calls = []
            for b in content:
                if getattr(b, "type", "") == "tool_use":
                    calls.append({
                        "id": b.id,
                        "type": "function",
                        "function": {"name": b.name,
                                     "arguments": json.dumps(b.input or {})},
                    })
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue

        # A user turn carrying tool results. Each result is its own message in
        # OpenAI's shape, and every one must be present -- a tool_call left
        # unanswered makes the next request invalid.
        for b in (content or []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                out.append({"role": "tool",
                            "tool_call_id": b.get("tool_use_id"),
                            "content": str(b.get("content") or "")})
            elif isinstance(b, dict) and b.get("type") == "text":
                out.append({"role": "user", "content": b.get("text") or ""})
    return out


def recover_dsml_tool_calls(message: dict) -> dict:
    """Pull a tool call out of the prose, or failing that get the markup out."""
    if not isinstance(message, dict):
        return message
    content = message.get("content") or ""
    if not content or "DSML" not in content:
        return message

    recovered = []
    for block in _DSML_BLOCK_RE.findall(content):
        for name, body in _DSML_INVOKE_RE.findall(block):
            args = {k: v.strip() for k, v in _DSML_PARAM_RE.findall(body)}
            recovered.append({
                "id": "dsml_" + str(uuid.uuid4())[:8],
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })

    cleaned = _DSML_BLOCK_RE.sub("", content)
    cleaned = _DSML_ANY_RE.sub("", cleaned).strip()
    out = {**message, "content": cleaned}

    # Only when the model did not also send them properly -- a real tool_calls
    # field is the contract and always wins.
    if recovered and not message.get("tool_calls"):
        out["tool_calls"] = recovered
        log.warning("atlas: recovered %d tool call(s) from DSML markup in "
                    "message content", len(recovered))
    elif not recovered:
        log.warning("atlas: stripped DSML markup carrying no recoverable call")
    return out


def response_to_blocks(message: dict) -> list:
    """An OpenAI assistant message -> the block list the loop reads."""
    blocks: list = []
    text = (message.get("content") or "").strip()
    if text:
        blocks.append(_Block("text", text=text))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (TypeError, ValueError):
            # Malformed arguments are the model's mistake, and the loop can
            # report it back as a tool error the model can correct. Dropping
            # the call would leave it waiting for a result that never comes.
            log.warning("atlas: tool %s had unparseable arguments",
                        fn.get("name"))
            args = {"_unparseable_arguments": str(raw)[:500]}
        if not isinstance(args, dict):
            args = {"value": args}
        blocks.append(_Block("tool_use", id=call.get("id") or str(uuid.uuid4()),
                             name=fn.get("name") or "", input=args))
    return blocks


# ------------------------------------------------------------------- the client

class DeepSeekClient:
    """Speaks DeepSeek. Answers in the shape the loop already reads."""

    def __init__(self, api_key: str, *, base_url: str = DEEPSEEK_URL,
                 timeout: float = 600.0):
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Atlas cannot reason without it "
                "and refuses to start a cycle that would silently do nothing.")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    async def call(self, *, system: str, convo: list, tools: list, model: str,
                   max_tokens: int, temperature: float = 0.3) -> _Response:
        payload = {
            "model": model,
            "messages": messages_to_openai(system, convo),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools_to_openai(tools)
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            resp = await http.post(
                self.base_url,
                headers={"Authorization": "Bearer %s" % self.api_key,
                         "Content-Type": "application/json"},
                json=payload)

        if resp.status_code >= 400:
            # Raised, not swallowed. A cycle that silently produced nothing is
            # the failure this whole agent spent a day being rescued from.
            raise RuntimeError("DeepSeek %s: %s"
                               % (resp.status_code, resp.text[:400]))

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = recover_dsml_tool_calls(choice.get("message") or {})
        usage = data.get("usage") or {}

        return _Response(
            content=response_to_blocks(message),
            usage=_Usage(int(usage.get("prompt_tokens") or 0),
                         int(usage.get("completion_tokens") or 0)),
            # Mapped to the loop's vocabulary. DeepSeek has no refusal stop
            # reason; a refusal arrives as ordinary text, so nothing here ever
            # claims one happened.
            stop_reason={"tool_calls": "tool_use",
                         "stop": "end_turn",
                         "length": "max_tokens"}.get(
                             choice.get("finish_reason") or "", "end_turn"),
        )
