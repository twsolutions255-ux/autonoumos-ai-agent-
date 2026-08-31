"""DeepSeek, wearing the shape the reasoning loop already speaks.

Atlas ran on Claude Opus at high effort every fifteen minutes and spent about
ten dollars in an hour. The reasoning was excellent and the bill was not
survivable for a company with no revenue yet.

The loop in engine.py is good code and none of it is vendor-specific, so this
is an ADAPTER rather than a second loop: DeepSeek in, Anthropic-shaped blocks
out, and everything above the request untouched.

THE TEST THAT MATTERS MOST IS THE DSML ONE.

DeepSeek's V4 models sometimes emit a tool call as markup inside the message
CONTENT instead of in the tool_calls field. Their docs say otherwise; this is
the exception, not the contract. Unhandled, Atlas would see a turn with no tool
calls, conclude the model was finished, and end a cycle having done nothing --
while the request logged as a perfectly healthy 200. That is the exact failure
this agent already spent a day being rescued from, and it would look identical.

The recovery is ported from the app Atlas drives, where it was found the hard
way. It is not going to be rediscovered here.
"""
import json

import pytest

from atlas.llm.deepseek import (
    DEEPSEEK_PRICING, messages_to_openai, recover_dsml_tool_calls,
    response_to_blocks, tools_to_openai, _Block,
)
from atlas.llm.engine import PRICING, is_deepseek

# DeepSeek's real output uses the FULL-WIDTH pipe, U+FF5C. ASCII is accepted
# too, so both are exercised -- a pattern that only handles the one nobody
# sends is worse than no pattern, because it looks handled.
WIDE = chr(0xFF5C)
ASCII_PIPE = chr(124)


def dsml(name, pipe=WIDE, **params):
    p = pipe
    body = "".join('<%sDSML%sparameter name="%s">%s</%sDSML%sparameter>'
                   % (p, p, k, v, p, p) for k, v in params.items())
    return ("<%sDSML%stool_calls><%sDSML%sinvoke name=\"%s\">%s"
            "</%sDSML%sinvoke></%sDSML%stool_calls>"
            % (p, p, p, p, name, body, p, p, p, p))


# ------------------------------------------------------------------ the quirk

def test_a_tool_call_hidden_in_prose_is_recovered():
    """Otherwise the cycle silently does nothing and logs a healthy 200."""
    msg = {"content": "Let me check the platform health.\n" + dsml("check_platform_health")}
    out = recover_dsml_tool_calls(msg)
    assert out.get("tool_calls"), "the call was thrown away"
    assert out["tool_calls"][0]["function"]["name"] == "check_platform_health"
    assert "DSML" not in out["content"], "raw markup survived into the prose"
    assert out["content"].startswith("Let me check")


def test_both_pipe_styles_are_recovered():
    """Full-width is what DeepSeek actually sends. ASCII is accepted because
    the two are indistinguishable on screen and one day it will be both."""
    for pipe, label in ((WIDE, "full-width"), (ASCII_PIPE, "ascii")):
        out = recover_dsml_tool_calls(
            {"content": "check " + dsml("check_platform_health", pipe=pipe)})
        assert out.get("tool_calls"), "%s pipes were not recovered" % label
        assert "DSML" not in out["content"]


def test_parameters_inside_the_markup_survive():
    msg = {"content": dsml("set_prospect_status", lead_id="L1", status="no_answer")}
    out = recover_dsml_tool_calls(msg)
    args = json.loads(out["tool_calls"][0]["function"]["arguments"])
    assert args == {"lead_id": "L1", "status": "no_answer"}


def test_a_real_tool_calls_field_always_wins():
    """The documented contract beats the recovery. Two calls where the model
    made one would run something twice."""
    real = [{"id": "1", "type": "function",
             "function": {"name": "get_team_roster", "arguments": "{}"}}]
    msg = {"content": dsml("archive_booking"), "tool_calls": real}
    out = recover_dsml_tool_calls(msg)
    assert out["tool_calls"] == real
    assert "DSML" not in out["content"]


def test_markup_with_no_recoverable_call_is_still_stripped():
    """A customer never sees a leaked internal format."""
    msg = {"content": "Here you go. <%sDSML%sthinking>hmm</%sDSML%sthinking>"
                      % (WIDE, WIDE, WIDE, WIDE)}
    out = recover_dsml_tool_calls(msg)
    assert "DSML" not in out["content"]
    assert out["content"].startswith("Here you go")


def test_ordinary_messages_pass_through_untouched():
    msg = {"content": "Nothing unusual here.", "tool_calls": None}
    assert recover_dsml_tool_calls(msg) == msg


# ------------------------------------------------------------- shape mapping

def test_tools_become_openai_functions_without_losing_the_schema():
    spec = [{"name": "archive_booking", "description": "  Take it out.  ",
             "input_schema": {"type": "object",
                              "properties": {"booking_id": {"type": "string"}},
                              "required": ["booking_id"],
                              "additionalProperties": False},
             "strict": True}]
    out = tools_to_openai(spec)
    assert out[0]["type"] == "function"
    fn = out[0]["function"]
    assert fn["name"] == "archive_booking"
    # The schema is what describes the contract. strict only decided who
    # enforced it, and DeepSeek has no equivalent.
    assert fn["parameters"]["additionalProperties"] is False
    assert fn["parameters"]["required"] == ["booking_id"]
    assert "strict" not in out[0] and "strict" not in fn


def test_an_assistant_turn_with_tool_calls_round_trips():
    convo = [
        {"role": "user", "content": "what is booked"},
        {"role": "assistant", "content": [
            _Block("text", text="Looking."),
            _Block("tool_use", id="tu_1", name="get_bookings", input={"month": "2026-08"}),
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "two demos"},
        ]},
    ]
    out = messages_to_openai("SYSTEM", convo)
    assert out[0] == {"role": "system", "content": "SYSTEM"}
    assert out[1]["role"] == "user"

    assistant = out[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Looking."
    assert assistant["tool_calls"][0]["id"] == "tu_1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"month": "2026-08"}

    result = out[3]
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "tu_1", \
        "an unanswered tool_call makes the NEXT request invalid"
    assert result["content"] == "two demos"


def test_every_parallel_tool_result_gets_its_own_message():
    """The loop answers parallel calls in one user turn. OpenAI needs one
    message each, and a missing one invalidates the whole next request."""
    convo = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "1"},
        {"type": "tool_result", "tool_use_id": "b", "content": "2"},
        {"type": "tool_result", "tool_use_id": "c", "content": "3"},
    ]}]
    out = messages_to_openai("S", convo)
    tool_msgs = [m for m in out if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b", "c"]


def test_a_response_becomes_blocks_the_loop_can_read():
    msg = {"content": "On it.",
           "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": "find_leads",
                                        "arguments": '{"trade": "hvac"}'}}]}
    blocks = response_to_blocks(msg)
    assert [b.type for b in blocks] == ["text", "tool_use"]
    assert blocks[0].text == "On it."
    assert blocks[1].name == "find_leads"
    assert blocks[1].input == {"trade": "hvac"}
    assert blocks[1].id == "c1"


def test_unparseable_arguments_become_a_tool_error_not_a_dropped_call():
    """Dropping the call leaves the model waiting for a result that never
    comes. Passing the mess through lets the tool report it and the model
    correct itself."""
    msg = {"content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "archive_booking", "arguments": "{not json"}}]}
    blocks = response_to_blocks(msg)
    assert len(blocks) == 1 and blocks[0].type == "tool_use"
    assert "_unparseable_arguments" in blocks[0].input


def test_a_text_only_answer_has_no_tool_blocks():
    blocks = response_to_blocks({"content": "Nothing to do today."})
    assert [b.type for b in blocks] == ["text"]


# ----------------------------------------------------------------- the money

def test_deepseek_is_priced_in_the_budget_guard():
    """A model the guard cannot price defaults to Opus rates, which would stop
    Atlas hours early for spending it never did."""
    for model in DEEPSEEK_PRICING:
        assert model in PRICING, "%s is not priced in the budget guard" % model
        assert PRICING[model] == DEEPSEEK_PRICING[model]


def test_deepseek_really_is_the_cheap_option():
    flash_in, flash_out = PRICING["deepseek-v4-flash"]
    opus_in, opus_out = PRICING["claude-opus-5"]
    assert flash_in < opus_in / 10
    assert flash_out < opus_out / 10


def test_the_vendor_is_decided_by_the_model_name():
    assert is_deepseek("deepseek-v4-flash")
    assert is_deepseek("deepseek-v4-pro")
    assert not is_deepseek("claude-opus-5")
    assert not is_deepseek("")
