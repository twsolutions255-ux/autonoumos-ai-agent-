"""The agentic loop, with Claude stubbed.

What matters here is not that the model is clever — it is that the machinery
around it is correct. Specifically: tool results must be returned in ONE user
message per assistant turn (splitting them silently teaches the model to stop
issuing parallel calls), a refusal must be detected before content is read,
the iteration cap must hold, and the budget must stop the loop rather than
letting it run all night.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from atlas.config import load
from atlas.llm.engine import BudgetExceeded, Engine, Usage


def settings_for(**over):
    env = {"ANTHROPIC_API_KEY": "test-key", "MONGO_URL": "mongodb://x",
           "ATLAS_CONSOLE_API_KEY": "k", "TWS_API_URL": "http://x",
           "TWS_EMAIL": "a@b.c", "TWS_PASSWORD": "p"}
    env.update({k: str(v) for k, v in over.items()})
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return load()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})


def usage(i=100, o=50):
    return SimpleNamespace(input_tokens=i, output_tokens=o,
                           cache_read_input_tokens=0, cache_creation_input_tokens=0)


def text_block(t):
    return SimpleNamespace(type="text", text=t)


def tool_block(name, args, id_):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id_)


class FakeStream:
    def __init__(self, message):
        self._m = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get_final_message(self):
        return self._m


class FakeMessages:
    """Replays a scripted list of responses and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def stream(self, **kwargs):
        # Snapshot the message list. The engine keeps appending to the same
        # list object across turns, so storing the reference would make every
        # recorded request show the FINAL conversation rather than the one
        # actually sent at that moment.
        snap = dict(kwargs)
        snap["messages"] = list(kwargs.get("messages") or [])
        self.requests.append(snap)
        if not self.script:
            raise AssertionError("the loop asked for more turns than were scripted")
        return FakeStream(self.script.pop(0))


def install(engine, script):
    fake = FakeMessages(script)
    engine.client = SimpleNamespace(
        beta=SimpleNamespace(messages=fake),
        messages=fake,
    )
    return fake


def make_engine(calls, settings=None, spend=0.0):
    """calls: a list that each dispatch appends to, returning a canned result."""
    async def dispatch(name, args):
        calls.append((name, args))
        return f"ok:{name}"
    return Engine(settings or settings_for(), dispatch=dispatch,
                  spend_today=lambda: spend)


@pytest.mark.asyncio
async def test_finishes_when_the_model_stops_calling_tools():
    calls = []
    e = make_engine(calls)
    install(e, [SimpleNamespace(content=[text_block("All done.")],
                                stop_reason="end_turn", usage=usage())])
    r = await e.run(system="s", messages=[{"role": "user", "content": "go"}], tools=[])
    assert r.text == "All done."
    assert r.iterations == 1
    assert calls == []


@pytest.mark.asyncio
async def test_executes_a_tool_then_finishes():
    calls = []
    e = make_engine(calls)
    install(e, [
        SimpleNamespace(content=[tool_block("business_snapshot", {}, "t1")],
                        stop_reason="tool_use", usage=usage()),
        SimpleNamespace(content=[text_block("12 clients.")],
                        stop_reason="end_turn", usage=usage()),
    ])
    r = await e.run(system="s", messages=[{"role": "user", "content": "go"}], tools=[])
    assert calls == [("business_snapshot", {})]
    assert r.text == "12 clients."
    assert len(r.actions) == 1 and r.actions[0]["tool"] == "business_snapshot"


@pytest.mark.asyncio
async def test_parallel_tool_results_come_back_in_exactly_one_user_message():
    calls = []
    e = make_engine(calls)
    fake = install(e, [
        SimpleNamespace(
            content=[tool_block("get_alerts", {}, "t1"),
                     tool_block("get_clients", {}, "t2"),
                     tool_block("get_team", {}, "t3")],
            stop_reason="tool_use", usage=usage()),
        SimpleNamespace(content=[text_block("done")], stop_reason="end_turn", usage=usage()),
    ])
    await e.run(system="s", messages=[{"role": "user", "content": "go"}], tools=[])
    assert len(calls) == 3

    convo = fake.requests[-1]["messages"]
    tool_result_msgs = [
        m for m in convo
        if m["role"] == "user" and isinstance(m["content"], list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_result_msgs) == 1, "results were split across messages"
    assert len(tool_result_msgs[0]["content"]) == 3, "not every call was answered"
    ids = {b["tool_use_id"] for b in tool_result_msgs[0]["content"]}
    assert ids == {"t1", "t2", "t3"}


@pytest.mark.asyncio
async def test_a_failing_tool_is_reported_to_the_model_not_raised():
    async def dispatch(name, args):
        raise RuntimeError("the app is down")
    e = Engine(settings_for(), dispatch=dispatch, spend_today=lambda: 0.0)
    fake = install(e, [
        SimpleNamespace(content=[tool_block("get_alerts", {}, "t1")],
                        stop_reason="tool_use", usage=usage()),
        SimpleNamespace(content=[text_block("I could not read the alerts.")],
                        stop_reason="end_turn", usage=usage()),
    ])
    r = await e.run(system="s", messages=[{"role": "user", "content": "go"}], tools=[])
    assert r.text == "I could not read the alerts."

    blocks = [b for m in fake.requests[-1]["messages"]
              if m["role"] == "user" and isinstance(m["content"], list)
              for b in m["content"]
              if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert len(blocks) == 1
    assert blocks[0]["is_error"] is True
    assert "the app is down" in blocks[0]["content"]
    assert r.actions[0]["error"] is True


@pytest.mark.asyncio
async def test_a_refusal_is_detected_before_content_is_read():
    e = make_engine([])
    install(e, [SimpleNamespace(
        content=[], stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation="declined"),
        usage=usage())])
    r = await e.run(system="s", messages=[{"role": "user", "content": "go"}], tools=[])
    assert r.refusal["category"] == "cyber"
    assert r.stop_reason == "refusal"


@pytest.mark.asyncio
async def test_the_iteration_cap_holds_and_is_reported():
    calls = []
    e = make_engine(calls, settings_for(ATLAS_MAX_TOOL_ITERATIONS=3))
    install(e, [
        SimpleNamespace(content=[tool_block("get_alerts", {}, f"t{i}")],
                        stop_reason="tool_use", usage=usage())
        for i in range(3)
    ])
    r = await e.run(system="s", messages=[{"role": "user", "content": "go"}], tools=[])
    assert r.truncated is True
    assert r.iterations == 3
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_the_budget_stops_the_loop():
    e = make_engine([], settings_for(ATLAS_DAILY_LLM_BUDGET_USD=5), spend=5.01)
    install(e, [SimpleNamespace(content=[text_block("hi")],
                                stop_reason="end_turn", usage=usage())])
    with pytest.raises(BudgetExceeded) as err:
        await e.run(system="s", messages=[{"role": "user", "content": "go"}], tools=[])
    assert "daily thinking budget" in str(err.value)


@pytest.mark.asyncio
async def test_the_system_prompt_is_sent_with_a_cache_breakpoint():
    e = make_engine([])
    fake = install(e, [SimpleNamespace(content=[text_block("hi")],
                                       stop_reason="end_turn", usage=usage())])
    await e.run(system="STABLE DOCTRINE",
                messages=[{"role": "user", "content": "go"}], tools=[])
    system = fake.requests[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == "STABLE DOCTRINE"
    # Adaptive thinking and effort, not the removed budget_tokens knob.
    assert fake.requests[0]["thinking"] == {"type": "adaptive"}
    assert "effort" in fake.requests[0]["output_config"]
    assert "temperature" not in fake.requests[0]
    assert "budget_tokens" not in str(fake.requests[0]["thinking"])


def test_usage_accounting_prices_cache_reads_cheaper_than_fresh_input():
    fresh, cached = Usage(), Usage()
    fresh.add("claude-opus-5", usage(i=1_000_000, o=0))
    cached.add("claude-opus-5", SimpleNamespace(
        input_tokens=0, output_tokens=0,
        cache_read_input_tokens=1_000_000, cache_creation_input_tokens=0))
    assert fresh.cost_usd == pytest.approx(5.0)
    assert cached.cost_usd == pytest.approx(0.5)
    assert cached.as_dict()["cache_hit_rate"] == 1.0
