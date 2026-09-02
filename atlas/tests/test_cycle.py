"""A whole autonomous cycle, start to finish.

Everything real except Claude: the runtime, the policy, the memory, the tools
and the HTTP client all run, against the fake TWS app. The model is scripted,
so the test asserts what the machinery does with a decision rather than
pretending to test the decision itself.

This is the test that would catch the failure that matters most — a cycle that
appears to succeed while having done nothing, or having done something it did
not report.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest

from tests.fake_tws import CRON_SECRET, GOOD_EMAIL, GOOD_PASSWORD, app as tws_app, state
from tests.test_engine import install, text_block, tool_block, usage
from tests.test_integration import MockStore, settings_for

from atlas.brain.loop import Runtime
from atlas.llm.engine import Engine
from atlas.tws.client import TWSClient


@pytest.fixture(autouse=True)
def _reset():
    state.reset()
    yield


async def build_runtime(**over):
    """A Runtime wired to the fake app and a mock database, not started."""
    settings = settings_for(**over)
    rt = Runtime(settings)
    rt.store = MockStore()
    from atlas.memory.store import MemoryStore
    rt.memory = MemoryStore(rt.store)

    rt.client = TWSClient(settings.tws_api_url, email=settings.tws_email,
                          password=settings.tws_password,
                          cron_secret=settings.tws_cron_secret,
                          audit=rt._audit_http)
    rt.client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=tws_app),
        base_url="http://tws.test", timeout=10)
    rt.identity = await rt.client.verify_access()
    rt.engine = Engine(settings, dispatch=rt._dispatch,
                       spend_today=lambda: rt._spend_today)
    await rt.store.ensure_indexes()
    return rt


@pytest.mark.asyncio
async def test_a_morning_cycle_reads_plans_speaks_and_briefs():
    rt = await build_runtime(ATLAS_AUTONOMY="operate", ATLAS_SANDBOX="false")
    install(rt.engine, [
        # Look at the business.
        SimpleNamespace(content=[tool_block("business_snapshot", {}, "t1")],
                        stop_reason="tool_use", usage=usage()),
        # Decide, and write the plan down.
        SimpleNamespace(content=[tool_block("set_plan", {
            "north_star": "Twelve live clients by the end of the quarter.",
            "rationale": "Two deals are won but never activated — that is the fastest money.",
            "objectives": [{"objective": "Activate every won deal",
                            "measure": "deals awaiting setup",
                            "target": "0 by Friday"}],
        }, "t2")], stop_reason="tool_use", usage=usage()),
        # Tell the team, and brief the owner.
        SimpleNamespace(content=[
            tool_block("post_to_channel",
                       {"channel": "general",
                        "message": "Today: activate the two won deals before any new outreach."}, "t3"),
            tool_block("brief_owner", {
                "kind": "morning",
                "headline": "Two won deals are not live yet",
                "body": "Echo Electric is signed and has no account. That is live MRR sitting still.",
            }, "t4"),
        ], stop_reason="tool_use", usage=usage()),
        SimpleNamespace(content=[text_block(
            "Priority today is activating the two won deals. Told the team; briefed you.")],
            stop_reason="end_turn", usage=usage()),
    ])

    record = await rt.run_cycle("morning")

    assert record["status"] == "done"
    assert record["kind"] == "morning"
    assert record["actions"] == 4
    assert "activating the two won deals" in record["summary"]

    # The cycle was persisted, not just returned.
    stored = await rt.store["cycles"].find_one({"id": record["id"]}, {"_id": 0})
    assert stored["status"] == "done" and stored["summary"]

    # The plan is real and versioned.
    plan = await rt.store["plan"].find_one({}, {"_id": 0}, sort=[("version", -1)])
    assert plan["version"] == 1
    assert "Twelve live clients" in plan["north_star"]

    # The team was actually told, in the real app.
    assert any("activate the two won deals" in m["body"].lower() for m in state.messages)

    # The briefing is retrievable.
    brief = await rt.store["briefs"].find_one({"kind": "morning"}, {"_id": 0})
    assert "not live yet" in brief["headline"]

    # Every tool call is in the audit trail, tagged to this cycle.
    actions = await rt.store["actions"].find({"cycle_id": record["id"]}, {"_id": 0}).to_list(20)
    assert {a["tool"] for a in actions} == {
        "business_snapshot", "set_plan", "post_to_channel", "brief_owner"}
    assert all(a["outcome"] == "allow" for a in actions)
    await rt.close()


@pytest.mark.asyncio
async def test_a_blocked_action_is_recorded_and_the_cycle_still_finishes():
    """The failure mode that matters: acting blocked, reporting success."""
    rt = await build_runtime(ATLAS_AUTONOMY="assist", ATLAS_SANDBOX="false")
    install(rt.engine, [
        SimpleNamespace(content=[tool_block("release_cold_call_batch", {}, "t1")],
                        stop_reason="tool_use", usage=usage()),
        SimpleNamespace(content=[text_block(
            "I could not release the batch — my authority does not cover placing calls.")],
            stop_reason="end_turn", usage=usage()),
    ])

    record = await rt.run_cycle("work")
    assert record["status"] == "done"
    assert state.released == 0, "a blocked tool actually dialled"

    actions = await rt.store["actions"].find({"cycle_id": record["id"]}, {"_id": 0}).to_list(10)
    assert len(actions) == 1
    assert actions[0]["outcome"] == "deny"
    assert actions[0]["gate"] == "autonomy"
    await rt.close()


@pytest.mark.asyncio
async def test_an_action_needing_approval_is_queued_not_performed():
    rt = await build_runtime(ATLAS_AUTONOMY="autopilot", ATLAS_SANDBOX="false")
    install(rt.engine, [
        SimpleNamespace(content=[tool_block(
            "set_cold_call_autonomy", {"enabled": True, "daily_cap": 40}, "t1")],
            stop_reason="tool_use", usage=usage()),
        SimpleNamespace(content=[text_block("Queued for your approval.")],
                        stop_reason="end_turn", usage=usage()),
    ])

    record = await rt.run_cycle("work")
    assert record["status"] == "done"
    assert state.autonomy["enabled"] is False, "queued action was performed anyway"

    pending = await rt.store["approvals"].find({"status": "pending"}, {"_id": 0}).to_list(5)
    assert len(pending) == 1
    assert pending[0]["tool"] == "set_cold_call_autonomy"
    assert pending[0]["args"]["daily_cap"] == 40
    await rt.close()


@pytest.mark.asyncio
async def test_the_kill_switch_makes_a_cycle_a_no_op():
    rt = await build_runtime()
    rt.policy.kill_switch = True
    install(rt.engine, [])          # asking the model at all would raise
    record = await rt.run_cycle("work")
    assert record["status"] == "done"
    assert "kill switch" in record["summary"].lower()
    assert record["actions"] == 0
    await rt.close()


@pytest.mark.asyncio
async def test_the_next_cycle_can_see_what_the_last_one_did():
    """Memory across cycles is the whole point — verify it reaches the prompt."""
    rt = await build_runtime()
    install(rt.engine, [SimpleNamespace(
        content=[text_block("Dentists are not converting; stopping that market.")],
        stop_reason="end_turn", usage=usage())])
    await rt.run_cycle("evening")

    await rt.memory.remember(kind="lesson", title="Dentists do not convert",
                             body="0 of 35 touches over two weeks", importance=5)

    opener = await rt._opening_message("morning", "")
    assert "Dentists do not convert" in opener
    assert "not converting; stopping that market" in opener
    await rt.close()


@pytest.mark.asyncio
async def test_a_refused_action_is_named_to_the_next_cycle_so_it_is_not_asked_again():
    """At 'recommend' every action is a denial, so without this each work
    cycle re-found the same gap, called the same tool, was refused the same
    way and recommended the same thing. Approvals already had a "do not
    queue again" line; denials get the same, as a list, not prose."""
    rt = await build_runtime(ATLAS_AUTONOMY="recommend", ATLAS_SANDBOX="true")
    before = await rt._opening_message("work", "")
    assert "ALREADY REFUSED" not in before          # nothing refused yet

    install(rt.engine, [
        SimpleNamespace(content=[tool_block("release_cold_call_batch", {}, "t1")],
                        stop_reason="tool_use", usage=usage()),
        SimpleNamespace(content=[text_block(
            "Recommend releasing the batch; I cannot do it myself.")],
            stop_reason="end_turn", usage=usage()),
    ])
    record = await rt.run_cycle("work")
    assert record["status"] == "done" and state.released == 0

    after = await rt._opening_message("work", "")
    assert "ALREADY REFUSED" in after
    assert "release_cold_call_batch" in after
    assert "blocked by autonomy" in after
    assert "do not recommend it again" in after    # the recommend-rung tail
    # The list is not fed to the model's own prose summary twice: one line
    # per tool however many times it was refused.
    assert after.count("  - release_cold_call_batch (") == 1
    await rt.close()


@pytest.mark.asyncio
async def test_work_cycles_are_rationed_and_run_on_the_cheap_model():
    """Hourly ticks with no cap were ~22 planner calls a day on the pro
    model, mostly re-reading the same numbers. Work is now owed only every
    ATLAS_WORK_EVERY_HOURS, and takes the fast model at half the cap."""
    from datetime import timedelta as _td
    from atlas.brain.loop import now as _now
    rt = await build_runtime(ATLAS_WORK_EVERY_HOURS="4", DEEPSEEK_API_KEY="test-only",
                             ATLAS_MODEL="deepseek-v4-pro",
                             ATLAS_FAST_MODEL="deepseek-v4-flash",
                             ATLAS_MAX_TOOL_ITERATIONS="24")
    today = _now().date().isoformat()
    # Both once-a-day cycles done: what is owed depends only on work timing.
    assert rt._due_kind(today, today, None) == "work"
    assert rt._due_kind(today, today, _now() - _td(hours=1)) == ""
    assert rt._due_kind(today, today, _now() - _td(hours=5)) == "work"

    # Only the morning plans, so only the morning pays for the strong model.
    # The evening review reads numbers and writes them up; it was on pro for
    # no reason anybody could name, and it was one of the two most expensive
    # cycles of the day.
    assert rt._model_for("work") == ("deepseek-v4-flash", 12)
    assert rt._model_for("evening") == ("deepseek-v4-flash", 12)
    assert rt._model_for("morning") == ("deepseek-v4-pro", 24)

    # And the choice actually reaches the engine.
    seen = {}
    async def fake_run(**kw):
        seen.update(kw)
        raise RuntimeError("stop here")
    rt.engine.run = fake_run
    await rt.run_cycle("work")
    assert seen["model"] == "deepseek-v4-flash" and seen["max_iterations"] == 12

    # A redeploy reads the last work start from storage rather than granting
    # a fresh one.
    await rt.store["cycles"].insert_one({
        "id": "x", "kind": "work", "status": "done",
        "started_at": (_now() - _td(hours=1)).isoformat()})
    last = await rt._last_work_at()
    assert last is not None and rt._due_kind(today, today, last) == ""
    await rt.close()


@pytest.mark.asyncio
async def test_a_cycle_that_crashes_still_leaves_a_record():
    rt = await build_runtime()

    class Boom:
        def stream(self, **kw):
            raise RuntimeError("the model is unreachable")
    rt.engine.client = SimpleNamespace(beta=SimpleNamespace(messages=Boom()),
                                       messages=Boom())

    record = await rt.run_cycle("work")
    assert record["status"] == "failed"
    assert "unreachable" in record["error"]
    stored = await rt.store["cycles"].find_one({"id": record["id"]}, {"_id": 0})
    assert stored["status"] == "failed" and stored["ended_at"]
    await rt.close()


@pytest.mark.asyncio
async def test_two_cycles_cannot_run_at_once():
    """A second scheduler tick must not start a parallel agent."""
    import asyncio
    rt = await build_runtime()
    install(rt.engine, [SimpleNamespace(content=[text_block("done")],
                                        stop_reason="end_turn", usage=usage())])

    async def slow(*a, **k):
        await asyncio.sleep(0.2)
        return SimpleNamespace(text="done", usage=__import__(
            "atlas.llm.engine", fromlist=["Usage"]).Usage(), actions=[],
            stop_reason="end_turn", iterations=1, truncated=False, refusal=None)
    rt.engine.run = slow

    first, second = await asyncio.gather(rt.run_cycle("work"), rt.run_cycle("work"))
    outcomes = [first, second]
    assert sum(1 for r in outcomes if r.get("skipped")) == 1, \
        "two cycles ran at the same time"
    await rt.close()


# ---------------------------------------------------------------- job driver

@pytest.mark.asyncio
async def test_the_job_driver_guard_blocks_every_way_it_should():
    """The driver runs on a timer with no model in the loop, so its guard is
    the only thing between it and phoning strangers. Tested directly rather
    than by waiting out the real interval."""
    rt = await build_runtime(ATLAS_AUTONOMY="operate", ATLAS_SANDBOX="false")

    ok, why = rt.may_drive_app_jobs()
    assert ok is True and why == ""

    rt.policy.sandbox = True
    ok, why = rt.may_drive_app_jobs()
    assert ok is False and "sandbox" in why

    rt.policy.sandbox = False
    rt.policy.set_autonomy("assist")
    ok, why = rt.may_drive_app_jobs()
    assert ok is False and "below 'operate'" in why

    rt.policy.set_autonomy("operate")
    rt.policy.kill_switch = True
    ok, why = rt.may_drive_app_jobs()
    assert ok is False and "kill switch" in why

    # The kill switch outranks everything, as it does in the policy gate.
    rt.policy.sandbox = True
    assert rt.may_drive_app_jobs()[1] == "the kill switch is on"

    rt.policy.kill_switch = False
    rt.policy.sandbox = False
    rt.client = None
    assert rt.may_drive_app_jobs()[0] is False
    await rt.store.close()


@pytest.mark.asyncio
async def test_the_job_driver_refuses_to_start_without_a_cron_secret():
    rt = await build_runtime(TWS_CRON_SECRET="", INTERNAL_CRON_SECRET="")
    rt._running = True
    await rt.drive_app_jobs_forever()      # returns immediately rather than looping
    assert state.jobs_run == []
    await rt.close()


@pytest.mark.asyncio
async def test_the_job_driver_actually_drives_when_permitted():
    import asyncio
    state.reset()
    rt = await build_runtime(ATLAS_AUTONOMY="operate", ATLAS_SANDBOX="false")
    rt._running = True
    # Reach past the sleep by driving one iteration's worth of work directly,
    # rather than waiting out the real 60s floor.
    for path in ("/internal/speed-to-lead/drain", "/internal/workflows/run"):
        await rt.client.run_internal_job(path)
    assert "speed-to-lead/drain" in state.jobs_run[0]
    assert "workflows/run" in state.jobs_run[1]
    await rt.close()


# ---------------------------------------------------------------- scheduling

@pytest.mark.asyncio
async def test_a_restart_in_the_evening_does_not_write_a_morning_plan():
    """The bug this pins: `hour >= morning_hour` with in-memory state means any
    redeploy after 7am produces a 'morning plan' — at 10pm, for a day that is
    over — and then never runs the evening review, because morning matched
    first."""
    rt = await build_runtime(ATLAS_MORNING_HOUR=7, ATLAS_EVENING_HOUR=19)

    class At:
        def __init__(self, hour): self.hour = hour
        def date(self): return self
        def isoformat(self): return "2026-08-30"

    import atlas.brain.loop as loop
    real_now = loop.now
    try:
        for hour, expected in [
            (6, "work"),        # before the morning window
            (7, "morning"),     # window opens
            (12, "morning"),    # still owed
            (19, "evening"),    # morning window closed; evening owed
            (22, "evening"),    # still owed
        ]:
            loop.now = lambda h=hour: At(h)
            assert rt._due_kind("", "") == expected, f"at {hour}:00 wanted {expected}"

        # Already done today -> plain work ticks, at any hour.
        loop.now = lambda: At(20)
        assert rt._due_kind("2026-08-30", "2026-08-30") == "work"
        loop.now = lambda: At(9)
        assert rt._due_kind("2026-08-30", "") == "work"
    finally:
        loop.now = real_now
    await rt.close()


@pytest.mark.asyncio
async def test_the_scheduler_reads_back_what_already_ran_today():
    """Held only in memory, every redeploy re-briefs the owner."""
    rt = await build_runtime()
    assert await rt._last_cycle_days() == ("", "")

    install(rt.engine, [SimpleNamespace(content=[text_block("morning done")],
                                        stop_reason="end_turn", usage=usage())])
    rec = await rt.run_cycle("morning")
    assert rec["status"] == "done"

    last_morning, last_evening = await rt._last_cycle_days()
    assert last_morning == rec["day"]
    assert last_evening == ""
    await rt.close()


@pytest.mark.asyncio
async def test_spend_survives_a_restart_including_what_chat_cost():
    rt = await build_runtime(ATLAS_DAILY_LLM_BUDGET_USD=10)
    install(rt.engine, [SimpleNamespace(content=[text_block("hi")],
                                        stop_reason="end_turn", usage=usage())])
    await rt.run_cycle("work")

    from atlas.db import iso
    await rt.store["chat"].insert_one({
        "id": "c1", "role": "assistant", "text": "hello",
        "created_at": iso(), "usage": {"cost_usd": 3.25}})

    rt._spend_today = 0.0
    await rt._load_spend()
    assert rt._spend_today >= 3.25, "a redeploy handed back a budget already spent"
    await rt.close()
