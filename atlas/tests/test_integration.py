"""End-to-end: real client, real policy, real tools, against a real ASGI app.

No mocking of Atlas's own code. The only thing stubbed is Claude, because the
point is to prove the machinery around the model is correct — a model that
asks for a tool gets exactly the behaviour the policy says it should.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest
from mongomock_motor import AsyncMongoMockClient

from tests import fake_tws
from tests.fake_tws import CRON_SECRET, GOOD_EMAIL, GOOD_PASSWORD, app as tws_app, state

from atlas.config import load
from atlas.db import Store
from atlas.guardrails.policy import Policy, RateLimiter
from atlas.memory.store import MemoryStore
from atlas.tools.registry import registry
# Importing the tool modules is what registers them — the same thing the
# runtime does at import time. Without it the registry is empty.
from atlas.tools import comms, growth, money, observe, reflect  # noqa: F401
from atlas.tws.client import TWSClient, TWSError


def settings_for(**over):
    env = {
        "TWS_API_URL": "http://tws.test", "TWS_EMAIL": GOOD_EMAIL,
        "TWS_PASSWORD": GOOD_PASSWORD, "TWS_CRON_SECRET": CRON_SECRET,
        "MONGO_URL": "mongodb://x", "ATLAS_CONSOLE_API_KEY": "k",
        "ANTHROPIC_API_KEY": "test", "ATLAS_SANDBOX": "false",
        "ATLAS_AUTONOMY": "operate",
    }
    env.update({k: str(v) for k, v in over.items()})
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return load()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class MockStore(Store):
    """Store backed by mongomock, keeping the prefix and collection guard."""

    def __init__(self, prefix="atlas_"):
        self._client = AsyncMongoMockClient()
        self._db = self._client["tws"]
        self._prefix = prefix

    async def close(self):
        pass


def make_client(settings) -> TWSClient:
    c = TWSClient(settings.tws_api_url, email=settings.tws_email,
                  password=settings.tws_password,
                  cron_secret=settings.tws_cron_secret)
    # Route the client's HTTP straight into the fake ASGI app — real request
    # objects, real status codes, no network.
    c._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=tws_app),
                                  base_url="http://tws.test", timeout=10)
    return c


class Harness:
    """Everything a tool needs, wired the way the runtime wires it."""

    def __init__(self, settings):
        self.settings = settings
        self.store = MockStore()
        self.memory = MemoryStore(self.store)
        self.policy = Policy(settings, RateLimiter())
        self.client = make_client(settings)
        self.recorded: list = []
        self.approvals: list = []

    def ctx(self, cycle_id="cyc-1"):
        from atlas.tools.registry import ToolContext

        async def rec(entry):
            self.recorded.append(entry)
            await self.store["actions"].insert_one({"id": str(len(self.recorded)), **entry})

        async def appr(entry):
            self.approvals.append(entry)
            aid = f"appr-{len(self.approvals)}"
            await self.store["approvals"].insert_one(
                {"id": aid, "status": "pending", "created_at": "now", **entry})
            return aid

        return ToolContext(client=self.client, store=self.store, memory=self.memory,
                           policy=self.policy, settings=self.settings,
                           cycle_id=cycle_id, _recorder=rec, _approver=appr)

    async def call(self, name, args=None, approved=False):
        return await registry.dispatch(self.ctx(), name, args or {}, approved=approved)

    async def close(self):
        await self.client.close()


@pytest.fixture(autouse=True)
def _reset():
    state.reset()
    yield


@pytest.fixture
async def h():
    harness = Harness(settings_for())
    yield harness
    await harness.close()


# ---------------------------------------------------------------- auth

@pytest.mark.asyncio
async def test_logs_in_and_identifies_itself(h):
    who = await h.client.verify_access()
    assert who["is_superadmin"] is True
    assert who["email"] == GOOD_EMAIL


@pytest.mark.asyncio
async def test_recovers_from_an_expired_token_without_help(h):
    await h.client.get("/admin/overview")
    state.expire_next_token = True           # next call 401s once
    data = await h.client.get("/admin/overview")
    assert data["clients"] == 12             # replayed successfully


@pytest.mark.asyncio
async def test_a_wrong_password_stops_instead_of_locking_the_account_out():
    s = settings_for(TWS_PASSWORD="wrong")
    c = make_client(s)
    for _ in range(2):
        with pytest.raises(TWSError):
            await c.get("/admin/overview")
    # The app locks an identifier after 5 failures. Atlas must stop well short.
    with pytest.raises(TWSError) as e:
        await c.get("/admin/overview")
    assert "Not attempting login" in str(e.value)
    attempts = [p for m, p in state.calls if p == "/api/auth/login"]
    assert len(attempts) <= 2, f"tried to log in {len(attempts)} times — would lock out"
    await c.close()


# ---------------------------------------------------------------- reading

@pytest.mark.asyncio
async def test_snapshot_survives_a_half_configured_deployment(h):
    snap = await h.call("business_snapshot")
    assert snap["totals"]["clients"] == 12
    assert snap["alerts"]["count"] == 2
    assert len(snap["alerts"]["high"]) == 1


@pytest.mark.asyncio
async def test_leaked_client_api_key_never_reaches_the_audit_log(h):
    await h.call("get_clients")
    stored = await h.store["actions"].find({}, {"_id": 0}).to_list(10)
    blob = str(stored)
    assert "twsagent_live_SHOULD_BE_REDACTED" not in blob
    assert "***" in blob


# ---------------------------------------------------------------- comms

@pytest.mark.asyncio
async def test_asks_an_ai_employee_and_collects_the_async_reply(h):
    out = await h.call("ask_ai_employee",
                       {"handle": "viktor", "question": "How is the week looking?",
                        "wait_seconds": 8})
    assert out["answered"] is True
    assert "demos booked" in out["answer"]


@pytest.mark.asyncio
async def test_refuses_to_mention_an_employee_via_a_plain_channel_post(h):
    out = await h.call("post_to_channel",
                       {"channel": "general", "message": "@viktor what is up?"})
    assert "ask_ai_employee" in out
    assert not any(m["body"].startswith("@viktor") for m in state.messages)


@pytest.mark.asyncio
async def test_rejects_a_channel_that_would_404(h):
    out = await h.call("post_to_channel", {"channel": "random", "message": "hi"})
    assert "not a channel" in out


@pytest.mark.asyncio
async def test_warns_that_a_dm_to_an_ai_employee_is_a_dead_end(h):
    out = await h.call("direct_message", {"user_id": "ai-viktor", "message": "hello"})
    assert "never answered" in out
    assert not state.dms


# ---------------------------------------------------------------- guardrails

@pytest.mark.asyncio
async def test_staging_a_batch_dials_nobody(h):
    out = await h.call("stage_cold_call_batch", {"limit": 5})
    assert out["staged"] >= 1
    assert out["refused"]
    assert state.released == 0


@pytest.mark.asyncio
async def test_releasing_a_batch_requires_operate_and_then_really_calls(h):
    await h.call("stage_cold_call_batch", {"limit": 5})
    h.policy.set_autonomy("assist")
    out = await h.call("release_cold_call_batch")
    assert "NOT DONE" in out and state.released == 0

    h.policy.set_autonomy("operate")
    out = await h.call("release_cold_call_batch")
    assert out["approved"] >= 1 and state.released >= 1


@pytest.mark.asyncio
async def test_sandbox_blocks_the_dialler_even_at_autopilot():
    h = Harness(settings_for(ATLAS_SANDBOX="true", ATLAS_AUTONOMY="autopilot"))
    await h.call("stage_cold_call_batch", {"limit": 5})
    out = await h.call("release_cold_call_batch")
    assert "Sandbox mode is on" in out
    assert state.released == 0
    await h.close()


@pytest.mark.asyncio
async def test_turning_on_the_apps_own_calling_autonomy_always_needs_a_human():
    h = Harness(settings_for(ATLAS_AUTONOMY="autopilot"))
    out = await h.call("set_cold_call_autonomy", {"enabled": True, "daily_cap": 50})
    assert "held for the owner's approval" in out
    assert state.autonomy["enabled"] is False       # nothing changed
    assert h.approvals and h.approvals[0]["tool"] == "set_cold_call_autonomy"

    # ...and approving it actually performs the call.
    out = await h.call("set_cold_call_autonomy", {"enabled": True, "daily_cap": 50},
                       approved=True)
    assert state.autonomy["enabled"] is True
    await h.close()


@pytest.mark.asyncio
async def test_kill_switch_stops_action_but_not_observation(h):
    h.policy.kill_switch = True
    assert (await h.call("business_snapshot"))["totals"]["clients"] == 12
    out = await h.call("post_to_channel", {"channel": "general", "message": "hi"})
    assert "kill switch" in out.lower()


@pytest.mark.asyncio
async def test_blocked_calls_are_still_audited(h):
    h.policy.set_autonomy("observe")
    await h.call("post_to_channel", {"channel": "general", "message": "hi"})
    rows = await h.store["actions"].find({}, {"_id": 0}).to_list(10)
    assert rows and rows[0]["outcome"] == "deny"
    assert rows[0]["gate"] == "autonomy"


@pytest.mark.asyncio
async def test_rate_limit_survives_a_restart(h):
    h.policy.limits["chat"].limit = 2
    for _ in range(2):
        await h.call("post_to_channel", {"channel": "general", "message": "x"})
    out = await h.call("post_to_channel", {"channel": "general", "message": "x"})
    assert "cap is spent" in out

    # A restart must not hand back a fresh allowance. The counters were
    # persisted as the calls happened, so a new Policy that hydrates from
    # storage comes back already at the cap.
    import time
    persisted = await h.store["counters"].find({"bucket": "chat"}, {"_id": 0}).to_list(10)
    assert len(persisted) == 2, "allowed calls should have been persisted"

    fresh = Policy(h.settings, RateLimiter())
    fresh.limits["chat"].limit = 2
    fresh.limiter.hydrate("chat", [time.time() for _ in persisted])
    assert fresh.limiter.would_exceed(fresh.limits["chat"]), \
        "a restart handed back a fresh outreach allowance"


# ---------------------------------------------------------------- jobs

@pytest.mark.asyncio
async def test_runs_a_scheduled_job_with_the_cron_secret(h):
    out = await h.call("run_scheduled_job", {"job": "speed_to_lead"})
    assert out["result"]["ok"] is True
    assert "speed-to-lead/drain" in state.jobs_run[0]


@pytest.mark.asyncio
async def test_says_plainly_when_the_cron_secret_is_missing():
    h = Harness(settings_for(TWS_CRON_SECRET="", INTERNAL_CRON_SECRET=""))
    out = await h.call("run_scheduled_job", {"job": "workflows"})
    assert "cron secret" in str(out).lower()
    assert not state.jobs_run
    await h.close()


# ---------------------------------------------------------------- memory

@pytest.mark.asyncio
async def test_remembers_supersedes_and_recalls(h):
    a = await h.memory.remember(kind="lesson", title="Dentists convert badly",
                                body="0 of 35 touches", importance=4)
    await h.memory.remember(kind="lesson", title="Dentists convert fine after all",
                            body="4 of 40 with a new script", importance=4,
                            supersedes=a.id)
    mems = await h.memory.recall("dentists", limit=10)
    titles = [m.title for m in mems]
    assert "Dentists convert fine after all" in titles
    assert "Dentists convert badly" not in titles


@pytest.mark.asyncio
async def test_owner_directives_are_always_recalled(h):
    await h.memory.remember(kind="directive", title="Never call before 9am",
                            body="Owner said so", importance=5)
    for i in range(30):
        await h.memory.remember(kind="fact", title=f"noise {i}", body="filler")
    mems = await h.memory.recall("something unrelated entirely", limit=5)
    assert any(m.kind == "directive" for m in mems)
