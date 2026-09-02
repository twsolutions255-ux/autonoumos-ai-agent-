"""The send tools: what they refuse, what they skip, and what they never fake.

These go through `registry.dispatch`, not the handlers directly, because the
gate is half of what is being tested. A send tool that works when called
directly and is unreachable through the one execution path would be worse than
no tool at all.

The app has no prospect-send endpoint today, so a real send cannot be exercised
against it. The success path is proved by pointing SEND_ENDPOINTS at a route
the fake client answers — which tests Atlas's logic (the CRM touch, the
counting) without pretending the app can do something it cannot.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tests.test_integration import MockStore, settings_for

from atlas.guardrails.policy import Policy, RateLimiter, Risk
from atlas.memory.store import MemoryStore
from atlas.tools import outreach  # noqa: F401  — registers the tools
from atlas.tools.registry import ToolContext, registry


PROSPECTS = [
    {"id": "p1", "name": "Ace Roofing", "phone": "+15551110000",
     "email": "ace@example.test", "status": "new"},
    {"id": "p2", "name": "Bright HVAC", "phone": "+15551110001",
     "email": "bright@example.test", "status": "new"},
    {"id": "p3", "name": "Cedar Plumbing", "phone": "+15551110002",
     "email": "cedar@example.test", "status": "dnc"},
]


class FakeClient:
    """Stands in for TWSClient. Records every call so the tests can assert on
    what was NOT done, which is most of the point here."""

    def __init__(self, prospects=None, suppressed=(), dnc_errors=()):
        self.prospects = list(PROSPECTS if prospects is None else prospects)
        self.suppressed = set(suppressed)      # phones/emails on the list
        self.dnc_errors = set(dnc_errors)      # phones whose check blows up
        self.calls: list = []

    async def get(self, path, **params):
        self.calls.append(("GET", path, params))
        if path == "/admin/prospects":
            return {"count": len(self.prospects), "prospects": self.prospects}
        if path == "/dnc/check":
            phone = params.get("phone") or ""
            if phone in self.dnc_errors:
                raise RuntimeError("HTTP 502: the suppression list is down")
            hit = bool(phone in self.suppressed
                       or (params.get("email") or "") in self.suppressed)
            return {"suppressed": hit, "entry": None,
                    "checked": {"phone": phone, "email": params.get("email")}}
        raise AssertionError("unexpected GET %s" % path)

    async def post(self, path, body=None, **kw):
        self.calls.append(("POST", path, body or {}))
        return {"ok": True}

    def paths(self, method=None):
        return [p for m, p, _ in self.calls if method is None or m == method]


def harness(client, autonomy="operate"):
    settings = settings_for(ATLAS_AUTONOMY=autonomy, ATLAS_SANDBOX="false")
    store = MockStore()
    return ToolContext(client=client, store=store, memory=MemoryStore(store),
                       policy=Policy(settings, RateLimiter()), settings=settings,
                       cycle_id="cyc-test")


async def call(ctx, name, args):
    return await registry.dispatch(ctx, name, args)


@pytest.fixture
def wired():
    """Point the send endpoints at a route the fake client answers."""
    before = dict(outreach.SEND_ENDPOINTS)
    outreach.SEND_ENDPOINTS.update({
        "email": "/admin/prospects/{prospect_id}/send-email",
        "sms": "/admin/prospects/{prospect_id}/send-sms",
    })
    yield
    outreach.SEND_ENDPOINTS.clear()
    outreach.SEND_ENDPOINTS.update(before)


# ------------------------------------------------------------- the risk class

@pytest.mark.parametrize("name", ["send_prospect_email", "send_prospect_sms",
                                  "send_outreach_batch"])
def test_every_send_tool_is_external_comms(name):
    """The class is the gate. Anything softer and these run unattended at
    'assist', which is the rung the owner is told changes nothing outside."""
    tool = registry.get(name)
    assert tool is not None, "%s is not registered" % name
    assert tool.policy.risk is Risk.EXTERNAL_COMMS
    assert tool.policy.required_level() == "operate"


@pytest.mark.parametrize("name", ["send_prospect_email", "send_prospect_sms",
                                  "send_outreach_batch"])
def test_the_description_tells_the_model_not_to_retry(name):
    """A queued call reads as a failure unless the tool says otherwise, and a
    model that reads it as failure sends the same message again later."""
    assert "queues for the owner's approval below operate" in \
        registry.get(name).description


# ------------------------------------------------------------------ the gate

@pytest.mark.asyncio
async def test_nothing_is_sent_when_the_policy_denies():
    client = FakeClient()
    ctx = harness(client, autonomy="recommend")
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p1"], "channel": "sms",
        "template_id_or_body": "hello"})
    assert "NOT DONE" in result
    assert client.calls == [], "a denied send still talked to the app"


@pytest.mark.asyncio
async def test_at_operate_the_same_call_reaches_the_app():
    client = FakeClient()
    ctx = harness(client, autonomy="operate")
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p1"], "channel": "sms",
        "template_id_or_body": "hello"})
    assert isinstance(result, dict)
    assert client.calls, "the permitted send never reached the app"
    assert "/admin/prospects" in client.paths("GET")


# ---------------------------------------------------------------- the ceiling

@pytest.mark.asyncio
async def test_a_batch_over_fifty_is_refused_before_any_call():
    client = FakeClient()
    ctx = harness(client)
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": [f"p{i}" for i in range(51)], "channel": "email",
        "template_id_or_body": "hi", "subject": "hi"})
    assert "51" in result["refused"] and "50" in result["refused"]
    assert client.calls == [], "51 prospects were refused only after contacting people"


@pytest.mark.asyncio
async def test_fifty_is_allowed():
    """The boundary, so the ceiling is a ceiling and not an off-by-one."""
    client = FakeClient(prospects=[{"id": f"p{i}", "phone": f"+1555111{i:04d}",
                                    "status": "new"} for i in range(50)])
    ctx = harness(client)
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": [f"p{i}" for i in range(50)], "channel": "sms",
        "template_id_or_body": "hi"})
    assert "refused" not in result
    assert result["requested"] == 50


# ----------------------------------------------------------------- opting out

@pytest.mark.asyncio
async def test_a_prospect_on_the_suppression_list_is_skipped_and_counted():
    client = FakeClient(suppressed={"+15551110001"})
    ctx = harness(client)
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p1", "p2"], "channel": "sms",
        "template_id_or_body": "hi"})
    assert result["skipped_opt_out"] == 1
    row = next(r for r in result["results"] if r["prospect_id"] == "p2")
    assert row["status"] == "skipped_opt_out"
    assert "do-not-contact" in row["reason"]


@pytest.mark.asyncio
async def test_a_prospect_the_app_marks_dnc_is_skipped_without_asking_twice():
    client = FakeClient()
    ctx = harness(client)
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p3"], "channel": "sms", "template_id_or_body": "hi"})
    assert result["skipped_opt_out"] == 1
    assert "/dnc/check" not in client.paths("GET")


@pytest.mark.asyncio
async def test_a_check_that_could_not_be_made_is_not_treated_as_permission():
    """'I could not ask' must never collapse into 'they are not on it'."""
    client = FakeClient(prospects=[{"id": "p9", "name": "No Contact Co",
                                    "status": "new"}])
    ctx = harness(client)
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p9"], "channel": "sms", "template_id_or_body": "hi"})
    assert result["sent"] == 0 and result["failed"] == 1
    assert "could not be checked" in result["results"][0]["reason"]


# ------------------------------------------------------- per-prospect failure

@pytest.mark.asyncio
async def test_one_failing_app_call_does_not_take_the_batch_down():
    client = FakeClient(dnc_errors={"+15551110000"})
    ctx = harness(client)
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p1", "p2"], "channel": "sms",
        "template_id_or_body": "hi"})
    assert result["requested"] == 2 and len(result["results"]) == 2
    bad = next(r for r in result["results"] if r["prospect_id"] == "p1")
    assert bad["status"] == "failed" and "suppression list is down" in bad["reason"]
    # The other prospect was still processed rather than abandoned.
    other = next(r for r in result["results"] if r["prospect_id"] == "p2")
    assert other["status"] != "failed" or "suppression list" not in other["reason"]


@pytest.mark.asyncio
async def test_an_unknown_prospect_is_reported_not_raised():
    client = FakeClient()
    ctx = harness(client)
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["nope"], "channel": "email", "subject": "s",
        "template_id_or_body": "hi"})
    assert result["failed"] == 1
    assert "no such prospect" in result["results"][0]["reason"]


# ------------------------------------------------------------- not-wired path

@pytest.mark.asyncio
async def test_the_send_is_honest_that_the_app_cannot_send():
    client = FakeClient()
    ctx = harness(client)
    result = await call(ctx, "send_prospect_email", {
        "prospect_id": "p1", "subject": "Your site", "body": "Hello"})
    assert result["sent"] == 0 and result["failed"] == 1
    assert result["results"][0]["status"] == "not_wired"
    assert "no endpoint" in result["results"][0]["reason"]
    # And crucially, nothing was written to the CRM for a message never sent.
    assert client.paths("POST") == []


@pytest.mark.asyncio
async def test_the_sms_tool_reports_counts_never_a_bare_ok():
    client = FakeClient()
    ctx = harness(client)
    result = await call(ctx, "send_prospect_sms", {"prospect_id": "p1",
                                                   "body": "Hello"})
    assert set(["sent", "skipped_opt_out", "failed", "results"]) <= set(result)


# ------------------------------------------------------------ the CRM write

@pytest.mark.asyncio
async def test_a_real_send_writes_the_crm_touch(wired):
    """The moment a send endpoint exists, the follow-up clock must start —
    otherwise the same prospect is worked twice."""
    client = FakeClient()
    ctx = harness(client)
    result = await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p1", "p2", "p3"], "channel": "sms",
        "template_id_or_body": "Hello"})
    assert result["sent"] == 2 and result["skipped_opt_out"] == 1
    assert "/admin/prospects/p1/touch" in client.paths("POST")
    assert "/admin/prospects/p2/touch" in client.paths("POST")
    # The opted-out one was neither sent nor logged.
    assert "/admin/prospects/p3/touch" not in client.paths("POST")


@pytest.mark.asyncio
async def test_every_send_in_a_batch_spends_the_hourly_cap(wired):
    """dispatch counts one event per call; a batch of forty must not cost one."""
    client = FakeClient()
    ctx = harness(client)
    await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p1", "p2"], "channel": "sms",
        "template_id_or_body": "Hello"})
    assert ctx.policy.limiter.used(ctx.policy.limits["outreach"]) == 2
