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

    #: Optional per-test hook: (path, payload) -> the app's JSON answer.
    answer_post = None

    async def post(self, path, body=None, **kw):
        self.calls.append(("POST", path, body or {}))
        if self.answer_post is not None:
            return self.answer_post(path, body or {})
        if path.endswith("/send"):
            # The real route answers an outcome word, never a bare ok.
            return {"outcome": "sent", "detail": "Sent and recorded on the prospect."}
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
    """Historical name. SEND_ENDPOINTS now points at the real route,
    POST /admin/prospects/{id}/send, and the fake client answers it the way
    the app does -- with an outcome word. Kept so the send-path tests read
    as what they are: tests of the wired path."""
    before = dict(outreach.SEND_ENDPOINTS)
    assert all(v and v.endswith("/send") for v in before.values()), before
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
async def test_the_send_is_honest_when_the_app_cannot_send(monkeypatch):
    """The endpoints are wired now; this keeps the honesty for the day one
    is not (a route renamed, a deploy behind). An unwired channel says so
    and writes nothing, rather than logging a message that never left."""
    monkeypatch.setitem(outreach.SEND_ENDPOINTS, "email", None)
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
async def test_a_real_send_is_recorded_by_the_route_not_by_atlas(wired):
    """POST /admin/prospects/{id}/send records the CRM touch itself. Atlas
    posting a second touch would turn one message into two rows and start
    the follow-up clock twice, so it must not."""
    client = FakeClient()
    ctx = harness(client)
    result = await outreach._run(ctx, ["p1", "p2", "p3"], "email", "Subject", "Body")
    sends = [p for p in client.paths("POST") if p.endswith("/send")]
    assert "/admin/prospects/p1/send" in sends and "/admin/prospects/p2/send" in sends
    assert "/admin/prospects/p3/send" not in sends          # opted out, never sent
    assert not any(p.endswith("/touch") for p in client.paths("POST"))
    assert result["sent"] == 2


async def test_the_apps_outcome_words_are_read_not_assumed(wired):
    """The route answers a word, not a boolean: suppressed and no_address and
    failed are three different reasons nothing went out. An answer the tool
    cannot read counts as failed, never as sent."""
    client = FakeClient(prospects=[
        {"id": pid, "name": "Clean Co " + pid, "phone": "+1555222000%d" % n,
         "email": "%s@clean.test" % pid, "status": "new"}
        for n, pid in enumerate(("p1", "p2", "p4", "p5"))])
    ctx = harness(client)
    answers = {"p1": {"outcome": "suppressed", "detail": "asked not to be contacted"},
               "p2": {"outcome": "no_address", "detail": "no phone on file"},
               "p4": {"outcome": "sent", "detail": "Sent and recorded on the prospect."},
               "p5": {"something": "else"}}
    client.answer_post = lambda path, payload: answers[path.split("/")[3]]
    result = await outreach._run(ctx, ["p1", "p2", "p4", "p5"], "sms", "", "Body")
    by_id = {r["prospect_id"]: r for r in result["results"]}
    assert by_id["p1"]["status"] == "opted_out" and "asked not" in by_id["p1"]["reason"]
    assert by_id["p2"]["status"] == "failed" and "no phone" in by_id["p2"]["reason"]
    assert by_id["p4"]["status"] == "sent"
    assert by_id["p5"]["status"] == "failed" and "readable outcome" in by_id["p5"]["reason"]
    assert result["sent"] == 1


@pytest.mark.asyncio
async def test_every_send_in_a_batch_spends_the_hourly_cap(wired):
    """dispatch counts one event per call; a batch of forty must not cost one."""
    client = FakeClient()
    ctx = harness(client)
    await call(ctx, "send_outreach_batch", {
        "prospect_ids": ["p1", "p2"], "channel": "sms",
        "template_id_or_body": "Hello"})
    assert ctx.policy.limiter.used(ctx.policy.limits["outreach"]) == 2


# --------------------------------------------------------- the prose is a claim

def test_no_description_claims_the_endpoint_is_missing():
    """The most dangerous defect this module has had, and it lived in
    prose rather than in code.

    For a few hours SEND_ENDPOINTS pointed at the live route while the module
    docstring and all three tool descriptions still told the model that no send
    endpoint existed, that every row came back "not_wired", and that nothing
    reached the CRM. Those sentences are what the model reads before deciding
    whether to fire. A fifty-row batch described as inert is one a model has
    positive reason to run, and the same descriptions instructed it to report
    the result as un-sent -- so fifty real messages could have been summarised
    to the owner as nothing having happened.

    The endpoints and the words about them are one claim. This fails when they
    disagree.
    """
    wired_channels = {c for c, v in outreach.SEND_ENDPOINTS.items() if v}
    assert wired_channels, "this test is meaningless if nothing is wired"

    denials = ("NO HTTP endpoint", "no HTTP endpoint", "sends nothing",
               "returns 'not_wired'", "comes back 'not_wired'",
               "do not log a touch for it")
    for name in ("send_prospect_email", "send_prospect_sms", "send_outreach_batch"):
        tool = registry.get(name)
        assert tool is not None, name
        text = tool.description
        for phrase in denials:
            assert phrase not in text, (
                "%s still tells the model %r while SEND_ENDPOINTS is wired for %s"
                % (name, phrase, sorted(wired_channels)))
        assert "SENDS FOR REAL" in text, name
        assert "dry-run" in text, name

    doc = outreach.__doc__ or ""
    for phrase in ("There is no route of the shape", "declared and empty",
                   "reachable"):
        assert phrase not in doc, (
            "the module docstring still denies the route: %r" % phrase)


@pytest.mark.asyncio
async def test_a_repeated_id_is_messaged_once(wired):
    """A batch is a list somebody approved. If an id appears twice, they
    approved one message to that prospect, not two -- and nothing downstream
    would catch it: the app route does not read previous touches before it
    sends."""
    client = FakeClient(prospects=[
        {"id": pid, "name": "Clean Co " + pid, "phone": "+1555333000%d" % n,
         "email": "%s@clean.test" % pid, "status": "new"}
        for n, pid in enumerate(("p1", "p2", "p4"))])
    ctx = harness(client)
    result = await outreach._run(ctx, ["p1", "p1", "p2", "p1"], "sms", "", "Hi")
    sends = [p for p in client.paths("POST") if p.endswith("/send")]
    assert sends.count("/admin/prospects/p1/send") == 1
    assert result["requested"] == 2
    assert "repeated id" in result["note"]


@pytest.mark.asyncio
async def test_a_send_with_no_answer_is_not_a_failure_to_retry(wired):
    """A read timeout on a POST the app completed leaves a message already
    delivered. Calling that "failed" invites a retry, and the retry messages
    the same stranger a second time. TWSClient never auto-retries a POST
    precisely so that this decision is made deliberately, here."""
    client = FakeClient(prospects=[
        {"id": pid, "name": "Clean Co " + pid, "phone": "+1555333000%d" % n,
         "email": "%s@clean.test" % pid, "status": "new"}
        for n, pid in enumerate(("p1", "p2", "p4"))])
    ctx = harness(client)

    async def timeout_post(path, body=None, **kw):
        client.calls.append(("POST", path, body or {}))
        raise TimeoutError("read timeout")
    client.post = timeout_post

    result = await outreach._run(ctx, ["p1"], "sms", "", "Hi")
    row = result["results"][0]
    assert row["status"] == outreach.UNKNOWN_STATUS
    assert row["status"] != "failed"
    assert "MAY have been delivered" in row["reason"]
    assert "do not retry" in row["reason"].lower()
    assert result["unknown"] == 1 and result["sent"] == 0 and result["failed"] == 0
    assert "Do NOT retry" in result["note"]


@pytest.mark.asyncio
async def test_the_apps_own_opt_out_is_counted_as_an_opt_out(wired):
    """The pre-check and the route's own check establish the same fact: this
    person asked us to stop. Counting the second as a failure reads as
    "something went wrong, try again" for the one outcome that must never be
    retried."""
    client = FakeClient(prospects=[
        {"id": pid, "name": "Clean Co " + pid, "phone": "+1555333000%d" % n,
         "email": "%s@clean.test" % pid, "status": "new"}
        for n, pid in enumerate(("p1", "p2", "p4"))])
    ctx = harness(client)
    client.answer_post = lambda path, payload: {
        "outcome": "suppressed", "detail": "asked not to be contacted"}
    result = await outreach._run(ctx, ["p1"], "sms", "", "Hi")
    assert result["results"][0]["status"] == "opted_out"
    assert result["skipped_opt_out"] == 1
    assert result["failed"] == 0


@pytest.mark.asyncio
async def test_every_send_in_a_batch_survives_a_restart(wired):
    """The in-memory limiter was topped up per send, so the hourly cap held --
    until a restart rebuilt the hour from storage, found ONE row for a
    forty-send batch, and handed back an allowance already spent on real
    people."""
    client = FakeClient(prospects=[
        {"id": pid, "name": "Clean Co " + pid, "phone": "+1555333000%d" % n,
         "email": "%s@clean.test" % pid, "status": "new"}
        for n, pid in enumerate(("p1", "p2", "p4"))])
    ctx = harness(client)
    rows = []

    class Counters:
        async def insert_one(self, doc):
            rows.append(doc)

    class Store:
        def __getitem__(self, name):
            assert name == "counters", name
            return Counters()

    ctx.store = Store()
    result = await outreach._run(ctx, ["p1", "p2", "p4"], "sms", "", "Hi")
    assert result["sent"] == 3, result
    # dispatch persists one row for the call itself; the rest are written here,
    # so the durable count matches what actually went out.
    assert len(rows) == result["sent"] - 1
    assert {r["bucket"] for r in rows} == {"outreach"}
