"""Atlas Builder: the sweep, and the two routes that expose it.

The failure this guards against is a green dashboard. An audit that quietly
drops a client it could not check, or that invents progress for a service
nobody has built, is worse than no audit — somebody stops looking.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest

from tests.test_integration import MockStore, settings_for

from atlas.guardrails.policy import Policy, RateLimiter, Risk
from atlas.memory.store import MemoryStore
from atlas.tools import builder
from atlas.tools.registry import ToolContext, registry


def step(name, ok, detail="", fix=""):
    return {"name": name, "title": name, "ok": ok, "detail": detail, "fix": fix}


GREEN = [step("retell_key", True), step("number", True), step("live_call", True)]

BROKEN = [
    step("retell_key", True, "Retell answered."),
    step("disclosure", False,
         "The opening line does not mention recording.",
         "Super Admin > this client > AI Manager > Save."),
    step("live_call", False, "No call has ever arrived for this client.",
         "Ring the number from a mobile and run this again."),
]


class FakeClient:
    """Records every call, so the test can prove the audit only ever reads."""

    def __init__(self, clients, preflights, errors=()):
        self._clients = clients
        self._preflights = preflights
        self._errors = set(errors)
        self.calls: list = []

    async def get(self, path, **params):
        self.calls.append(("GET", path, params))
        if path == "/admin/clients":
            return {"clients": self._clients}
        if path == builder.RECEPTIONIST_PREFLIGHT:
            cid = params.get("client_id")
            if cid in self._errors:
                raise RuntimeError("HTTP 500: preflight blew up")
            return {"client_id": cid, "steps": self._preflights[cid]}
        raise AssertionError("unexpected GET %s" % path)

    async def post(self, path, body=None, **kw):
        self.calls.append(("POST", path, body or {}))
        raise AssertionError("the audit must not write: POST %s" % path)


def two_clients(**over):
    clients = [{"id": "c1", "business_name": "Cedar Plumbing"},
               {"id": "c2", "business_name": "Ace Roofing"}]
    pre = {"c1": BROKEN, "c2": GREEN}
    return FakeClient(clients, pre, **over)


@pytest.fixture(autouse=True)
def _no_commerce_url():
    before = os.environ.pop("COMMERCE_URL", None)
    yield
    if before is not None:
        os.environ["COMMERCE_URL"] = before


# ------------------------------------------------------------------ the sweep

@pytest.mark.asyncio
async def test_a_failing_step_appears_with_the_apps_own_fix_text():
    audit = await run_audit_on(two_clients())
    c1 = next(r for r in audit["clients"] if r["client_id"] == "c1")
    assert c1["ok"] is False
    fixes = [p["fix"] for p in c1["problems"]]
    assert "Super Admin > this client > AI Manager > Save." in fixes
    details = [p["detail"] for p in c1["problems"]]
    assert "The opening line does not mention recording." in details
    assert all(p["system"] == "receptionist" for p in c1["problems"])


@pytest.mark.asyncio
async def test_an_all_green_client_has_no_problems():
    audit = await run_audit_on(two_clients())
    c2 = next(r for r in audit["clients"] if r["client_id"] == "c2")
    assert c2["ok"] is True and c2["problems"] == []
    assert c2["name"] == "Ace Roofing"


@pytest.mark.asyncio
async def test_the_unproved_step_is_not_the_same_alarm_as_a_broken_one():
    """"Correct but never proved by a real call" is a real finding and not a
    fault; flattening the two makes the list unreadable."""
    audit = await run_audit_on(two_clients())
    c1 = next(r for r in audit["clients"] if r["client_id"] == "c1")
    sev = {p["detail"]: p["severity"] for p in c1["problems"]}
    assert sev["The opening line does not mention recording."] == "high"
    assert sev["No call has ever arrived for this client."] == "medium"


@pytest.mark.asyncio
async def test_a_client_that_cannot_be_checked_becomes_a_problem_not_a_crash():
    """One broken preflight must not hide every other client."""
    audit = await run_audit_on(two_clients(errors={"c1"}))
    assert audit["summary"]["clients"] == 2
    c1 = next(r for r in audit["clients"] if r["client_id"] == "c1")
    assert c1["ok"] is False
    assert "could not be read" in c1["problems"][0]["detail"]


@pytest.mark.asyncio
async def test_the_summary_counts_what_the_rows_say():
    audit = await run_audit_on(two_clients())
    assert audit["summary"] == {"clients": 2, "with_problems": 1, "problems": 2}
    assert set(audit) == {"ran_at", "clients", "custom_tier", "summary"}
    assert set(audit["clients"][0]) == {"client_id", "name", "ok", "problems"}


@pytest.mark.asyncio
async def test_the_audit_never_calls_a_write_endpoint():
    client = two_clients()
    await run_audit_on(client)
    assert all(m == "GET" for m, _, _ in client.calls), \
        "the audit wrote to the app: %r" % client.calls


# ----------------------------------------------------------------- persistence

@pytest.mark.asyncio
async def test_the_sweep_is_stored_and_read_back_newest_first():
    store = MockStore()
    await builder.run_audit(two_clients(), store)
    await builder.run_audit(two_clients(errors={"c1", "c2"}), store)
    latest = await builder.latest_audit(store)
    assert latest["ran_at"]
    assert latest["summary"]["with_problems"] == 2, "read back an older sweep"


@pytest.mark.asyncio
async def test_latest_with_nothing_stored_is_honestly_empty():
    latest = await builder.latest_audit(MockStore())
    assert latest["ran_at"] is None and latest["clients"] == []
    assert latest["custom_tier"] == builder.NOT_DEPLOYED


# ---------------------------------------------------------------- custom tier

@pytest.mark.asyncio
async def test_no_commerce_url_reports_not_deployed_in_those_exact_words():
    section = await builder.custom_tier_section()
    assert section == {
        "deployed": False,
        "detail": ("commerce service is not deployed; discovery questions "
                   "(store platform, channels, support cost) unanswered"),
    }


@pytest.mark.asyncio
async def test_a_deployed_commerce_service_is_reported_as_it_answers(monkeypatch):
    os.environ["COMMERCE_URL"] = "https://commerce.test"

    async def fake(url):
        assert url == "https://commerce.test"
        return {"ok": True, "adapters": []}

    monkeypatch.setattr(builder, "_fetch_readiness", fake)
    section = await builder.custom_tier_section()
    assert section["deployed"] is True and section["readiness"]["ok"] is True


@pytest.mark.asyncio
async def test_an_unreachable_commerce_service_is_not_reported_as_ready(monkeypatch):
    os.environ["COMMERCE_URL"] = "https://commerce.test"

    async def boom(url):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(builder, "_fetch_readiness", boom)
    section = await builder.custom_tier_section()
    assert section["reachable"] is False and "connection refused" in section["detail"]


# ---------------------------------------------------------------------- tool

def test_the_tool_is_a_read():
    tool = registry.get("audit_all_clients")
    assert tool is not None and tool.policy.risk is Risk.READ
    assert tool.policy.required_level() == "observe"


@pytest.mark.asyncio
async def test_the_tool_runs_through_dispatch():
    settings = settings_for(ATLAS_AUTONOMY="observe", ATLAS_SANDBOX="true")
    store = MockStore()
    client = two_clients()
    ctx = ToolContext(client=client, store=store, memory=MemoryStore(store),
                      policy=Policy(settings, RateLimiter()), settings=settings)
    result = await registry.dispatch(ctx, "audit_all_clients", {})
    assert result["summary"]["clients"] == 2


# -------------------------------------------------------------------- routes

def _app_and_key():
    import atlas.main as main
    object.__setattr__(main.settings, "console_api_key", "console-key")
    return main


async def _request(main, method, path, headers=None, **kw):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://atlas.test") as http:
        return await http.request(method, path, headers=headers or {}, **kw)


@pytest.mark.asyncio
async def test_builder_latest_without_the_key_is_refused():
    main = _app_and_key()
    resp = await _request(main, "GET", "/builder/latest")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_builder_run_without_the_key_is_refused():
    main = _app_and_key()
    resp = await _request(main, "POST", "/builder/run")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_the_routes_run_and_read_back_the_same_sweep():
    main = _app_and_key()
    store = MockStore()
    before_client, before_store = main.runtime.client, main.runtime.store
    main.runtime.client, main.runtime.store = two_clients(), store
    try:
        headers = {"X-Atlas-Key": "console-key"}
        ran = await _request(main, "POST", "/builder/run", headers=headers)
        assert ran.status_code == 200
        body = ran.json()
        assert body["summary"] == {"clients": 2, "with_problems": 1, "problems": 2}

        got = await _request(main, "GET", "/builder/latest", headers=headers)
        assert got.status_code == 200
        assert got.json()["ran_at"] == body["ran_at"]
    finally:
        main.runtime.client, main.runtime.store = before_client, before_store


@pytest.mark.asyncio
async def test_builder_run_says_so_when_atlas_cannot_reach_the_app():
    main = _app_and_key()
    before = main.runtime.client
    main.runtime.client = None
    try:
        resp = await _request(main, "POST", "/builder/run",
                              headers={"X-Atlas-Key": "console-key"})
        assert resp.status_code == 503
    finally:
        main.runtime.client = before


# ------------------------------------------------------------------- helpers

async def run_audit_on(client):
    return await builder.run_audit(client, MockStore())
