"""Atlas Builder: one sweep that asks every client's systems whether they work.

The app can already answer "is THIS client's receptionist working end to end?"
— honestly, and in detail, with a fix written for each failing step. What
nobody does is ask it for every client, on a schedule, and keep the answer. So
a client whose Retell agent was deleted, or whose webhook started being
rejected, stays broken until they complain.

This is that sweep. It is deliberately thin: every judgement in it comes from
the app's own preflight, including the wording of the fix. Atlas adds no
opinion of its own about what is wrong, because a second opinion computed from
the same data is how two screens end up disagreeing and nobody knows which to
believe.

WHICH PREFLIGHTS ARE USED, AND WHY ONLY ONE
-------------------------------------------
``GET /admin/receptionist/end-to-end?client_id=<id>`` is the only preflight in
the app that is scoped to a client. The others —

    GET /admin/sms/preflight            Telnyx, one account for everyone
    GET /admin/cold-calling/preflight   the dialler, agency-wide
    GET /admin/meta/status              the Meta connection, agency-wide
    GET /admin/agents/health            the AI employees, agency-wide

— are account-wide. Attributing an agency-wide fault to each client in turn
would report one broken Telnyx account as forty broken clients, so they are
left out rather than invented into a per-client shape. (The receptionist
preflight already folds the SMS answer in as one of its steps, so the Telnyx
half does reach this report — attached to the step it actually affects.)

CUSTOM TIER
-----------
The Custom Tier commerce service lives in its own repo with its own database.
If ``COMMERCE_URL`` is set, its ``/readiness`` is read and reported as given.
If it is not set, this reports it as not deployed, in exactly those words. It
does not estimate progress, because the discovery questions behind it — which
store platforms, which channels, what support costs — have no answers yet, and
a percentage invented for a dashboard is worse than a blank.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..db import iso
from ..guardrails.policy import Risk
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.builder")

#: The one client-scoped preflight the app has.
RECEPTIONIST_PREFLIGHT = "/admin/receptionist/end-to-end"

#: The step in that preflight which means "correct but never proved by a real
#: call". Genuinely worth surfacing, and genuinely not the same as broken.
LIVE_CALL_STEP = "live_call"

#: Where the sweep is kept, so /builder/latest can answer without re-running it.
COLLECTION = "builder_audits"

NOT_DEPLOYED = {
    "deployed": False,
    "detail": ("commerce service is not deployed; discovery questions "
               "(store platform, channels, support cost) unanswered"),
}


def _custom_tier_not_deployed() -> dict:
    return dict(NOT_DEPLOYED)


# --------------------------------------------------------------------------
# custom tier
# --------------------------------------------------------------------------

async def _fetch_readiness(url: str) -> Any:
    """GET <COMMERCE_URL>/readiness. Split out so tests can stand in for it."""
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as http:
        resp = await http.get(url.rstrip("/") + "/readiness")
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text[:2000]}


async def custom_tier_section() -> dict:
    url = (os.environ.get("COMMERCE_URL") or "").strip()
    if not url:
        return _custom_tier_not_deployed()
    try:
        readiness = await _fetch_readiness(url)
    except Exception as e:
        return {"deployed": True, "url": url, "reachable": False,
                "detail": f"COMMERCE_URL is set but /readiness could not be read: "
                          f"{str(e)[:200]}"}
    return {"deployed": True, "url": url, "reachable": True, "readiness": readiness}


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------

def _rows(data: Any, key: str) -> list:
    """The app returns a bare list on some routes and {key: [...]} on others."""
    if isinstance(data, dict):
        return data.get(key) or []
    return list(data or [])


def _problems_from_preflight(pre: dict) -> list:
    """Every failing step, with the app's own fix text carried through."""
    out = []
    for step in (pre.get("steps") or []):
        if step.get("ok"):
            continue
        out.append({
            "system": "receptionist",
            # A configuration fault is broken now; the live-call step means
            # "nothing has proved this works", which is a real finding and not
            # the same alarm.
            "severity": "medium" if step.get("name") == LIVE_CALL_STEP else "high",
            "detail": step.get("detail") or step.get("title") or "",
            "fix": step.get("fix") or "",
        })
    return out


async def run_audit(client, store=None) -> dict:
    """Check every client, and remember the answer.

    Reads only. One client that cannot be checked becomes one problem on that
    client, never an exception that hides the other thirty-nine.
    """
    clients = _rows(await client.get("/admin/clients"), "clients")

    rows: list = []
    for c in clients:
        cid = str(c.get("id") or "")
        name = c.get("business_name") or c.get("name") or cid
        if not cid:
            continue
        try:
            pre = await client.get(RECEPTIONIST_PREFLIGHT, client_id=cid)
            problems = _problems_from_preflight(pre if isinstance(pre, dict) else {})
        except Exception as e:
            problems = [{
                "system": "receptionist",
                "severity": "high",
                "detail": f"The receptionist preflight could not be read for this "
                          f"client: {str(e)[:200]}",
                "fix": "Open this client in Super Admin and run the end-to-end check "
                       "by hand; the sweep could not reach it.",
            }]
        rows.append({"client_id": cid, "name": name,
                     "ok": not problems, "problems": problems})

    audit = {
        "ran_at": iso(),
        "clients": rows,
        "custom_tier": await custom_tier_section(),
        "summary": {
            "clients": len(rows),
            "with_problems": sum(1 for r in rows if r["problems"]),
            "problems": sum(len(r["problems"]) for r in rows),
        },
    }

    if store is not None:
        try:
            await store[COLLECTION].insert_one(dict(audit))
        except Exception:
            # A sweep that ran is worth returning even if it could not be kept.
            log.exception("atlas: could not store the builder audit")
    return audit


async def latest_audit(store=None) -> dict:
    """The newest stored sweep, or an honest empty one."""
    empty = {"ran_at": None, "clients": [],
             "custom_tier": _custom_tier_not_deployed(),
             "summary": {"clients": 0, "with_problems": 0, "problems": 0}}
    if store is None:
        return empty
    try:
        # _id breaks the tie: two sweeps in the same millisecond carry the same
        # ran_at on a coarse clock, and "newest" then silently means "first".
        doc = await store[COLLECTION].find_one({}, {"_id": 0},
                                               sort=[("ran_at", -1), ("_id", -1)])
    except Exception:
        log.exception("atlas: could not read the builder audits")
        return empty
    return doc or empty


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------

@registry.tool(
    "audit_all_clients",
    group="clientcare",
    risk=Risk.READ,
    description="""Atlas Builder: check EVERY client's receptionist and AI systems in one
sweep, and list what is broken with the app's own fix for each.
This is the question nobody asks often enough — not "is this client fine" but "which of
my clients are not". A deleted Retell agent, a rejected webhook or a missing recording
disclosure looks identical to healthy from any dashboard pill, and stays that way until
the client complains.
Reads only; it changes nothing. Each problem carries the fix text written by the app,
so quote it rather than inventing your own remedy. Also reports the Custom Tier commerce
service, which says plainly that it is not deployed rather than estimating progress.
The result is stored, so the console can show the last sweep without re-running it.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def audit_all_clients(ctx: ToolContext) -> dict:
    return await run_audit(ctx.client, ctx.store)
