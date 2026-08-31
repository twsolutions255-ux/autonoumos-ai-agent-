"""Seeing the business: the read surface Atlas reasons from.

Two kinds of tool live here. `business_snapshot` is the wide one — it fans out
across a dozen endpoints concurrently and returns the whole state of the
company in a single tool call, because an agent that needs fifteen sequential
round trips before it can think is an agent that is always out of date. The
rest are narrow, for when the snapshot raises a question worth chasing.

Every read here tolerates failure individually. A half-configured deploy (no
Meta connection, no Render key) must still yield a usable picture rather than
one exception taking the whole snapshot down — and where something *is*
missing, the snapshot says so explicitly instead of quietly omitting it, so
Atlas never mistakes "not wired up" for "zero".
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..guardrails.policy import Risk
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.observe")


async def _try(coro, label: str) -> Any:
    """Run one read; turn any failure into a value the model can interpret."""
    try:
        return await coro
    except Exception as e:
        status = getattr(e, "status", None)
        if status == 503:
            return {"unavailable": f"{label} is not configured in this deployment."}
        if status == 403:
            return {"unavailable": f"Atlas is not allowed to read {label} (403)."}
        return {"unavailable": f"Could not read {label}: {e}"}


@registry.tool(
    "business_snapshot",
    group="observe",
    risk=Risk.READ,
    description="""The whole company in one call: clients, calls, demos, revenue, the
lead pipeline, team activity, live alerts, integration health and AI spend.
Start almost every cycle with this. It is one round trip, so prefer it over calling
five narrower tools. Anything the deployment has not configured is reported as
'unavailable' rather than as a zero — never read a missing integration as bad
performance.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def business_snapshot(ctx: ToolContext) -> dict:
    c = ctx.client
    (overview, alerts, health, integrations, leadstats, leaderboard,
     bookings, clients, ai_usage, opportunities, deals, prospects) = await asyncio.gather(
        _try(c.get("/admin/overview"), "overview"),
        _try(c.get("/admin/alerts"), "alerts"),
        _try(c.get("/admin/system-health"), "system health"),
        _try(c.get("/admin/integrations"), "integrations"),
        _try(c.get("/admin/leads/stats", days=7), "team lead activity"),
        _try(c.get("/leaderboard"), "leaderboard"),
        _try(c.get("/admin/bookings/stats"), "demo bookings"),
        _try(c.get("/admin/clients"), "clients"),
        _try(c.get("/admin/ai-usage"), "AI usage"),
        _try(c.get("/admin/lead-opportunities"), "lead opportunities"),
        _try(c.get("/admin/deals/awaiting-setup"), "deals awaiting setup"),
        _try(c.get("/admin/prospects"), "prospects"),
    )

    alert_rows = alerts.get("alerts", []) if isinstance(alerts, dict) else []
    client_rows = clients.get("clients", clients) if isinstance(clients, dict) else clients

    return {
        "totals": overview,
        "alerts": {
            "count": len(alert_rows),
            "high": [a for a in alert_rows if a.get("severity") == "high"][:10],
            "other": [a for a in alert_rows if a.get("severity") != "high"][:10],
        } if alert_rows or isinstance(alerts, dict) else alerts,
        "system_health": health,
        "integrations": integrations,
        "team_activity_7d": leadstats,
        "leaderboard": leaderboard,
        "demo_bookings": bookings,
        "clients": {
            "count": len(client_rows) if isinstance(client_rows, list) else client_rows,
            "sample": (client_rows[:15] if isinstance(client_rows, list) else None),
        },
        "ai_spend": ai_usage,
        "lead_opportunities": opportunities,
        "deals_awaiting_setup": deals,
        "prospects": prospects,
        "how_to_read_this": (
            "'unavailable' means that integration is not configured — it is not a zero and "
            "not a failure of the business. deals_awaiting_setup is money already won that "
            "is not yet live, so it is usually the fastest revenue in the building."
        ),
    }


@registry.tool(
    "get_alerts",
    group="observe",
    risk=Risk.READ,
    description="""Everything currently wrong that the app knows about: failed webhooks,
clients with a receptionist but no phone number, and clients gone quiet for a week.
High-severity items here are usually worth more than any new lead — a client whose
receptionist has silently stopped working is churn that has already started.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_alerts(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/alerts")


@registry.tool(
    "get_clients",
    group="observe",
    risk=Risk.READ,
    description="""Every paying client with their status. Use it to find churn risk,
upsell candidates, and clients whose setup was never finished.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_clients(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/clients")


@registry.tool(
    "get_client_report",
    group="observe",
    risk=Risk.READ,
    description="""One client's full performance report: calls handled, appointments
booked and revenue recovered. This is the evidence for a renewal conversation, an
upsell, or catching a client who is quietly getting no value.

Defaults to the current month. Pass month as YYYY-MM for a historical one.""",
    schema={
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "month": {"type": "string",
                      "description": "YYYY-MM, e.g. 2026-07. Omit for this month."},
        },
        "required": ["client_id"],
        "additionalProperties": False,
    },
)
async def get_client_report(ctx: ToolContext, client_id: str, month: str = "") -> Any:
    """A historical month is now reachable, which it was not before.

    This tool used to declare only client_id and send no query parameters, while
    the endpoint required `month`. So every call it ever made returned 422 --
    unconditionally, from the day it was written.

    The endpoint now defaults the month, which is the actual fix and the one
    that also repaired client_value_report. Declaring `month` here as well is
    the smaller half: without it the parameter is unreachable, because _invoke
    drops any argument a handler does not declare. That is what made the old
    failure unrecoverable -- a model could read the 422, understand it, send
    `month`, and have the argument deleted before the request was built.
    """
    return await ctx.client.get(f"/admin/clients/{client_id}/report",
                                month=month or None)


@registry.tool(
    "get_finances",
    group="observe",
    risk=Risk.READ,
    description="""The money: books, payouts owed, and AI spend against budget.
Atlas's objective is profitable growth, so read the cost side before proposing anything
that increases it.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_finances(ctx: ToolContext) -> dict:
    c = ctx.client
    books, payouts, runs, ai = await asyncio.gather(
        _try(c.get("/admin/books"), "books"),
        _try(c.get("/admin/payouts"), "payouts"),
        _try(c.get("/admin/payouts/runs"), "payout runs"),
        _try(c.get("/admin/ai-usage"), "AI usage"),
    )
    return {"books": books, "payouts_owed": payouts, "payout_runs": runs, "ai_spend": ai}


@registry.tool(
    "get_pipeline",
    group="observe",
    risk=Risk.READ,
    description="""The acquisition pipeline end to end: the shared lead pool, prospects,
scheduled follow-ups, demo bookings, speed-to-lead health, and deals won but not yet
set up. This is where growth is won or lost — read it before touching outreach.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_pipeline(ctx: ToolContext) -> dict:
    c = ctx.client
    pool, prospects, followups, bookings, s2l, deals, nurture = await asyncio.gather(
        _try(c.get("/leads/pool"), "lead pool"),
        _try(c.get("/admin/prospects"), "prospects"),
        _try(c.get("/admin/prospects/followups"), "follow-ups"),
        _try(c.get("/admin/bookings/stats"), "demo bookings"),
        _try(c.get("/admin/speed-to-lead/status"), "speed to lead"),
        _try(c.get("/admin/deals/awaiting-setup"), "deals awaiting setup"),
        _try(c.get("/admin/demo-nurture"), "demo nurture"),
    )
    return {
        "lead_pool": pool, "prospects": prospects, "followups_due": followups,
        "demo_bookings": bookings, "speed_to_lead": s2l,
        "deals_awaiting_setup": deals, "demo_nurture": nurture,
        "note": ("Speed to lead is the highest-leverage number in this list: a lead "
                 "contacted in minutes converts far better than one contacted tomorrow."),
    }


@registry.tool(
    "get_team",
    group="observe",
    risk=Risk.READ,
    description="""The human team: roster, setters, closers, their activity this week and
the leaderboard. Read it before coaching anyone or reassigning work — and before
concluding a number is a people problem rather than a lead-quality problem.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_team(ctx: ToolContext) -> dict:
    c = ctx.client
    roster, setters, closers, activity, board, coaching = await asyncio.gather(
        _try(c.get("/admin/team/roster"), "roster"),
        _try(c.get("/admin/setters"), "setters"),
        _try(c.get("/admin/closers"), "closers"),
        _try(c.get("/admin/leads/stats", days=7), "activity"),
        _try(c.get("/leaderboard"), "leaderboard"),
        _try(c.get("/call-reviews/coaching"), "coaching"),
    )
    return {"roster": roster, "setters": setters, "closers": closers,
            "activity_7d": activity, "leaderboard": board, "coaching_signals": coaching}


@registry.tool(
    "get_marketing",
    group="observe",
    risk=Risk.READ,
    description="""Marketing and ads: campaign performance, Meta ad insights, website
funnel and the generated client sites. Read before changing spend.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_marketing(ctx: ToolContext) -> dict:
    c = ctx.client
    ads, meta, metrics, funnel, sites = await asyncio.gather(
        _try(c.get("/admin/ads"), "ads"),
        _try(c.get("/admin/meta/insights"), "Meta ad insights"),
        _try(c.get("/marketing/metrics"), "marketing metrics"),
        _try(c.get("/website/funnel"), "website funnel"),
        _try(c.get("/admin/sites"), "generated sites"),
    )
    return {"campaigns": ads, "meta_insights": meta, "metrics": metrics,
            "website_funnel": funnel, "generated_sites": sites}


@registry.tool(
    "check_integrations",
    group="observe",
    risk=Risk.READ,
    description="""Which integrations are actually wired up: Retell, Twilio, Resend,
Whop, Meta, Google Places, Render. Read this before blaming a number on performance —
in this app an unset key usually means the feature silently does nothing, and the
classic example is call reporting, which reports provisioning success and then
delivers no calls at all.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def check_integrations(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/integrations")


@registry.tool(
    "get_insights",
    group="observe",
    risk=Risk.READ,
    description="""The app's own generated insights and their metrics — what it has
already concluded about the business. Read before generating your own analysis so you
build on it rather than repeat it.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_insights(ctx: ToolContext) -> dict:
    c = ctx.client
    insights, metrics = await asyncio.gather(
        _try(c.get("/insights"), "insights"),
        _try(c.get("/insights/metrics"), "insight metrics"),
    )
    return {"insights": insights, "metrics": metrics}


@registry.tool(
    "get_system_health",
    group="observe",
    risk=Risk.READ,
    description="""The app's own health check. Use when something looks broken rather than
merely disappointing — a sudden drop in a number is far more often a failing integration
than a market change.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_system_health(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/system-health")


@registry.tool(
    "other_agents_activity",
    group="observe",
    risk=Risk.READ,
    description="""What OTHER automated agents are doing to this business through the
app's scoped agent-key API — most importantly the Automaton agent, which pushes leads,
creates sites and places calls on its own.
Atlas is not the only thing acting here. Read this before deciding the pipeline is
empty or that nobody is working a market: another agent may already be on it, and two
agents working the same list is worse than either working it alone.
Their limits are per-key and entirely separate from Atlas's, so nothing Atlas does
consumes their allowance or vice versa.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def other_agents_activity(ctx: ToolContext) -> dict:
    keys, usage = await asyncio.gather(
        _try(ctx.client.get("/admin/agent-keys"), "agent keys"),
        _try(ctx.client.get("/admin/agent-keys/usage"), "agent key usage"),
    )
    return {
        "keys": keys, "usage": usage,
        "note": ("These are external automations with narrow, scoped access — they can "
                 "push leads, build sites and dial leads they created, but cannot read "
                 "revenue totals, transcripts or recordings. Do not duplicate their work."),
    }
