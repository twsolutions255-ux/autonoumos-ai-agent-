"""The money: what the company earns, what it owes, and the fastest paths to more.

TWS has two separate money systems and it matters that Atlas never mixes them:

  CLIENT-SIDE   what a client's receptionist recovered for them. Driven by
                `deal_status` on each call — a human grades it won or lost, and
                every client-facing dollar figure is a sum over that one field.
  AGENCY-SIDE   what TWS itself earns. There is no invoice or charge object
                anywhere; deal value is derived from price constants applied to
                a booking's tier. Whop is the billing system of record but never
                tells the app how much anyone paid.

Two consequences shape every tool here. First, **nothing in this codebase moves
money** — there is no transfer, no charge, no payout rail. The most Atlas can do
is provision an account, which spends real money on a phone number and an agent
clone. Second, several figures are computed from current env constants rather
than stored, so historical numbers move when pricing changes. Atlas must
therefore report figures as the app computes them and never re-derive its own.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from ..guardrails.policy import Risk
from .observe import _try
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.money")

#: Roughly what a new client is worth per month, used only to decide whether an
#: action crosses the approval threshold. Deliberately conservative — the real
#: figure comes from the app.
ASSUMED_MONTHLY_VALUE_USD = 1000.0


@registry.tool(
    "get_books",
    group="money",
    risk=Risk.READ,
    description="""The agency's own P&L for a month: revenue in, AI spend out, people
owed, and the difference. This is your objective function — read it every cycle and
judge yourself against it.
Note that revenue here is derived from each deal's tier and the current price
constants, not from anything Whop reported, so it is what the app believes rather
than what a bank statement would show.""",
    schema={
        "type": "object",
        "properties": {"month": {"type": "string", "description": "YYYY-MM. Omit for this month."}},
        "required": [],
    },
)
async def get_books(ctx: ToolContext, month: str = "") -> Any:
    return await ctx.client.get("/admin/books", month=month or None)


@registry.tool(
    "deals_awaiting_setup",
    group="money",
    risk=Risk.READ,
    description="""Deals that are WON but whose account was never created — money the
company has already earned and is not yet collecting.
This is routinely the highest-yield thing in the whole app: no selling required, the
customer has already said yes. Each row carries a blocked_reason explaining why it has
not gone live. Check this before scanning a single new market.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def deals_awaiting_setup(ctx: ToolContext) -> Any:
    res = await ctx.client.get("/admin/deals/awaiting-setup")
    return {"deals": res,
            "why_this_matters": "Every one of these is signed revenue that is not live. "
                                "Unblock them before spending anything on new prospects."}


@registry.tool(
    "provision_won_deal",
    group="money",
    risk=Risk.MONEY,
    estimate_cost=lambda a: ASSUMED_MONTHLY_VALUE_USD,
    description="""Turn a won deal into a live client account: create the login, clone
the receptionist agent and assign a phone number.
This SPENDS REAL MONEY — a phone number and an agent clone are billed — and it creates a
customer-facing account, so it is money-class and goes to the owner for approval.
Only do this for a deal that is genuinely won and not blocked. Read the blocked_reason
first: provisioning into a missing configuration produces an account with a working
login and a receptionist that never reports a single call.""",
    schema={
        "type": "object",
        "properties": {"booking_id": {"type": "string", "description": "From deals_awaiting_setup."}},
        "required": ["booking_id"],
    },
)
async def provision_won_deal(ctx: ToolContext, booking_id: str) -> Any:
    return await ctx.client.post(f"/admin/bookings/{booking_id}/provision-client")


@registry.tool(
    "get_ai_spend",
    group="money",
    risk=Risk.READ,
    description="""What the company's AI is costing, against its budget, broken down by
task, tier and person. Your own thinking is part of this cost. Growth that costs more
than it earns is not growth — check this before proposing anything that raises usage.""",
    schema={
        "type": "object",
        "properties": {"days": {"type": "integer", "description": "Window. Default 30."}},
        "required": [],
    },
)
async def get_ai_spend(ctx: ToolContext, days: int = 30) -> Any:
    return await ctx.client.get("/admin/ai-usage", days=max(1, min(int(days or 30), 365)))


@registry.tool(
    "get_payouts",
    group="money",
    risk=Risk.READ,
    description="""What the team is owed: the payout sheet and past runs.
Two things to know before reporting these numbers. The per-conversion rate for setters
defaults to zero, so every setter line can legitimately read $0 — read the notes and
blockers rather than only the total. And payout eligibility uses a person's PRIMARY role
only, so someone whose primary role is manager but who also closes will be missing.""",
    schema={
        "type": "object",
        "properties": {"month": {"type": "string", "description": "YYYY-MM."}},
        "required": [],
    },
)
async def get_payouts(ctx: ToolContext, month: str = "") -> dict:
    sheet, runs = await asyncio.gather(
        _try(ctx.client.get("/admin/payouts", month=month or None), "payout sheet"),
        _try(ctx.client.get("/admin/payouts/runs"), "payout runs"),
    )
    return {"sheet": sheet, "runs": runs}


@registry.tool(
    "start_payout_run",
    group="money",
    risk=Risk.MONEY,
    always_approve=True,
    description="""Open the monthly payout run. It does not move any money — nothing in
this app can — but it decides what people are told they are owed, so it always goes to
the owner.
The run comes back with questions[] and blockers[]: deals it could not attribute to a
person. Treat it as a verification job, resolve the questions from the evidence, and
report what is still unresolved.""",
    schema={
        "type": "object",
        "properties": {"month": {"type": "string", "description": "YYYY-MM."}},
        "required": ["month"],
    },
)
async def start_payout_run(ctx: ToolContext, month: str) -> Any:
    return await ctx.client.post("/admin/payouts/run", {"month": month})


@registry.tool(
    "client_value_report",
    group="money",
    risk=Risk.READ,
    description="""What one client has actually got for their money: calls handled,
appointments booked, revenue recovered and their guarantee position.
This is the evidence for a renewal, an upsell, or for catching a client who is quietly
getting nothing and will leave without ever complaining.""",
    schema={
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "month": {"type": "string", "description": "YYYY-MM. Omit for this month."},
        },
        "required": ["client_id"],
    },
)
async def client_value_report(ctx: ToolContext, client_id: str, month: str = "") -> dict:
    report, guarantee, recovery = await asyncio.gather(
        _try(ctx.client.get(f"/admin/clients/{client_id}/report", month=month or None),
             "client report"),
        _try(ctx.client.get(f"/admin/clients/{client_id}/guarantee", month=month or None),
             "guarantee"),
        _try(ctx.client.get("/dashboard/recovery", client_id=client_id), "recovery"),
    )
    return {"report": report, "guarantee": guarantee, "recovery": recovery}


@registry.tool(
    "find_ungraded_calls",
    group="money",
    risk=Risk.READ,
    description="""Calls carrying a revenue estimate that nobody has yet marked won or
lost. Every dollar figure in the app ignores these, so a large backlog means the product
looks less valuable than it is — which is exactly what loses a renewal.
Atlas cannot grade them: only the business owner knows whether the job was won. Surface
them, and get a human to decide.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string", "description": "Required — calls are per-client."}},
        "required": ["client_id"],
    },
)
async def find_ungraded_calls(ctx: ToolContext, client_id: str) -> dict:
    calls = await ctx.client.get("/calls", client_id=client_id)
    rows = calls.get("calls", calls) if isinstance(calls, dict) else calls
    pending = [c for c in (rows or [])
               if (c.get("deal_status") or "pending") == "pending"
               and float(c.get("revenue_estimate") or 0) > 0]
    pending.sort(key=lambda c: float(c.get("revenue_estimate") or 0), reverse=True)
    total = sum(float(c.get("revenue_estimate") or 0) for c in pending)
    return {
        "client_id": client_id,
        "ungraded": len(pending),
        "unclaimed_value_usd": round(total, 2),
        "top": [{"id": c.get("id"), "at": c.get("created_at"),
                 "estimate": c.get("revenue_estimate"),
                 "summary": (c.get("summary") or "")[:200]} for c in pending[:15]],
        "note": "Ask the client to grade these. Until they do, none of it counts toward "
                "the value they think they are getting.",
    }


@registry.tool(
    "upgrade_client_features",
    group="money",
    risk=Risk.MONEY,
    estimate_cost=lambda a: 800.0,
    description="""Turn on a paid add-on for a client — the AI Advisor, or the Agent
Suite. These are what separate the Foundation tier from Growth, and Growth is worth
roughly $800/month more per client.
Money-class deliberately: switching a paid feature on without the contract to match
gives it away, and switching it off removes something a paying client is entitled to.
Propose the upsell, let the owner confirm the commercials.""",
    schema={
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "ai_manager_enabled": {"type": "boolean", "description": "The AI Advisor add-on."},
            "agent_suite_enabled": {"type": "boolean", "description": "The Agent Suite add-on."},
        },
        "required": ["client_id"],
    },
)
async def upgrade_client_features(ctx: ToolContext, client_id: str,
                                  ai_manager_enabled: Optional[bool] = None,
                                  agent_suite_enabled: Optional[bool] = None) -> Any:
    body: dict[str, Any] = {}
    if ai_manager_enabled is not None:
        body["ai_manager_enabled"] = bool(ai_manager_enabled)
    if agent_suite_enabled is not None:
        body["agent_suite_enabled"] = bool(agent_suite_enabled)
    if not body:
        return "Nothing to change — name at least one feature."
    return await ctx.client.patch(f"/admin/clients/{client_id}", body)


@registry.tool(
    "get_subscriptions",
    group="money",
    risk=Risk.READ,
    description="""Billing status per client as Whop reports it: active, trial, past due,
or not connected. Past-due and trial-expiring clients are churn you can still prevent.
One caveat worth respecting: the app returns 'no_subscription' both when Whop genuinely
has nothing AND when the Whop API is down, so never treat it as proof someone stopped
paying without checking another signal.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
    },
)
async def get_subscriptions(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.get("/subscription", client_id=client_id)


@registry.tool(
    "get_tiers",
    group="money",
    risk=Risk.READ,
    description="""The pricing tiers and what each includes. Read before proposing an
upsell or quoting a price, so what you say matches what the app will actually bill.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_tiers(ctx: ToolContext) -> Any:
    return await ctx.client.get("/tiers")
