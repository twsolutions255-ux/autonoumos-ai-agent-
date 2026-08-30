"""Making existing clients more valuable — the half of growth that is not selling.

Every client TWS keeps is worth more than one it has to win, and a client who
cannot see what they are getting leaves at renewal without ever complaining.
The app has real analysis tools for this and they are mostly unused, because
running them per client is exactly the kind of patient, repetitive work nobody
gets to.

All of these are client-scoped: the app requires a `client_id` on every one,
and a superadmin who omits it gets a 400 rather than a sensible default. Get
ids from get_clients.

Note the deliberate split: the analyses here CHANGE NOTHING. They read a
client's real site, their real competitors and their real numbers, and produce
a prioritised list. Turning that list into work is a separate decision.
"""
from __future__ import annotations

import logging
from typing import Any

from ..guardrails.policy import Risk
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.clientcare")


@registry.tool(
    "analyze_client_website",
    group="clients",
    risk=Risk.INTERNAL,
    description="""Fetch a client's REAL website and audit the concrete things that cost
them conversions, then turn the failures into a prioritised list.
Every check is verified against the actual HTML — nothing is assumed — which is what
makes the output usable in a conversation with the owner rather than generic advice.
Use it before a renewal, or when a client's numbers are soft and you need to know
whether the problem is the receptionist or everything upstream of it.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
    },
)
async def analyze_client_website(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.post("/website/analyze", {}, params={"client_id": client_id})


@registry.tool(
    "analyze_client_competitors",
    group="clients",
    risk=Risk.INTERNAL,
    description="""Scan a client's competitors and say where they are losing.
The app refuses this if no competitors have been added for that client yet — add them
first with add_client_competitor. That refusal is correct: an analysis with nothing to
compare against is invention.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
    },
)
async def analyze_client_competitors(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.post("/competitors/analyze", {}, params={"client_id": client_id})


@registry.tool(
    "get_client_competitors",
    group="clients",
    risk=Risk.READ,
    description="""The competitors on file for a client, and the most recent analysis.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
    },
)
async def get_client_competitors(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.get("/competitors", client_id=client_id)


@registry.tool(
    "add_client_competitor",
    group="clients",
    risk=Risk.INTERNAL,
    description="""Put a competitor on file for a client so the scan has something real to
compare against. Add the businesses that actually take their calls — the local ones
ranking above them — not national chains.""",
    schema={
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "name": {"type": "string"},
            "website": {"type": "string"},
        },
        "required": ["client_id", "name"],
    },
)
async def add_client_competitor(ctx: ToolContext, client_id: str, name: str,
                                website: str = "") -> Any:
    return await ctx.client.post("/competitors", {"name": name, "website": website or None},
                                 params={"client_id": client_id})


@registry.tool(
    "analyze_client_marketing",
    group="clients",
    risk=Risk.INTERNAL,
    description="""Assess a client's marketing position against their own live numbers and
produce recommendations. Useful ammunition for an upsell conversation, and for spotting
a client whose problem is demand rather than call handling.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
    },
)
async def analyze_client_marketing(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.post("/marketing/analyze", {}, params={"client_id": client_id})


@registry.tool(
    "seo_review",
    group="clients",
    risk=Risk.INTERNAL,
    description="""Read a client's live site and say what is costing them local search.
Analysis only — it changes nothing by design. This is a Scale-tier feature, so running it
for a Foundation client is also a concrete, specific reason for them to upgrade.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
    },
)
async def seo_review(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.post(f"/admin/clients/{client_id}/seo-review")


@registry.tool(
    "get_seo_review",
    group="clients",
    risk=Risk.READ,
    description="""The most recent SEO review for a client, without re-running it.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
    },
)
async def get_seo_review(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.get(f"/admin/clients/{client_id}/seo-review")


@registry.tool(
    "generate_client_site",
    group="clients",
    risk=Risk.STAGE,
    description="""Build a client a one-page conversion-focused site that drives calls into
the receptionist they are already paying for.
One page by design — no CMS, nothing per-client to maintain. It prefers their real
receptionist number, so the site feeds the product rather than pointing at a number that
will drift out of date.""",
    schema={
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "business_name": {"type": "string",
                              "description": "Only if it should differ from their account name."},
        },
        "required": ["client_id"],
    },
)
async def generate_client_site(ctx: ToolContext, client_id: str,
                               business_name: str = "") -> Any:
    body = {"client_id": client_id}
    if business_name:
        body["business_name"] = business_name
    return await ctx.client.post("/admin/sites/generate", body)


@registry.tool(
    "get_client_guarantee",
    group="clients",
    risk=Risk.READ,
    description="""Where a client stands against the guarantee they were sold. A client
about to fall short of it is a refund and a churn at the same time — and the month to
find that out is the one you are still in, not the one after.""",
    schema={
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "month": {"type": "string", "description": "YYYY-MM. Omit for this month."},
        },
        "required": ["client_id"],
    },
)
async def get_client_guarantee(ctx: ToolContext, client_id: str, month: str = "") -> Any:
    return await ctx.client.get(f"/admin/clients/{client_id}/guarantee", month=month or None)
