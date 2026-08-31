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
    "check_receptionist_end_to_end",
    group="clients",
    risk=Risk.READ,
    description="""Walk the whole path a customer takes to reach a client's AI
receptionist, and report each of the nine steps separately: the Retell key proved by
using it, the agent, the phone number's inbound binding, the webhook URL, the live
prompt, the recording disclosure Retell actually speaks, rejected call reports, Telnyx
for the follow-up text, and whether a real call has ever arrived with a transcript.

Every step asks Retell or Telnyx directly rather than reading the app's own settings
back, so a setting saved in the CRM that never reached the provider shows as a failure
rather than a tick.

READ THE VERDICT FIELD CAREFULLY, IT HAS THREE STATES AND THE MIDDLE ONE MATTERS MOST:

  pass          a real call came through and everything worked
  not_verified  every setting is right AND NOTHING HAS BEEN PROVED, because nobody
                has rung the number. This is NOT a pass. Do not report it as one,
                do not average it into a health score, and do not describe the
                receptionist as working. The only thing that clears it is a human
                dialling the number, and saying so plainly is the useful output.
  failing       something is actually broken; each step carries the fix

Run this for every client whose receptionist is meant to be live. It is the one check
that distinguishes 'configured' from 'working', and those look identical everywhere
else in the app.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
        "additionalProperties": False,
    },
)
async def check_receptionist_end_to_end(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.get("/admin/receptionist/end-to-end", client_id=client_id)


@registry.tool(
    "analyze_client_website",
    group="clients",
    risk=Risk.APP_WRITE,
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
    risk=Risk.APP_WRITE,
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
    risk=Risk.APP_WRITE,
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
    risk=Risk.APP_WRITE,
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
    risk=Risk.APP_WRITE,
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


# --------------------------------------------------------------------------
# talking to a paying client
# --------------------------------------------------------------------------
#
# Everything in comms.py is internal — the team, the AI employees, the owner.
# This is the one place Atlas addresses a customer, and it is deliberately
# separate: a message to a paying client is the company speaking, not a note
# between colleagues.

@registry.tool(
    "message_client",
    group="clients",
    risk=Risk.EXTERNAL_COMMS,
    rate_bucket="outreach",
    description="""Send a message to a paying client, in the thread they already use to
talk to the agency. They see it as coming from TW Solutions, so write like the company,
not like a bot.
This is the highest-value use of your attention on the retention side: a client whose
receptionist has gone quiet, or who has never graded a single call, will not complain —
they will decide the product does not work and leave at renewal, and by then it has been
weeks. Reaching out with their real numbers, before they ask, is what stops that.
Lead with something specific and true about their account. Never send a check-in that
could have been sent to anybody.""",
    schema={
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "message": {"type": "string", "description": "Grounded in their real numbers."},
        },
        "required": ["client_id", "message"],
    },
)
async def message_client(ctx: ToolContext, client_id: str, message: str) -> str:
    await ctx.client.post("/messages", {"text": message, "client_id": client_id})
    return f"Sent to client {client_id} in their agency thread."


@registry.tool(
    "read_client_messages",
    group="clients",
    risk=Risk.READ,
    description="""The message thread with one client. Read it before writing — a client
who has already asked something and not been answered needs an answer, not a check-in.""",
    schema={
        "type": "object",
        "properties": {"client_id": {"type": "string"}},
        "required": ["client_id"],
    },
)
async def read_client_messages(ctx: ToolContext, client_id: str) -> Any:
    return await ctx.client.get("/messages", client_id=client_id)


@registry.tool(
    "client_message_threads",
    group="clients",
    risk=Risk.READ,
    description="""Every client thread with its unread count — who is waiting on a reply.
An unanswered client is the cheapest churn there is to prevent. Check this every cycle.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def client_message_threads(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/messages/threads")
