"""Growth: the acquisition pipeline Atlas actually runs.

TW Solutions already contains a complete agency pipeline — find local
businesses in a trade and city, audit their existing web presence, rank them,
build each a real site, draft outreach that cites the specific faults found,
follow up, and convert. What it has never had is anything to *drive* that
pipeline: a person has to choose the markets, press the buttons in order, and
remember who was touched when.

That is Atlas's job here. These tools are deliberately thin over the app's
endpoints; the intelligence is in choosing which market to scan next, which
prospects deserve a site built, and when to stop touching someone.

One honesty note that shapes this whole module: **the app drafts outreach, it
does not send it.** `build_prospect_pitch` produces a site and a message;
`record_outreach_sent` logs that a human sent it. The only genuinely outbound
actions in this file are the cold-call ones, and they are gated accordingly.
Atlas must never claim to have emailed anybody.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..guardrails.policy import Risk
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.growth")

TOUCH_CHANNELS = ("email", "sms", "call", "dm", "other")

#: The app refuses a fourth touch on an unanswered prospect. Mirrored here so
#: Atlas plans around it rather than discovering it as an error.
MAX_TOUCHES = 3


# --------------------------------------------------------------------------
# discovery — the top of the funnel
# --------------------------------------------------------------------------

@registry.tool(
    "scan_market",
    group="growth",
    risk=Risk.INTERNAL,
    description="""Find real businesses in one trade and one city, audit the web presence
of each, and rank them as prospects. This is the top of the acquisition funnel and the
single highest-leverage thing Atlas can do.
Pick the trade and city deliberately: results feed everything downstream, and scanning
a market the team cannot service wastes the whole chain. Prefer trades where a missed
phone call obviously costs money — roofing, HVAC, plumbing, towing, dental.
Needs a Google Places key in the app; falls back to Yelp, whose results carry no
website and therefore score poorly. Returns ranked prospects, saved to the CRM so the
same business is never worked twice.""",
    schema={
        "type": "object",
        "properties": {
            "trade": {"type": "string", "description": "e.g. 'roofing contractor', 'HVAC', 'plumber'"},
            "location": {"type": "string", "description": "e.g. 'Miami FL', 'Austin TX'"},
            "limit": {"type": "integer", "description": "How many to pull, 1-50. Default 20."},
        },
        "required": ["trade", "location"],
    },
)
async def scan_market(ctx: ToolContext, trade: str, location: str, limit: int = 20) -> Any:
    return await ctx.client.post("/admin/prospects/scan", {
        "trade": trade, "location": location, "limit": max(1, min(int(limit or 20), 50)),
    })


@registry.tool(
    "save_market_search",
    group="growth",
    risk=Risk.INTERNAL,
    description="""Save a trade+city so it re-scans by itself overnight, permanently
topping up the prospect list without anyone asking.
This is how Atlas turns a one-off scan into a standing source of leads, and it is also
a hard prerequisite for Nadia's automatic lead-pool refill: with no saved searches she
reports that there is nothing to re-run and does nothing at all.
Save the markets that convert; do not save every market you try.""",
    schema={
        "type": "object",
        "properties": {
            "trade": {"type": "string"},
            "location": {"type": "string"},
            "limit": {"type": "integer", "description": "Per overnight run, 1-50."},
        },
        "required": ["trade", "location"],
    },
)
async def save_market_search(ctx: ToolContext, trade: str, location: str,
                             limit: int = 20) -> Any:
    res = await ctx.client.post("/admin/prospects/saved-searches", {
        "trade": trade, "location": location, "limit": max(1, min(int(limit or 20), 50)),
    })
    return {"saved": res, "note": "This market now re-scans overnight and is eligible "
                                  "for the automatic lead-pool refill."}


@registry.tool(
    "list_market_searches",
    group="growth",
    risk=Risk.READ,
    description="""The saved trade+city searches that re-run overnight. Note that the
automatic pool refill only re-runs the five most recent, so a long list quietly starves
the older entries — prune rather than accumulate.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def list_market_searches(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/prospects/saved-searches")


@registry.tool(
    "drop_market_search",
    group="growth",
    risk=Risk.INTERNAL,
    description="""Stop re-scanning a market that is not converting. Use it to keep the
saved list short, because only the five newest are re-run automatically.""",
    schema={
        "type": "object",
        "properties": {"search_id": {"type": "string"}},
        "required": ["search_id"],
    },
)
async def drop_market_search(ctx: ToolContext, search_id: str) -> str:
    await ctx.client.delete(f"/admin/prospects/saved-searches/{search_id}")
    return f"Stopped re-scanning saved search {search_id}."


@registry.tool(
    "list_prospects",
    group="growth",
    risk=Risk.READ,
    description="""The prospect CRM: who has been found, their audit score, what has been
sent to them and when. Read this before any outreach decision — it is what stops the
same business being worked twice.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def list_prospects(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/prospects")


@registry.tool(
    "audit_prospect",
    group="growth",
    risk=Risk.INTERNAL,
    description="""Re-audit one prospect's current website and score it. Use when a
prospect looks promising but was scanned from a source that carried no website, so the
original score is unreliable.""",
    schema={
        "type": "object",
        "properties": {"prospect_id": {"type": "string"}},
        "required": ["prospect_id"],
    },
)
async def audit_prospect(ctx: ToolContext, prospect_id: str) -> Any:
    return await ctx.client.post(f"/admin/prospects/{prospect_id}/audit")


# --------------------------------------------------------------------------
# build and outreach
# --------------------------------------------------------------------------

@registry.tool(
    "build_prospect_pitch",
    group="growth",
    risk=Risk.STAGE,
    description="""For one prospect, build them a real working site AND draft the outreach
message in a single step. The message cites the actual faults found on their current
site and links the live replacement, which is the entire reason this converts.
This DRAFTS — it does not send. A human sends the message, then you log it with
record_outreach_sent so follow-ups are timed correctly.
Spend this on prospects that scored badly on their existing site and have a real phone
number; building for a business with a good site wastes the strongest asset you have.""",
    schema={
        "type": "object",
        "properties": {"prospect_id": {"type": "string"}},
        "required": ["prospect_id"],
    },
)
async def build_prospect_pitch(ctx: ToolContext, prospect_id: str) -> Any:
    res = await ctx.client.post(f"/admin/prospects/{prospect_id}/build")
    return {"built": res,
            "reminder": "Nothing has been sent. A person sends this, then call "
                        "record_outreach_sent so the follow-up clock starts."}


@registry.tool(
    "draft_followup",
    group="growth",
    risk=Risk.STAGE,
    description="""Draft the next, shorter message to a prospect who has not replied.
The app refuses a fourth message to someone who has never answered — three unanswered
messages is where persistence becomes pestering — and it refuses to draft at all if
nothing was ever sent. Both refusals are correct; do not work around them.""",
    schema={
        "type": "object",
        "properties": {"prospect_id": {"type": "string"}},
        "required": ["prospect_id"],
    },
)
async def draft_followup(ctx: ToolContext, prospect_id: str) -> Any:
    return await ctx.client.post(f"/admin/prospects/{prospect_id}/followup")


@registry.tool(
    "record_outreach_sent",
    group="growth",
    risk=Risk.INTERNAL,
    description="""Log that a message really went out to a prospect. Only call this when
it actually did — the follow-up sequence is timed off these records, and a follow-up
written against a first message that never left is worse than no follow-up at all.""",
    schema={
        "type": "object",
        "properties": {
            "prospect_id": {"type": "string"},
            "channel": {"type": "string", "enum": list(TOUCH_CHANNELS)},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["prospect_id", "channel"],
    },
)
async def record_outreach_sent(ctx: ToolContext, prospect_id: str, channel: str,
                               subject: str = "", body: str = "", note: str = "") -> str:
    if channel not in TOUCH_CHANNELS:
        return f"channel must be one of {', '.join(TOUCH_CHANNELS)}"
    await ctx.client.post(f"/admin/prospects/{prospect_id}/touch", {
        "channel": channel, "subject": subject, "body": body, "note": note})
    return f"Logged a {channel} touch for prospect {prospect_id}."


@registry.tool(
    "convert_prospect",
    group="growth",
    risk=Risk.STAGE,
    description="""Turn a prospect who said yes into a real client account. Use only when
they have actually agreed — this creates the account and starts the onboarding.""",
    schema={
        "type": "object",
        "properties": {"prospect_id": {"type": "string"}},
        "required": ["prospect_id"],
    },
)
async def convert_prospect(ctx: ToolContext, prospect_id: str) -> Any:
    return await ctx.client.post(f"/admin/prospects/{prospect_id}/convert")


@registry.tool(
    "site_outreach_copy",
    group="growth",
    risk=Risk.INTERNAL,
    description="""Regenerate the outreach copy for a site already built, citing the real
faults found on the business's current site. Use when the first message did not land and
you want a different angle on the same evidence.""",
    schema={
        "type": "object",
        "properties": {"slug": {"type": "string", "description": "The generated site's slug."}},
        "required": ["slug"],
    },
)
async def site_outreach_copy(ctx: ToolContext, slug: str) -> Any:
    return await ctx.client.post(f"/admin/sites/{slug}/outreach")


@registry.tool(
    "list_lead_opportunities",
    group="growth",
    risk=Risk.READ,
    description="""Businesses the app has noticed are worth pursuing but nobody has
promoted into the pipeline yet. Cheap, warm, and routinely ignored — check it every
cycle before scanning a cold market.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def list_lead_opportunities(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/lead-opportunities")


@registry.tool(
    "promote_lead_opportunity",
    group="growth",
    risk=Risk.STAGE,
    description="""Promote a noticed opportunity into the real pipeline so the team works
it. Prefer this over a fresh cold scan when the opportunity list is not empty.""",
    schema={
        "type": "object",
        "properties": {
            "opportunity_ids": {"type": "array", "items": {"type": "string"},
                                "description": "Which opportunities to promote."},
        },
        "required": ["opportunity_ids"],
    },
)
async def promote_lead_opportunity(ctx: ToolContext, opportunity_ids: list) -> Any:
    return await ctx.client.post("/admin/lead-opportunities/promote",
                                 {"ids": list(opportunity_ids or [])})


# --------------------------------------------------------------------------
# cold calling — the only genuinely outbound path here
# --------------------------------------------------------------------------

@registry.tool(
    "cold_call_preflight",
    group="growth",
    risk=Risk.READ,
    description="""Everything that must be true before any call is placed: is a calling
agent configured, is there a from-number, how many prospects are callable, what is the
daily cap and how much of it is spent. Always read this before staging a batch —
staging into a broken configuration produces a queue nobody drains.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def cold_call_preflight(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/cold-calling/preflight")


@registry.tool(
    "stage_cold_call_batch",
    group="growth",
    risk=Risk.STAGE,
    rate_bucket="calls",
    description="""Stage a set of prospects for calling. This DIALS NOTHING — it queues
them and reports what it refused and why (do-not-call, no phone, outside calling hours,
already worked). Read the refusals: a batch that silently drops half its list is how
somebody concludes the dialler is broken when it was protecting them.
Releasing the batch is a separate, deliberate step.""",
    schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "How many to stage. Default 30."},
            "prospect_ids": {"type": "array", "items": {"type": "string"},
                             "description": "Specific prospects. Omit to let the app choose the best."},
        },
        "required": [],
    },
)
async def stage_cold_call_batch(ctx: ToolContext, limit: int = 30,
                                prospect_ids: Optional[list] = None) -> Any:
    body: dict[str, Any] = {"limit": max(1, int(limit or 30))}
    if prospect_ids:
        body["prospect_ids"] = list(prospect_ids)
    res = await ctx.client.post("/admin/cold-calling/queue-batch", body)
    return {"staged": res.get("staged"), "prospects": res.get("prospects"),
            "refused": res.get("refused"),
            "reminder": "Nothing has been dialled. release_cold_call_batch starts calling."}


@registry.tool(
    "release_cold_call_batch",
    group="growth",
    risk=Risk.EXTERNAL_COMMS,
    rate_bucket="calls",
    description="""Release the staged batch to the dialler. THIS PLACES REAL CALLS TO REAL
STRANGERS' PHONES and cannot be undone once a call connects.
Only do this when you have read the preflight, read what the staging step refused, and
are satisfied the list is right. Anything outside a business's local calling hours waits
until 8am where they are.
If you are not certain, stage the batch and tell the owner it is ready instead.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def release_cold_call_batch(ctx: ToolContext) -> Any:
    res = await ctx.client.post("/admin/cold-calling/start-batch")
    return {"approved": res.get("approved"), "note": res.get("note")}


@registry.tool(
    "stop_cold_calling",
    group="growth",
    risk=Risk.INTERNAL,
    description="""Stop all cold calling immediately and un-approve everything still
staged. This is the emergency brake for the outbound dialler — use it the moment
anything looks wrong. Stopping is always safe; ask forgiveness, not permission.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def stop_cold_calling(ctx: ToolContext) -> Any:
    res = await ctx.client.post("/admin/cold-calling/stop")
    return {"stopped": True, "detail": res}


@registry.tool(
    "cold_call_autonomy",
    group="growth",
    risk=Risk.READ,
    description="""Read whether the app is allowed to place calls on its own, and its
daily ceiling. This one switch also gates the automatic lead-pool refill.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def cold_call_autonomy(ctx: ToolContext) -> Any:
    return await ctx.client.get("/admin/cold-calling/autonomy")


@registry.tool(
    "set_cold_call_autonomy",
    group="growth",
    risk=Risk.IRREVERSIBLE,
    always_approve=True,
    description="""Turn the app's own calling autonomy on or off and set its daily cap.
Turning it ON is the decision to let software phone strangers without a person in the
loop each time, so it always goes to the owner for approval no matter how Atlas is
configured. Turning it OFF is safe and immediate.""",
    schema={
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "daily_cap": {"type": "integer", "description": "1-100 calls per day."},
        },
        "required": ["enabled"],
    },
)
async def set_cold_call_autonomy(ctx: ToolContext, enabled: bool,
                                 daily_cap: int = 30) -> Any:
    return await ctx.client.put("/admin/cold-calling/autonomy", {
        "enabled": bool(enabled), "daily_cap": max(1, min(int(daily_cap or 30), 100))})


@registry.tool(
    "check_dnc",
    group="growth",
    risk=Risk.READ,
    description="""Is this phone or email on the do-not-call list? Check BEFORE spending
any budget on a contact. Contacting a listed business is a compliance failure, not a
performance problem.""",
    schema={
        "type": "object",
        "properties": {
            "phone": {"type": "string"},
            "email": {"type": "string"},
        },
        "required": [],
    },
)
async def check_dnc(ctx: ToolContext, phone: str = "", email: str = "") -> Any:
    if not phone and not email:
        return {"error": "Give a phone or an email to check."}
    return await ctx.client.get("/dnc/check", phone=phone or None, email=email or None)


@registry.tool(
    "discover_cold_call_leads",
    group="growth",
    risk=Risk.INTERNAL,
    description="""Find new businesses to call, using the app's own discovery. Separate
from scan_market: this fills the calling list specifically rather than the site-building
prospect list.""",
    schema={
        "type": "object",
        "properties": {
            "trade": {"type": "string"},
            "location": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["trade", "location"],
    },
)
async def discover_cold_call_leads(ctx: ToolContext, trade: str, location: str,
                                   limit: int = 20) -> Any:
    return await ctx.client.post("/admin/cold-calling/discover-leads", {
        "trade": trade, "location": location, "limit": max(1, min(int(limit or 20), 50))})


# --------------------------------------------------------------------------
# the scheduled jobs
# --------------------------------------------------------------------------

#: The app's timer-driven work. Their docstrings say "runs hourly", but nothing
#: inside the app calls them — they are driven from outside. Atlas can run them
#: on demand, which matters most right after it has created work for one:
#: pushing a lead and then draining speed-to-lead turns "called tomorrow" into
#: "called now", and speed to lead is the strongest conversion lever there is.
JOBS = {
    "speed_to_lead": ("/internal/speed-to-lead/drain",
                      "Call leads that came in outside hours, now that they are open."),
    "audits": ("/internal/audits/drain", "Place the queued missed-call audits."),
    "workflows": ("/internal/workflows/run", "Fire the scheduled follow-ups that are due."),
    "nightly_digest": ("/internal/nightly-digest", "Send the Growth-tier client brief."),
    "prospect_rescan": ("/internal/prospects/rescan", "Re-run the saved market searches."),
    "ai_employees": ("/internal/ai-employees/run",
                     "Run the AI employees WITH the power to act — this is the only path "
                     "by which they may start a call round."),
}


@registry.tool(
    "run_scheduled_job",
    group="growth",
    risk=Risk.EXTERNAL_COMMS,
    description="""Run one of the app's scheduled jobs right now instead of waiting for
its timer. Most useful immediately after creating work for one — draining speed-to-lead
after a lead arrives turns 'called tomorrow' into 'called within minutes', which is the
single strongest conversion lever in this business.
These reach the outside world: they place calls, send emails and text people. The
'ai_employees' job is the strongest of all, because it is the only route by which the
employees may start a cold-call round on their own.
Needs the app's cron secret; says so plainly if it is not configured.""",
    schema={
        "type": "object",
        "properties": {
            "job": {"type": "string", "enum": sorted(JOBS),
                    "description": "Which job to run."},
        },
        "required": ["job"],
    },
)
async def run_scheduled_job(ctx: ToolContext, job: str) -> Any:
    entry = JOBS.get(job)
    if not entry:
        return f"No such job. Available: {', '.join(sorted(JOBS))}."
    path, what = entry
    res = await ctx.client.run_internal_job(path)
    return {"job": job, "did": what, "result": res}


# --------------------------------------------------------------------------
# the setter pool
# --------------------------------------------------------------------------
#
# THREE separate things in this app are called "leads" in conversation, and
# confusing them is the fastest way to make a mess:
#
#   prospects            the Prospect Engine — superadmin-only, gets a free
#                        site built and grounded outreach drafted. scan_market
#                        and build_prospect_pitch above work on these.
#   the shared pool      businesses the human setters claim and work by phone
#                        and in person. The tools below work on these.
#   cold-call prospects  the dialler's own queue, with its own state machine
#                        and TCPA guards. stage/release above work on these.
#
# They are different collections with different rules. A business in one is not
# in the others.

@registry.tool(
    "search_pool_candidates",
    group="growth",
    risk=Risk.READ,
    description="""Search a trade and town and rank what comes back against the company's
own ideal-customer profile, WITHOUT saving anything.
This feeds the shared pool the human setters work — a different list from the Prospect
Engine, which is the one that gets a free site built. Use this when the constraint is
that the reps have nothing good to call, rather than that outreach is not converting.
Nothing is saved, so a bad search costs nothing. Review the ranking, then add only the
ones worth a rep's time with add_pool_leads.""",
    schema={
        "type": "object",
        "properties": {
            "trade": {"type": "string"},
            "location": {"type": "string"},
        },
        "required": ["trade", "location"],
    },
)
async def search_pool_candidates(ctx: ToolContext, trade: str, location: str) -> Any:
    return await ctx.client.post("/leads/discover", {"trade": trade, "location": location})


@registry.tool(
    "add_pool_leads",
    group="growth",
    risk=Risk.STAGE,
    description="""Add chosen businesses to the shared pool for the setters to claim.
Add the ones that scored well and have a phone number — a candidate without one is
skipped by the app anyway, because the pool enforces uniqueness on the number.
Be selective. Filling the pool with everything a search returned is how a team stops
trusting the pool and goes back to finding their own.""",
    schema={
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "description": "Rows from search_pool_candidates. Up to 40 per call.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "phone": {"type": "string"},
                        "address": {"type": "string"},
                        "trade": {"type": "string"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["candidates"],
    },
)
async def add_pool_leads(ctx: ToolContext, candidates: list) -> Any:
    res = await ctx.client.post("/leads/discover/add",
                                {"candidates": list(candidates or [])[:40]})
    return {"result": res,
            "note": "These are now claimable by any setter. Tell the team in #setters "
                    "that fresh leads are in, or they will not know."}


@registry.tool(
    "get_lead_pool",
    group="growth",
    risk=Risk.READ,
    description="""The unclaimed shared pool — what the setters have left to work.
An empty pool means the team is idle; a pool that never shrinks means they are not
working it, and those need opposite responses. Check which it is before adding more.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_lead_pool(ctx: ToolContext) -> dict:
    data = await ctx.client.get("/leads/pool")
    rows = data.get("leads", []) if isinstance(data, dict) else (data or [])
    return {"unclaimed": data.get("count", len(rows)) if isinstance(data, dict) else len(rows),
            "sample": rows[:20]}


@registry.tool(
    "get_lead_stats",
    group="growth",
    risk=Risk.READ,
    description="""What the setters actually did over a window, per rep: calls made,
contacts reached, appointments booked. This is the activity-versus-outcome picture —
use it to tell 'they are not working' from 'the leads are bad', which look identical in
a headline number and need opposite fixes.""",
    schema={
        "type": "object",
        "properties": {"days": {"type": "integer", "description": "Window, 1-90. Default 7."}},
        "required": [],
    },
)
async def get_lead_stats(ctx: ToolContext, days: int = 7) -> Any:
    return await ctx.client.get("/admin/leads/stats", days=max(1, min(int(days or 7), 90)))
