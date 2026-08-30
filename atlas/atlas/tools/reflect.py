"""Atlas's tools for managing itself: what it believes, what it intends, and
what it tells the owner.

These are the tools that make the difference between an agent and a script. A
script does the same thing every cycle. An agent writes down what it decided
and why, checks next cycle whether it worked, and changes its mind in public.

Nothing here touches the business, so it is all low-risk — but it is the part
that compounds. An Atlas that cannot record "cold email to dentists converted
at a third the rate of roofers" is an Atlas that will scan dentists again next
month.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from ..db import iso
from ..guardrails.policy import Risk
from ..memory.store import Kind
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.reflect")

OBJECTIVE_STATUSES = ("active", "achieved", "abandoned", "blocked")


@registry.tool(
    "remember",
    group="self",
    risk=Risk.INTERNAL,
    description="""Write something down for your future self, permanently.
Record a LESSON when evidence changes what you believe about what works. Record a FACT
about the business or market. Record a CONSTRAINT you must always respect.
Be specific and quantitative — 'roofing in Miami converted 3 of 40 touches, dentists 0
of 35' is worth keeping; 'outreach went okay' is not. Say what the evidence was, so a
later cycle can judge whether it still holds.
When new evidence contradicts something you recorded before, pass `supersedes` with the
old memory's id rather than writing a second contradictory note.""",
    schema={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": [Kind.LESSON, Kind.FACT, Kind.CONSTRAINT],
                     "description": "lesson = learned from evidence; fact = durable truth; "
                                    "constraint = a rule to always respect."},
            "title": {"type": "string", "description": "One line, specific."},
            "body": {"type": "string", "description": "The detail, including the evidence."},
            "importance": {"type": "integer", "description": "1-5. Reserve 5 for things that "
                                                             "should change behaviour every cycle."},
            "confidence": {"type": "number", "description": "0-1. One data point is not a law."},
            "tags": {"type": "array", "items": {"type": "string"}},
            "supersedes": {"type": "string", "description": "Id of a memory this replaces."},
        },
        "required": ["kind", "title", "body"],
    },
)
async def remember(ctx: ToolContext, kind: str, title: str, body: str,
                   importance: int = 3, confidence: float = 0.6,
                   tags: Optional[list] = None, supersedes: str = "") -> str:
    m = await ctx.memory.remember(
        kind=kind, title=title, body=body, tags=tags or [],
        importance=importance, confidence=confidence,
        source="atlas", supersedes=supersedes or None)
    return (f"Remembered ({kind}, importance {m.importance}): {m.title} — id {m.id}"
            + (f", superseding {supersedes}" if supersedes else ""))


@registry.tool(
    "recall",
    group="self",
    risk=Risk.READ,
    description="""Search your own memory for what you have already learned about
something. Do this before repeating an experiment — the fastest way to waste a cycle is
to rediscover a lesson you already paid for.""",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What you want to remember about."},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
)
async def recall(ctx: ToolContext, query: str, limit: int = 15) -> dict:
    mems = await ctx.memory.recall(query, limit=max(1, min(int(limit or 15), 50)))
    return {"memories": [
        {"id": m.id, "kind": m.kind, "title": m.title, "body": m.body,
         "importance": m.importance, "confidence": m.confidence, "at": m.created_at}
        for m in mems]}


@registry.tool(
    "get_plan",
    group="self",
    risk=Risk.READ,
    description="""Your current strategy and its objectives — what you decided you are
trying to achieve and how you will know. Read it at the start of every cycle so you are
continuing a plan rather than inventing a new one each time.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def get_plan(ctx: ToolContext) -> dict:
    doc = await ctx.store["plan"].find_one({}, {"_id": 0}, sort=[("version", -1)])
    if not doc:
        return {"plan": None,
                "note": "No plan yet. Set one with set_plan before acting — an agent "
                        "without a stated objective cannot tell progress from activity."}
    return {"plan": doc}


@registry.tool(
    "set_plan",
    group="self",
    risk=Risk.INTERNAL,
    description="""Write down the strategy you are running: the one thing that matters
most right now, the objectives under it, and how each is measured.
Keep it short and falsifiable. Every objective needs a number and a date, because the
point of writing it down is that a later cycle can tell whether it happened.
Do not rewrite this every cycle — change it when evidence says the strategy is wrong,
and say in `rationale` what changed your mind. Each save is versioned, so the history of
what you believed is preserved.""",
    schema={
        "type": "object",
        "properties": {
            "north_star": {"type": "string",
                           "description": "The single sentence that decides trade-offs."},
            "rationale": {"type": "string",
                          "description": "Why this, now — and what changed if you are revising."},
            "objectives": {
                "type": "array",
                "description": "Three to five. More than five is not a plan.",
                "items": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "measure": {"type": "string", "description": "The number that proves it."},
                        "target": {"type": "string", "description": "The value and the date."},
                        "status": {"type": "string", "enum": list(OBJECTIVE_STATUSES)},
                    },
                    "required": ["objective", "measure", "target"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["north_star", "objectives"],
    },
)
async def set_plan(ctx: ToolContext, north_star: str, objectives: list,
                   rationale: str = "") -> str:
    prev = await ctx.store["plan"].find_one({}, {"_id": 0}, sort=[("version", -1)])
    version = int((prev or {}).get("version", 0)) + 1
    doc = {
        "id": str(uuid.uuid4()), "version": version, "north_star": north_star,
        "rationale": rationale,
        "objectives": [
            {"objective": o.get("objective"), "measure": o.get("measure"),
             "target": o.get("target"), "status": o.get("status") or "active"}
            for o in (objectives or [])
        ],
        "created_at": iso(), "cycle_id": ctx.cycle_id,
    }
    await ctx.store["plan"].insert_one(dict(doc))
    return (f"Plan v{version} saved: {north_star} "
            f"({len(doc['objectives'])} objectives). Earlier versions are kept.")


@registry.tool(
    "brief_owner",
    group="self",
    risk=Risk.INTERNAL_COMMS,
    description="""Write a briefing to the owner — the morning plan, the evening summary,
or something that cannot wait.
Lead with what changed and what you did about it. Give real numbers. Say plainly what
you tried that did not work; a briefing that is always good news stops being read.
If you are blocked on a decision only they can make, say exactly what you need and what
you will do by default if they say nothing.""",
    schema={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["morning", "evening", "alert", "ad_hoc"]},
            "headline": {"type": "string", "description": "One line they could read alone."},
            "body": {"type": "string", "description": "The briefing. Markdown is fine."},
            "needs_decision": {"type": "string",
                               "description": "A decision only they can make. Omit if none."},
        },
        "required": ["kind", "headline", "body"],
    },
)
async def brief_owner(ctx: ToolContext, kind: str, headline: str, body: str,
                      needs_decision: str = "") -> str:
    doc = {
        "id": str(uuid.uuid4()), "kind": kind, "headline": headline, "body": body,
        "needs_decision": needs_decision or None, "created_at": iso(),
        "cycle_id": ctx.cycle_id, "read_at": None,
    }
    await ctx.store["briefs"].insert_one(dict(doc))
    return (f"Briefing saved for the owner ({kind}): {headline}"
            + (" — flagged as needing their decision." if needs_decision else ""))


@registry.tool(
    "record_result",
    group="self",
    risk=Risk.INTERNAL,
    description="""Close the loop on something you tried: what happened, and whether it
worked. This is the feedback half of the agent — without it every cycle starts from
zero and Atlas never gets better at its job.
Be honest about failures. A recorded failure is worth more than an unrecorded success,
because it stops the next cycle repeating it.""",
    schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "What you did."},
            "result": {"type": "string", "description": "What happened."},
            "worked": {"type": "boolean"},
            "evidence": {"type": "string", "description": "The numbers behind the judgement."},
        },
        "required": ["action", "result", "worked"],
    },
)
async def record_result(ctx: ToolContext, action: str, result: str, worked: bool,
                        evidence: str = "") -> str:
    m = await ctx.memory.record_outcome(action, result, bool(worked), evidence=evidence)
    return f"Recorded: {m.title} (id {m.id})"


@registry.tool(
    "snapshot_metrics",
    group="self",
    risk=Risk.INTERNAL,
    description="""Save the numbers that matter right now, so later cycles can see the
trend rather than only today's value. Call it once per cycle after reading the business
snapshot. A single number means almost nothing; the direction means almost everything.""",
    schema={
        "type": "object",
        "properties": {
            "metrics": {
                "type": "object",
                "description": "Flat name → number map, e.g. {\"clients\": 12, \"demos_booked_7d\": 9}.",
                "additionalProperties": {"type": "number"},
            },
            "note": {"type": "string"},
        },
        "required": ["metrics"],
    },
)
async def snapshot_metrics(ctx: ToolContext, metrics: dict, note: str = "") -> str:
    clean = {}
    for k, v in (metrics or {}).items():
        try:
            clean[str(k)[:60]] = float(v)
        except (TypeError, ValueError):
            continue
    if not clean:
        return "No numeric metrics given — nothing saved."
    await ctx.store["metrics"].insert_one({
        "id": str(uuid.uuid4()), "at": iso(), "cycle_id": ctx.cycle_id,
        "metrics": clean, "note": note})
    return f"Saved {len(clean)} metrics: {', '.join(sorted(clean))}"


@registry.tool(
    "metric_history",
    group="self",
    risk=Risk.READ,
    description="""How the numbers have moved over recent cycles. Use it to tell a real
trend from a single good day, and to check whether what you changed last week actually
moved anything.""",
    schema={
        "type": "object",
        "properties": {"limit": {"type": "integer", "description": "How many snapshots back."}},
        "required": [],
    },
)
async def metric_history(ctx: ToolContext, limit: int = 14) -> dict:
    rows = await ctx.store["metrics"].find({}, {"_id": 0}) \
        .sort("at", -1).to_list(max(1, min(int(limit or 14), 90)))
    rows.reverse()
    series: dict[str, list] = {}
    for r in rows:
        for k, v in (r.get("metrics") or {}).items():
            series.setdefault(k, []).append({"at": r["at"], "value": v})
    trends = {}
    for k, pts in series.items():
        if len(pts) >= 2:
            first, last = pts[0]["value"], pts[-1]["value"]
            delta = last - first
            trends[k] = {"first": first, "latest": last, "change": round(delta, 2),
                         "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat")}
    return {"snapshots": len(rows), "series": series, "trends": trends}


@registry.tool(
    "list_approvals",
    group="self",
    risk=Risk.READ,
    description="""Actions of yours waiting on the owner's decision. Check before
re-proposing something — if it is already queued, chase it in a briefing rather than
queueing it twice.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def list_approvals(ctx: ToolContext) -> dict:
    rows = await ctx.store["approvals"].find(
        {"status": "pending"}, {"_id": 0}).sort("created_at", -1).to_list(50)
    return {"pending": rows, "count": len(rows)}
