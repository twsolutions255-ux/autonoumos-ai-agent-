"""Talking to everyone who works here — human, AI, or the Operator.

The single most important thing Atlas does is not "run campaigns"; it is being
the one place where the AI employees, the human team and the owner are
coordinated by something that remembers what it asked yesterday.

Three populations, three genuinely different mechanics, and getting them
confused is how an agent ends up shouting into a void:

  HUMANS        team chat channels and DMs. Reliable, delivered, notified.
  AI EMPLOYEES  @mention in a channel. Reply is FIRE-AND-FORGET, always lands
                in #general regardless of where you asked, and carries no
                correlation id. It must be polled for and matched on
                (sender_id, created_at > ours). A DM to one is a silent dead
                end — the app stores it and never answers.
  AI OPERATOR   a task/document queue, not a conversation.

Everything here was written against the app's real behaviour rather than its
API shape, because on this surface the two differ a lot.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional

from ..guardrails.policy import Risk
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.comms")

#: The app rejects any other channel with a 404 (CHAT_CHANNELS is a fixed tuple).
CHANNELS = ("general", "setters", "wins", "questions")

#: The four hardcoded AI colleagues. Atlas discovers the live roster at runtime
#: via /admin/ai-employees; this is the fallback and the documentation.
KNOWN_EMPLOYEES = {
    "viktor": "Pipeline watch — demos and bookings.",
    "nadia": "Lead flow — the shared pool and who is working what.",
    "iris": "Client watch — paying clients whose receptionist has gone quiet.",
    "sol": "Coaching — reads outcomes and tells a rep the one thing to change.",
}

#: An employee reply is only generated when the mention survives the app's
#: anchored regex. Mirrored here so Atlas can check before spending a post.
MENTION_RE = re.compile(r"(?<![\w.@+-])@(" + "|".join(KNOWN_EMPLOYEES) + r")\b", re.I)


# --------------------------------------------------------------------------
# humans
# --------------------------------------------------------------------------

@registry.tool(
    "post_to_channel",
    group="comms",
    risk=Risk.INTERNAL_COMMS,
    rate_bucket="chat",
    description="""Post a message to a team chat channel, where the whole team sees it.
Use for decisions, direction, and context the team needs. Channels: general, setters,
wins, questions. This is how you lead the team day to day — prefer it over silence.
Do NOT use it to ask an AI employee something; use ask_ai_employee for that.""",
    schema={
        "type": "object",
        "properties": {
            "channel": {"type": "string", "enum": list(CHANNELS),
                        "description": "Which room. 'general' unless you have a reason."},
            "message": {"type": "string", "description": "What to say. Write like a manager, not a bot."},
        },
        "required": ["channel", "message"],
    },
)
async def post_to_channel(ctx: ToolContext, channel: str, message: str) -> str:
    if channel not in CHANNELS:
        return (f"'{channel}' is not a channel in this app — it would 404. "
                f"Valid channels: {', '.join(CHANNELS)}.")
    if MENTION_RE.search(message or ""):
        # Posting an @mention here would trigger an employee reply Atlas is not
        # waiting for, and burn one of its 12 hourly reply slots invisibly.
        return ("That message @mentions an AI employee. Posting it this way starts a "
                "reply you will never collect. Use ask_ai_employee instead.")
    res = await ctx.client.post(f"/chat/channels/{channel}/messages", {"body": message})
    return f"Posted to #{channel}. (message id {res.get('id', '?')})"


@registry.tool(
    "direct_message",
    group="comms",
    risk=Risk.INTERNAL_COMMS,
    rate_bucket="chat",
    description="""Send a private message to one human teammate. Use for coaching, a
specific ask, or anything that would be noise in a channel. Get user ids from
list_team. Note: DMs to an AI employee (an id starting 'ai-') are silently never
answered by this app — use ask_ai_employee instead.""",
    schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "The teammate's user id, from list_team."},
            "message": {"type": "string"},
        },
        "required": ["user_id", "message"],
    },
)
async def direct_message(ctx: ToolContext, user_id: str, message: str) -> str:
    if str(user_id).startswith("ai-"):
        return ("A DM to an AI employee is stored and never answered — the app only "
                "generates replies for @mentions in a channel. Use ask_ai_employee.")
    await ctx.client.post(f"/chat/dm/{user_id}/messages", {"body": message})
    return f"Sent a direct message to {user_id}."


@registry.tool(
    "list_team",
    group="comms",
    risk=Risk.READ,
    description="""Everyone Atlas can talk to: human teammates with their ids and roles,
plus the AI employees (ids beginning 'ai-'). Call this before messaging anyone by id.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def list_team(ctx: ToolContext) -> dict:
    members = await ctx.client.get("/chat/members")
    rows = members.get("members", members) if isinstance(members, dict) else members
    humans, ais = [], []
    for m in rows or []:
        entry = {"id": m.get("id"), "name": m.get("name"), "role": m.get("role")}
        (ais if m.get("is_ai") else humans).append(entry)
    return {"humans": humans, "ai_employees": ais,
            "note": "Message humans with direct_message; ask AI employees with ask_ai_employee."}


@registry.tool(
    "announce",
    group="comms",
    risk=Risk.INTERNAL_COMMS,
    rate_bucket="chat",
    description="""Send a company-wide announcement that everyone sees, above the chat.
Reserve it for things that genuinely need everyone: a target change, a new offer, a
policy. Overuse burns the team's attention and makes the next one ignorable.""",
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["title", "body"],
    },
)
async def announce(ctx: ToolContext, title: str, body: str) -> str:
    await ctx.client.post("/admin/announcements", {"title": title, "body": body})
    return f"Announced to the whole team: {title!r}"


@registry.tool(
    "read_channel",
    group="comms",
    risk=Risk.READ,
    description="""Read recent messages in a channel — what the team is actually saying.
Use it before deciding anything about people: morale, blockers and confusion show up
here long before they show up in the numbers.""",
    schema={
        "type": "object",
        "properties": {
            "channel": {"type": "string", "enum": list(CHANNELS)},
            "limit": {"type": "integer", "description": "How many messages back. Default 30."},
        },
        "required": ["channel"],
    },
)
async def read_channel(ctx: ToolContext, channel: str, limit: int = 30) -> dict:
    if channel not in CHANNELS:
        return {"error": f"'{channel}' is not a channel. Valid: {', '.join(CHANNELS)}"}
    data = await ctx.client.get(f"/chat/channels/{channel}/messages", limit=limit)
    msgs = data.get("messages", data) if isinstance(data, dict) else data
    return {"channel": channel, "messages": [
        {"who": m.get("sender_name"), "is_ai": bool(m.get("is_ai")),
         "at": m.get("created_at"), "body": m.get("body")}
        for m in (msgs or [])[-limit:]
    ]}


# --------------------------------------------------------------------------
# AI employees
# --------------------------------------------------------------------------

@registry.tool(
    "ask_ai_employee",
    group="comms",
    risk=Risk.INTERNAL_COMMS,
    rate_bucket="chat",
    description="""Ask one of the AI employees a question and wait for the answer.
They are Viktor (pipeline), Nadia (lead flow), Iris (client health) and Sol (coaching),
and each is grounded in the live business figures. Use them as your analysts: ask what
they are seeing before you decide something in their area.
Ask a real question — a bare mention with no question makes them post a canned status
check instead of thinking. The answer takes ten to thirty seconds and always appears in
#general, which is normal and visible to the team.""",
    schema={
        "type": "object",
        "properties": {
            "handle": {"type": "string", "enum": sorted(KNOWN_EMPLOYEES),
                       "description": "Which colleague to ask."},
            "question": {"type": "string",
                         "description": "A specific question. Not empty — an empty one is not answered by a model."},
            "wait_seconds": {"type": "integer",
                             "description": "How long to wait for the reply. Default 45, max 90."},
        },
        "required": ["handle", "question"],
    },
)
async def ask_ai_employee(ctx: ToolContext, handle: str, question: str,
                          wait_seconds: int = 45) -> dict:
    """Post the mention, then poll #general for the reply.

    The app gives us nothing to correlate on: the reply carries no reference to
    the question. The only sound match is "a message from this employee, newer
    than the one we just posted", so we record our own message's timestamp and
    take the first employee message after it.
    """
    handle = (handle or "").lower().strip()
    if handle not in KNOWN_EMPLOYEES:
        return {"error": f"No AI employee called {handle!r}. Available: "
                         f"{', '.join(sorted(KNOWN_EMPLOYEES))}."}
    question = (question or "").strip()
    if not question:
        return {"error": "An empty question makes the app skip the model entirely and post a "
                         "canned check. Ask something specific."}

    wait_seconds = max(5, min(90, int(wait_seconds or 45)))
    body = f"@{handle} {question}"

    posted = await ctx.client.post("/chat/channels/general/messages", {"body": body})
    asked_at = posted.get("created_at") or ""
    log.info("atlas: asked @%s in #general", handle)

    sender = f"ai-{handle}"
    deadline = time.monotonic() + wait_seconds
    # Poll with a short backoff: the reply is generated by a background task,
    # so the first second or two is always empty.
    delay = 3.0
    while time.monotonic() < deadline:
        await asyncio.sleep(delay)
        delay = min(6.0, delay * 1.3)
        data = await ctx.client.get("/chat/channels/general/messages", limit=25)
        msgs = data.get("messages", data) if isinstance(data, dict) else data
        for m in reversed(msgs or []):
            if m.get("sender_id") != sender:
                continue
            if asked_at and str(m.get("created_at") or "") <= str(asked_at):
                continue
            return {"handle": handle, "answered": True, "answer": m.get("body"),
                    "at": m.get("created_at")}

    # A missing reply is normal enough to be worth explaining rather than
    # reporting as a failure: the app silently skips the reply when the asker
    # is over 12 replies in the clock hour, or when its AI budget is spent.
    return {
        "handle": handle, "answered": False,
        "note": (f"No reply from {handle} within {wait_seconds}s. The app generates these "
                 f"in the background and drops them silently when the asker is over 12 "
                 f"replies/hour or the team AI budget is spent. The question is posted in "
                 f"#general either way — check back later rather than asking again now."),
    }


@registry.tool(
    "ai_employee_status",
    group="comms",
    risk=Risk.READ,
    description="""The AI employee roster with, for each, when they last spoke and what
they said, when they last ran, and whether the scheduler is on at all.
Read this before concluding 'nothing is wrong' from a quiet channel: silence is their
normal healthy output, so quiet-because-fine and quiet-because-not-running look
identical without this.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def ai_employee_status(ctx: ToolContext) -> dict:
    return await ctx.client.get("/admin/ai-employees")


@registry.tool(
    "assign_ai_employee_task",
    group="comms",
    risk=Risk.STAGE,
    description="""Queue a specific job for an AI employee, done on their next wake-up.
The app narrows your free text to one of exactly four actions — write a report, refill
the lead pool, run a call round, or refuse — so phrase the task as one of those.
Anything else comes back as a polite refusal, which is a real outcome, not an error.
Refilling the pool only works if saved searches exist (see save_prospect_search).""",
    schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Plain-English instruction."},
            "handle": {"type": "string", "enum": sorted(KNOWN_EMPLOYEES),
                       "description": "Who should do it. Omit and the first to wake takes it — "
                                      "in practice almost always Viktor."},
        },
        "required": ["task"],
    },
)
async def assign_ai_employee_task(ctx: ToolContext, task: str,
                                  handle: Optional[str] = None) -> str:
    body: dict[str, Any] = {"task": task}
    if handle:
        body["handle"] = handle.lower()
    res = await ctx.client.post("/admin/ai-employees/tasks", body)
    return (f"Queued for {handle or 'whoever wakes first'}: {task!r} (id {res.get('id','?')}). "
            f"Read the outcome later with review_ai_employee_tasks — a task the app cannot "
            f"classify is stored as done with a refusal sentence, not as an error.")


@registry.tool(
    "review_ai_employee_tasks",
    group="comms",
    risk=Risk.READ,
    description="""What you have asked the AI employees to do and what came of it.
Read the result text, not just the status: a task the app could not classify is recorded
as 'done' with a refusal sentence inside it.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def review_ai_employee_tasks(ctx: ToolContext) -> dict:
    data = await ctx.client.get("/admin/ai-employees/tasks")
    rows = data.get("tasks", data) if isinstance(data, dict) else data
    out = []
    for t in rows or []:
        result = t.get("result") or ""
        refused = "cannot" in result.lower() or "my own schedule" in result.lower()
        out.append({"id": t.get("id"), "task": t.get("task"), "status": t.get("status"),
                    "handle": t.get("handle"), "result": result,
                    "actually_refused": refused})
    return {"tasks": out}


@registry.tool(
    "wake_ai_employees",
    group="comms",
    risk=Risk.INTERNAL_COMMS,
    description="""Make every AI employee look at the business right now instead of
waiting for their schedule, and report what they found.
They observe and speak; this path deliberately cannot make them act, so it never places
a call. Use it when you suspect the scheduler has stopped, or before a briefing when you
want their current read.
One cost worth knowing: this consumes each employee's time-window claim, so the next
scheduled tick will find the window taken and stay silent. Do not call it repeatedly.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def wake_ai_employees(ctx: ToolContext) -> dict:
    res = await ctx.client.post("/admin/ai-employees/run")
    return {"posted": res.get("posted"), "dms_sent": res.get("dmed"),
            "note": res.get("note"),
            "reminder": "Their windows are now claimed; the next scheduled run will be quiet."}


# --------------------------------------------------------------------------
# the AI Operator
# --------------------------------------------------------------------------

@registry.tool(
    "operator_tasks",
    group="comms",
    risk=Risk.READ,
    description="""The AI Operator's task list — the operational work queued in the app.
Read it so Atlas does not duplicate what the Operator is already handling.""",
    schema={"type": "object", "properties": {}, "required": []},
)
async def operator_tasks(ctx: ToolContext) -> Any:
    return await ctx.client.get("/operator/tasks")


@registry.tool(
    "create_operator_task",
    group="comms",
    risk=Risk.STAGE,
    description="""Put a piece of operational work on the AI Operator's queue.
Use it to hand off execution detail Atlas should not do itself, so the work is visible
in the app where the owner already looks.""",
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "detail": {"type": "string", "description": "Enough context to act without asking."},
        },
        "required": ["title"],
    },
)
async def create_operator_task(ctx: ToolContext, title: str, detail: str = "") -> str:
    res = await ctx.client.post("/operator/tasks", {"title": title, "detail": detail,
                                                    "description": detail})
    return f"Queued for the AI Operator: {title!r} (id {res.get('id','?')})"
