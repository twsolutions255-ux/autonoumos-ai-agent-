"""Appointments: noticing one, and making sure the owner hears about it.

THE GAP THIS CLOSES. Atlas could read a COUNT of demo bookings
(`/admin/bookings/stats`) and it could send a direct message. It could not see
WHICH appointments were new, so it had nothing specific to say and no way to
avoid saying it twice. The owner's own words for what he wanted: "atlas will
command one of the ai employees to dm me and then i will know that atlas got a
apointment with the ai cold caller".

WHY THIS IS ONE TOOL AND NOT TWO. The obvious build is a read tool plus a
prompt instruction telling the model to message the owner when it sees
something. That makes the notification depend on the model remembering, every
cycle, forever -- and the failure is silent, because a cycle that simply does
not mention appointments looks identical to a day with no appointments. So
finding them, announcing them, and recording that they were announced is one
call that either happens or does not.

WHAT MAKES IT SAFE TO RUN EVERY CYCLE. The announcement is idempotent: every
booking id that has been announced is written to `announced_appointments`, and
this only ever messages about ids that are not there. A restart, a redeploy or
three cycles in an hour cannot produce three messages about one appointment --
which matters more than it sounds, because a notification that cries wolf is
one the owner turns off, and then the real one arrives silently.

IT IS INTERNAL. This messages the owner inside the app's own chat. It sends
nothing to a lead, a prospect or a client, which is why it sits at the
`assist` rung rather than behind the external-comms gate.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..guardrails.policy import Risk
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.pipeline")

#: How far back a sweep looks when nothing has ever been announced. Without a
#: bound, the first run after deployment would message the owner about every
#: appointment in the history of the business.
FIRST_RUN_HOURS = 48

#: The most appointments named individually in one message. Beyond this the
#: message says how many there are and stops naming them -- a DM nobody
#: finishes reading is a DM that did not notify anybody.
MAX_NAMED = 8


def _booked_by_ai_caller(booking: dict) -> bool:
    """Whether the AI cold caller is what produced this appointment.

    Read from the fields the app actually sets rather than inferred from the
    absence of something: `source` is written by the ingest path, and the
    speed-to-lead and cold-call paths stamp their own marks.
    """
    source = str(booking.get("source") or "").lower()
    if booking.get("speed_to_lead_call_id") or booking.get("retell_call_id"):
        return True
    return any(mark in source for mark in ("cold", "retell", "speed", "agent_api"))


def _describe(booking: dict) -> str:
    who = (booking.get("business_name") or booking.get("full_name")
           or booking.get("email") or "a lead")
    when = str(booking.get("date") or "").strip()
    at = str(booking.get("time") or "").strip()
    slot = (" for %s %s" % (when, at)).rstrip() if when or at else ""
    closer = booking.get("closer_name")
    owner = " — %s is taking it" % closer if closer else " — NO CLOSER ASSIGNED"
    how = " (booked by the AI caller)" if _booked_by_ai_caller(booking) else ""
    return "%s%s%s%s" % (who, slot, owner, how)


async def _unannounced(ctx: ToolContext, since_hours: int) -> list:
    """Appointments this tool has not already messaged the owner about."""
    from ..db import now
    from datetime import timedelta

    rows = await ctx.client.get("/bookings")
    if not isinstance(rows, list):
        return []

    cutoff = (now() - timedelta(hours=max(1, since_hours))).isoformat()
    seen: set = set()
    if ctx.store is not None:
        try:
            marks = await ctx.store["announced_appointments"].find(
                {}, {"_id": 0, "booking_id": 1}).to_list(5000)
            seen = {m.get("booking_id") for m in marks}
        except Exception:
            # Better to say nothing than to re-announce everything: an
            # unreadable ledger is not evidence that nothing was announced.
            log.exception("atlas: could not read the announced-appointment ledger")
            raise

    fresh = []
    for b in rows:
        bid = b.get("id")
        if not bid or bid in seen:
            continue
        # Cancelled and lost bookings are not news worth waking somebody for.
        if str(b.get("status") or "").lower() in ("lost", "cancelled", "canceled"):
            continue
        when = str(b.get("created_at") or "")
        if when and when < cutoff:
            continue
        fresh.append(b)
    return fresh


@registry.tool(
    "new_appointments",
    group="pipeline",
    risk=Risk.READ,
    description="""Appointments booked recently that the owner has NOT already been
messaged about, newest first. Each says who it is with, when, which closer is taking it,
and whether the AI caller booked it. Read-only: seeing them here does not tell anybody.
Use announce_new_appointments to actually notify the owner.""",
    schema={
        "type": "object",
        "properties": {
            "since_hours": {"type": "integer",
                            "description": "How far back to look. Default 48."},
        },
    },
)
async def new_appointments(ctx: ToolContext, since_hours: int = FIRST_RUN_HOURS) -> dict:
    try:
        fresh = await _unannounced(ctx, since_hours)
    except Exception as e:
        return {"error": "Could not read the appointment list: %s" % str(e)[:200],
                "appointments": None,
                "note": "This is not 'no appointments' -- it is 'the list could "
                        "not be read'. Do not report it as a quiet day."}
    return {
        "appointments": [
            {"id": b.get("id"), "who": b.get("business_name") or b.get("full_name"),
             "date": b.get("date"), "time": b.get("time"),
             "closer": b.get("closer_name"), "closer_id": b.get("closer_id"),
             "booked_by_ai_caller": _booked_by_ai_caller(b),
             "created_at": b.get("created_at")}
            for b in fresh
        ],
        "count": len(fresh),
        "note": "These have not been announced. Nobody has been told yet.",
    }


@registry.tool(
    "announce_new_appointments",
    group="pipeline",
    risk=Risk.INTERNAL_COMMS,
    rate_bucket="internal_comms",
    description="""Message the OWNER, inside the app, about appointments booked since
the last time this ran — then record that they were announced so they are never sent
twice. This is the notification the owner asked for: he wants to know the moment the AI
caller books somebody.
Internal only: it messages the owner in the app's own chat and sends nothing to any lead
or client. Safe to call every cycle — with nothing new it messages nobody and says so.""",
    schema={
        "type": "object",
        "properties": {
            "since_hours": {"type": "integer",
                            "description": "How far back to look on a first run. Default 48."},
        },
    },
)
async def announce_new_appointments(ctx: ToolContext,
                                    since_hours: int = FIRST_RUN_HOURS) -> dict:
    from ..db import iso

    try:
        fresh = await _unannounced(ctx, since_hours)
    except Exception as e:
        return {"announced": 0,
                "error": "Could not read the appointments: %s" % str(e)[:200],
                "note": "Nobody was messaged, and this is NOT 'there were none'."}
    if not fresh:
        return {"announced": 0,
                "note": "No new appointments, so nobody was messaged. This is the "
                        "normal answer most cycles and is not a failure."}

    # Who to tell. The owner is the superadmin on the team list; asked for by
    # role rather than hardcoded, because an id in a constant is one that goes
    # stale silently the day the account changes.
    owner_id = None
    try:
        members = await ctx.client.get("/chat/members")
        people = members if isinstance(members, list) else (members or {}).get("members") or []
        for m in people:
            if str(m.get("role") or "").lower() in ("superadmin", "super_admin"):
                owner_id = m.get("id")
                break
    except Exception as e:
        return {"announced": 0,
                "error": "Could not read the team list, so there was nobody to "
                         "message: %s" % str(e)[:160],
                "found": len(fresh)}
    if not owner_id:
        return {"announced": 0,
                "error": "No superadmin found on the team list, so there is no "
                         "owner to message.",
                "found": len(fresh)}

    named = fresh[:MAX_NAMED]
    lines = ["%d new appointment%s booked." % (len(fresh), "" if len(fresh) == 1 else "s")]
    lines += ["  • %s" % _describe(b) for b in named]
    if len(fresh) > len(named):
        lines.append("  …and %d more." % (len(fresh) - len(named)))
    unassigned = [b for b in fresh if not b.get("closer_id")]
    if unassigned:
        # The one thing in this message that needs an action rather than a nod.
        lines.append("%d of them ha%s no closer assigned — they need an owner "
                     "before they turn into a no-show."
                     % (len(unassigned), "s" if len(unassigned) == 1 else "ve"))
    message = "\n".join(lines)

    await ctx.client.post("/chat/dm/%s/messages" % owner_id, {"body": message})

    # Recorded AFTER the message is away. The other order loses an appointment
    # to a failed send: marked as told, never told.
    if ctx.store is not None:
        for b in fresh:
            try:
                await ctx.store["announced_appointments"].insert_one({
                    "booking_id": b.get("id"), "at": iso(),
                    "who": b.get("business_name") or b.get("full_name"),
                    "closer_id": b.get("closer_id"),
                    "booked_by_ai_caller": _booked_by_ai_caller(b),
                })
            except Exception:
                # A ledger write that fails means this one may be announced
                # again tomorrow. That is the harmless direction.
                log.exception("atlas: could not record an announced appointment")

    return {"announced": len(fresh), "messaged": owner_id,
            "unassigned": len(unassigned), "message": message,
            "note": "The owner has been messaged in the app. Say in your summary "
                    "that you told him, and do not message him about these again."}
