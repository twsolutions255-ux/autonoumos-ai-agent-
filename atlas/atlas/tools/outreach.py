"""Sending: the one thing the acquisition pipeline could never do.

Atlas can find a prospect, audit their site, build them a replacement and draft
the message that cites the faults. Then the chain stops, because nothing in the
tool surface *sends*. `record_outreach_sent` only asserts that a human sent
something, which is why its docstring in growth.py spends a paragraph warning
that it can lie.

This module is the missing verb, and it is deliberately honest about how much
of it is currently real.

WHAT THE APP ACTUALLY EXPOSES
-----------------------------
`server.py` contains `send_prospect_email()` and `send_prospect_sms()` — real
functions, with the do-not-call assertion welded in — but **neither is reachable
over HTTP**. Every caller is another internal function (booking confirmations,
speed-to-lead, the SMS self-test). There is no route of the shape
``POST /admin/prospects/{id}/send``, and none of the POST routes matching
send/sms/message/email/outreach is a general prospect-outreach send:

    /admin/sites/{slug}/outreach     regenerates COPY, sends nothing
    /admin/prospects/{id}/followup   DRAFTS the next message
    /closer/messages                 internal staff chat
    /admin/sms/self-test             texts a number you supply, as a test
    /admin/system-test/email         same, for email

So the send endpoints below are declared and empty. Every send therefore
reports ``not_wired`` and NOTHING is written to the CRM — logging a touch for a
message that never left would be exactly the false assertion growth.py warns
about, and it costs the prospect permanently.

The surrounding machinery is real and runs today: the batch cap, the do-not-
contact check against the app's suppression list, the per-prospect failure
accounting, and the CRM write that fires the moment a send succeeds. Point
SEND_ENDPOINTS at a real route and this module sends for real, unchanged.

WHAT COUNTS AS OPTED OUT
------------------------
The app's suppression list, read through ``GET /dnc/check?phone=&email=``,
which answers ``{"suppressed": bool}``. That endpoint refuses (400) anything it
cannot turn into a lookup key, precisely so "I could not check" is never
mistaken for "they are not on it". This module keeps that distinction: a
prospect whose status could not be established is FAILED, never sent to.
Prospects the app has already marked ``status == "dnc"`` are treated as opted
out without a round trip.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..guardrails.policy import Risk
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.outreach")

#: One approval covers one batch, so the batch has to be small enough that a
#: person can actually judge it. Fifty is the ceiling; it is refused before any
#: call is made, not part-way through.
MAX_BATCH = 50

CHANNELS = ("email", "sms")

#: The app route that really sends, per channel. Both are None because the app
#: has none — see the module docstring. Filling one in is the ONLY change
#: needed to make sends real; everything downstream (the CRM touch, the counts)
#: already handles a success.
#: Wired on 2026-09-02 to POST /admin/prospects/{id}/send, which the app added
#: for exactly this caller. The route checks the do-not-contact list, sends,
#: and records the CRM touch itself; it answers with an `outcome` word from
#: the app's PROSPECT_SEND_OUTCOMES: sent / suppressed / no_address / failed.
SEND_ENDPOINTS: dict[str, Optional[str]] = {
    "email": "/admin/prospects/{prospect_id}/send",
    "sms": "/admin/prospects/{prospect_id}/send",
}

#: How the app's outcome words map onto this tool's statuses. "suppressed"
#: becomes opted_out so the batch counts it with the pre-checked opt-outs;
#: anything unrecognised is FAILED, because an answer this tool cannot read
#: is not evidence that anything was sent.
_APP_OUTCOMES = {"sent": "sent", "suppressed": "opted_out",
                 "no_address": "failed", "failed": "failed"}

_NOT_WIRED = (
    "not wired: the app has no endpoint for sending {what} to a prospect. "
    "server.py has send_prospect_{fn}() but no HTTP route reaches it, so "
    "nothing was sent and nothing was logged to the CRM."
)

#: Said in every description, verbatim, because a model that reads "denied"
#: retries and a model that reads this does not.
_GATE = ("This queues for the owner's approval below operate — it is not a "
         "failure and must not be retried; say it is waiting and move on.")


# --------------------------------------------------------------------------
# the parts that are real
# --------------------------------------------------------------------------

async def _load_prospects(client) -> dict:
    """Every prospect, by id. One call for the whole batch."""
    data = await client.get("/admin/prospects")
    rows = data.get("prospects", []) if isinstance(data, dict) else (data or [])
    return {str(r.get("id")): r for r in rows if r.get("id")}


async def _opt_out_state(client, prospect: dict) -> tuple:
    """(state, reason) where state is 'clear', 'opted_out' or 'unknown'.

    'unknown' is never treated as permission. The app's own /dnc/check refuses
    to answer about a contact it cannot key, and on this list the difference
    between "not on it" and "could not ask" is the difference between a legal
    message and a complaint.
    """
    if (prospect.get("status") or "").lower() == "dnc":
        return "opted_out", "the app has this prospect marked dnc"

    phone = (prospect.get("phone") or "").strip()
    email = (prospect.get("email") or "").strip()
    if not phone and not email:
        return "unknown", ("no phone or email on the prospect, so the "
                           "do-not-contact list could not be checked")
    try:
        res = await client.get("/dnc/check", phone=phone or None, email=email or None)
    except Exception as e:
        return "unknown", f"the do-not-contact check failed: {str(e)[:200]}"

    if not isinstance(res, dict):
        return "unknown", "the do-not-contact check returned no answer"
    if "suppressed" in res:
        hit = bool(res["suppressed"])
    elif "listed" in res:
        hit = bool(res["listed"])
    else:
        return "unknown", ("the do-not-contact check answered in a shape this "
                           "tool does not recognise: %r" % sorted(res)[:6])
    return ("opted_out", "on the app's do-not-contact list") if hit else ("clear", "")


async def _send_one(client, channel: str, prospect: dict, subject: str,
                    body: str) -> tuple:
    """(status, reason). status is 'sent', 'opted_out', 'not_wired' or 'failed'.

    The app's send route records the CRM touch ITSELF on a real send, so this
    function must not post a second one -- that is how one message becomes
    two rows and the follow-up clock starts twice. It reads the route's
    `outcome` word and passes the app's own `detail` back as the reason.
    """
    endpoint = SEND_ENDPOINTS.get(channel)
    if not endpoint:
        return "not_wired", _NOT_WIRED.format(
            what="an email" if channel == "email" else "a text message",
            fn="email" if channel == "email" else "sms")

    pid = str(prospect.get("id"))
    payload: dict[str, Any] = {"channel": channel, "body": body}
    if channel == "email":
        payload["subject"] = subject
    try:
        res = await client.post(endpoint.format(prospect_id=pid), payload)
    except Exception as e:
        return "failed", f"the app refused the send: {str(e)[:200]}"

    outcome = str((res or {}).get("outcome") or "") if isinstance(res, dict) else ""
    status = _APP_OUTCOMES.get(outcome, "failed")
    detail = str((res or {}).get("detail") or "") if isinstance(res, dict) else ""
    if status == "failed" and not detail:
        detail = f"the app answered without a readable outcome: {str(res)[:160]}"
    return status, detail


async def _run(ctx: ToolContext, prospect_ids: list, channel: str,
               subject: str, body: str) -> dict:
    """The whole of a send, batch or single. Never raises for one prospect."""
    ids = [str(p) for p in (prospect_ids or []) if str(p).strip()]
    if not ids:
        return {"refused": "No prospect ids were given, so nothing was sent."}
    if channel not in CHANNELS:
        return {"refused": f"channel must be one of {', '.join(CHANNELS)}."}
    if len(ids) > MAX_BATCH:
        # Before any call, deliberately. One approval covers one batch, and a
        # batch nobody can read is an approval nobody really gave.
        return {"refused": (
            f"Refused before contacting anyone: {len(ids)} prospects is over the "
            f"{MAX_BATCH} ceiling for a single batch. One approval covers one batch, "
            f"so a batch has to stay small enough for a person to judge. Split it.")}
    if not (body or "").strip():
        return {"refused": "There is no message body, so nothing was sent."}

    try:
        known = await _load_prospects(ctx.client)
    except Exception as e:
        return {"refused": f"Could not read the prospect list, so nothing was sent: {e}"}

    results: list = []
    sent = skipped = failed = 0
    for pid in ids:
        prospect = known.get(pid)
        if prospect is None:
            failed += 1
            results.append({"prospect_id": pid, "status": "failed",
                            "reason": "no such prospect in the CRM"})
            continue

        state, why = await _opt_out_state(ctx.client, prospect)
        if state == "opted_out":
            skipped += 1
            results.append({"prospect_id": pid, "status": "skipped_opt_out",
                            "reason": why})
            continue
        if state == "unknown":
            failed += 1
            results.append({"prospect_id": pid, "status": "failed",
                            "reason": f"not sent — {why}"})
            continue

        status, reason = await _send_one(ctx.client, channel, prospect, subject, body)
        if status == "sent":
            sent += 1
            # dispatch counts ONE outreach event for the whole call, so every
            # send after the first is counted here. Otherwise a batch of forty
            # would spend one unit of the hourly cap.
            if sent > 1:
                try:
                    ctx.policy.limiter.record("outreach")
                except Exception:
                    log.exception("atlas: could not count an outreach send")
        else:
            # 'not_wired' counts as failed, not as a quiet success. An agent
            # that reported "ok" here would be claiming a send that never was.
            failed += 1
        results.append({"prospect_id": pid, "status": status, "reason": reason})

    not_wired = sum(1 for r in results if r["status"] == "not_wired")
    if not_wired:
        note = (f"{not_wired} of {len(ids)} reached nobody: the app exposes no send "
                f"endpoint for prospects, so nothing was sent and NO CRM touch was "
                f"written for them. Do not claim these were sent, and do not retry.")
    else:
        note = "Counts are per prospect; read the reasons before reporting."

    return {
        "channel": channel,
        "requested": len(ids),
        "sent": sent,
        "skipped_opt_out": skipped,
        "failed": failed,
        "results": results,
        "note": note,
    }


# --------------------------------------------------------------------------
# the tools
# --------------------------------------------------------------------------

@registry.tool(
    "send_prospect_email",
    group="growth",
    risk=Risk.EXTERNAL_COMMS,
    rate_bucket="outreach",
    description="""Email one prospect — a real stranger outside the company, who cannot
be un-emailed. """ + _GATE + """
Check the prospect is not on the do-not-contact list first; this tool checks too, and
refuses rather than guessing when the list cannot be asked.
IMPORTANT, READ BEFORE USING: the app currently has NO HTTP endpoint that sends email to
a prospect, so this returns 'not_wired' and sends nothing. That is a fact about the app,
not a failure to retry. Report it as un-sent and do not log a touch for it.
Returns counts — sent / skipped_opt_out / failed — with a reason per prospect.""",
    schema={
        "type": "object",
        "properties": {
            "prospect_id": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string", "description": "The message, as it will be read."},
        },
        "required": ["prospect_id", "subject", "body"],
    },
)
async def send_prospect_email(ctx: ToolContext, prospect_id: str, subject: str,
                              body: str) -> dict:
    return await _run(ctx, [prospect_id], "email", subject, body)


@registry.tool(
    "send_prospect_sms",
    group="growth",
    risk=Risk.EXTERNAL_COMMS,
    rate_bucket="outreach",
    description="""Text one prospect — a real stranger's phone, and it cannot be unsent. """
    + _GATE + """
The do-not-contact list is checked first, and a check that cannot be made counts as a
refusal, never as permission.
IMPORTANT, READ BEFORE USING: the app currently has NO HTTP endpoint that texts a
prospect, so this returns 'not_wired' and sends nothing. Report it as un-sent; do not
retry it and do not log a touch for it.
Returns counts — sent / skipped_opt_out / failed — with a reason per prospect.""",
    schema={
        "type": "object",
        "properties": {
            "prospect_id": {"type": "string"},
            "body": {"type": "string", "description": "The message, as it will be read."},
        },
        "required": ["prospect_id", "body"],
    },
)
async def send_prospect_sms(ctx: ToolContext, prospect_id: str, body: str) -> dict:
    return await _run(ctx, [prospect_id], "sms", "", body)


@registry.tool(
    "send_outreach_batch",
    group="growth",
    risk=Risk.EXTERNAL_COMMS,
    rate_bucket="outreach",
    description="""Send the SAME message to a list of prospects, on one channel. """
    + _GATE + """
One approval covers the whole batch, which is why a batch over 50 is refused outright,
before anybody is contacted — split it instead. Every prospect is checked against the
app's do-not-contact list and skipped if listed; a prospect whose status cannot be
established is failed rather than sent to.
One prospect failing does not stop the rest: you get a per-prospect reason and the
counts sent / skipped_opt_out / failed.
IMPORTANT, READ BEFORE USING: the app currently has NO HTTP endpoint that sends to a
prospect, so every row comes back 'not_wired' and nothing is written to the CRM.""",
    schema={
        "type": "object",
        "properties": {
            "prospect_ids": {"type": "array", "items": {"type": "string"},
                             "description": f"Up to {MAX_BATCH}. More is refused."},
            "channel": {"type": "string", "enum": list(CHANNELS)},
            "template_id_or_body": {
                "type": "string",
                "description": "The message body. There is no template store in the app, "
                               "so this is the literal text every prospect receives."},
            "subject": {"type": "string", "description": "Email only."},
        },
        "required": ["prospect_ids", "channel", "template_id_or_body"],
    },
)
async def send_outreach_batch(ctx: ToolContext, prospect_ids: list, channel: str,
                              template_id_or_body: str, subject: str = "") -> dict:
    return await _run(ctx, list(prospect_ids or []), channel, subject,
                      template_id_or_body)
