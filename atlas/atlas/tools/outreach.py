"""Sending: the one thing the acquisition pipeline could never do.

Atlas can find a prospect, audit their site, build them a replacement and draft
the message that cites the faults. Then the chain stops, because nothing in the
tool surface *sends*. `record_outreach_sent` only asserts that a human sent
something, which is why its docstring in growth.py spends a paragraph warning
that it can lie.

This module is the missing verb, and since 2026-09-02 it is fully wired.

THIS SENDS FOR REAL
-------------------
``POST /admin/prospects/{id}/send`` exists in the app and is what these tools
call. It re-checks the do-not-contact list, sends through
``send_prospect_email()`` / ``send_prospect_sms()``, records the CRM touch
itself, and answers with one word from the app's ``PROSPECT_SEND_OUTCOMES``:
sent / suppressed / no_address / failed.

**This paragraph used to say the opposite, and that was the most dangerous
thing in the file.** For a few hours the endpoints were wired while the
docstring and all three tool descriptions still told the model that no send
route existed, that every row came back ``not_wired``, and that nothing was
written to the CRM — in other words, that calling this was a free rehearsal.
A model told a fifty-row batch is inert has a positive reason to run one. The
descriptions then went further and instructed it to report the result as
un-sent, so fifty real messages could have been summarised to the owner as
nothing having happened. Nothing about the code was wrong; the sentences
handed to the thing that decides were.

The rule that leaves behind: SEND_ENDPOINTS and the prose are one claim, and
``test_no_description_claims_the_endpoint_is_missing`` fails the build if they
disagree again.

``not_wired`` still exists, and now means only what it says: the route is
unexpectedly absent. It is the one status under which nothing left.

The surrounding machinery: the batch cap, the do-not-contact check against the
app's suppression list, per-prospect failure accounting, and de-duplication of
the id list before anybody is contacted.

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
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ..guardrails.policy import Risk
from .registry import ToolContext, registry

log = logging.getLogger("atlas.tools.outreach")

#: One approval covers one batch, so the batch has to be small enough that a
#: person can actually judge it. Fifty is the ceiling; it is refused before any
#: call is made, not part-way through.
MAX_BATCH = 50

CHANNELS = ("email", "sms")

#: The app route that really sends, per channel. Wired on 2026-09-02 to
#: POST /admin/prospects/{id}/send, which the app added
#: for exactly this caller. The route checks the do-not-contact list, sends,
#: and records the CRM touch itself; it answers with an `outcome` word from
#: the app's PROSPECT_SEND_OUTCOMES: sent / suppressed / no_address / failed.
SEND_ENDPOINTS: dict[str, Optional[str]] = {
    "email": "/admin/prospects/{prospect_id}/send",
    "sms": "/admin/prospects/{prospect_id}/send",
}

#: The statuses whose meaning is "nothing left, and it is safe to try again
#: later". Everything else is either a real send or an unknown, and an unknown
#: must never be retried -- see UNKNOWN_STATUS.
SAFE_TO_RETRY = ("failed",)

#: The status for "we do not know whether this person was messaged". A read
#: timeout on a POST the app may well have completed is NOT the same as the app
#: declining, and the difference matters more here than anywhere else in this
#: module: retrying a send that actually happened messages a stranger twice.
#: TWSClient deliberately never auto-retries a POST, so without this the two
#: collapse into "failed" and invite exactly that retry.
UNKNOWN_STATUS = "unknown"

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


async def _count_one_send(ctx: ToolContext) -> None:
    """Spend one unit of the hourly outreach cap, durably.

    `dispatch` persists ONE counter row for the whole tool call, so a batch of
    forty sends left a single row. In memory the limiter was topped up per
    send and the cap held -- until a restart, when `_hydrate_rate_limits`
    rebuilt the hour from storage and restored 1 instead of 40, handing back
    an allowance that had already been spent on real strangers.

    Both halves are written here for that reason. A failure to record is
    logged and swallowed: the message has already gone, and raising now would
    turn a bookkeeping problem into a lost send report.
    """
    try:
        ctx.policy.limiter.record("outreach")
    except Exception:
        log.exception("atlas: could not count an outreach send in memory")
    store = getattr(ctx, "store", None)
    if store is None:
        return
    try:
        await store["counters"].insert_one({
            "id": str(uuid.uuid4()), "bucket": "outreach",
            "tool": "send_outreach_batch",
            "at": datetime.now(timezone.utc).isoformat()})
    except Exception:
        log.exception("atlas: could not persist an outreach counter")


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
        # "The app said no" and "we never heard back" are different facts, and
        # only the first is safe to retry. A read timeout on a POST the app
        # completed leaves a message already delivered; calling that "failed"
        # invites a retry that messages the same stranger a second time.
        return UNKNOWN_STATUS, (
            "the send did not come back with an answer (%s). It MAY have been "
            "delivered -- do not retry it; check the prospect's touches in the "
            "CRM before doing anything else with them." % str(e)[:160])

    outcome = str((res or {}).get("outcome") or "") if isinstance(res, dict) else ""
    status = _APP_OUTCOMES.get(outcome, "failed")
    detail = str((res or {}).get("detail") or "") if isinstance(res, dict) else ""
    if status == "failed" and not detail:
        detail = f"the app answered without a readable outcome: {str(res)[:160]}"
    return status, detail


async def _run(ctx: ToolContext, prospect_ids: list, channel: str,
               subject: str, body: str) -> dict:
    """The whole of a send, batch or single. Never raises for one prospect."""
    # De-duplicated, in the order given. A repeated id in one batch is two
    # messages to one person, and nothing downstream would catch it: the app
    # route does not look at previous touches before sending.
    ids: list = []
    duplicates = 0
    for raw in (prospect_ids or []):
        pid = str(raw).strip()
        if not pid:
            continue
        if pid in ids:
            duplicates += 1
            continue
        ids.append(pid)
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
    sent = skipped = failed = unknown = 0
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
                await _count_one_send(ctx)
        elif status == "opted_out":
            # The app's own suppression check caught what the pre-check did
            # not -- the person still asked us to stop, which is the same fact
            # and belongs in the same column. Counting it as a failure would
            # read as "something went wrong, try again" for the one outcome
            # that must never be retried.
            skipped += 1
        elif status == UNKNOWN_STATUS:
            # Deliberately its own count. It is not a success to report and not
            # a failure to retry.
            unknown += 1
        else:
            # 'not_wired' counts as failed, not as a quiet success. An agent
            # that reported "ok" here would be claiming a send that never was.
            failed += 1
        results.append({"prospect_id": pid, "status": status, "reason": reason})

    not_wired = sum(1 for r in results if r["status"] == "not_wired")
    notes = []
    if duplicates:
        notes.append(f"{duplicates} repeated id(s) were removed before sending, so "
                     f"nobody was messaged twice by this call.")
    if not_wired:
        notes.append(f"{not_wired} of {len(ids)} reached nobody: the send route was "
                     f"unexpectedly absent, so nothing was sent and NO CRM touch was "
                     f"written for them. Do not claim these were sent.")
    if unknown:
        notes.append(f"{unknown} send(s) never came back with an answer and MAY have "
                     f"been delivered. Do NOT retry them; say so plainly in your "
                     f"summary and check the CRM touches before working those "
                     f"prospects again.")
    if sent:
        notes.append(f"{sent} message(s) actually reached a real person and cannot be "
                     f"un-sent.")
    notes.append("Counts are per prospect; read the reasons before reporting.")

    return {
        "channel": channel,
        "requested": len(ids),
        "sent": sent,
        "skipped_opt_out": skipped,
        "failed": failed,
        "unknown": unknown,
        "results": results,
        "note": " ".join(notes),
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
IMPORTANT, READ BEFORE USING: this SENDS FOR REAL. A row that comes back 'sent' is an
email a stranger has received and cannot un-receive, and the app has already recorded the
CRM touch and moved them to 'contacted'. There is no dry-run: do not call this to see
what would happen.
A row that comes back 'unknown' MAY have been delivered — never retry it; say so and
check the CRM.
Returns counts — sent / skipped_opt_out / failed / unknown — with a reason per prospect.""",
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
IMPORTANT, READ BEFORE USING: this SENDS FOR REAL. A row that comes back 'sent' is a
text on a real stranger's phone and cannot be unsent; the app has already recorded the
CRM touch. There is no dry-run mode.
A row that comes back 'unknown' MAY have been delivered — never retry it; say so and
check the CRM.
Returns counts — sent / skipped_opt_out / failed / unknown — with a reason per prospect.""",
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
IMPORTANT, READ BEFORE USING: this SENDS FOR REAL, to every prospect in the list at
once. Each 'sent' row is a message a stranger has received and cannot un-receive. There
is no dry-run mode and no way to recall a batch: do not call this to exercise the
pipeline or to see what it would do.
Repeated ids are removed before sending. A row that comes back 'unknown' MAY have been
delivered — never retry it.""",
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
