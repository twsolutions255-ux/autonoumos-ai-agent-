"""Telling the owner an appointment was booked — once.

The owner's ask, in his words: "atlas will command one of the ai employees to
dm me and then i will know that atlas got a apointment with the ai cold
caller". Atlas could already read a COUNT of bookings and could already send a
direct message; what it could not do was see WHICH appointments were new, so
it had nothing specific to say and no way to avoid saying it twice.

The two failures these pin down are the ones that make a notification
worthless rather than merely imperfect:

  announced twice   A notification that repeats is one the owner mutes, and
                    then the real one arrives silently. So the ledger is the
                    point of the tool, not a detail of it.
  silence read as   A cycle that could not READ the bookings must not look
  "nothing booked"  like a cycle that read them and found none. One of those
                    is a quiet day and the other is a broken pipeline, and
                    they are reported as different things.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.guardrails.policy import Policy, RateLimiter, Risk
from atlas.memory.store import MemoryStore
from atlas.tools import pipeline
from atlas.tools.registry import ToolContext, registry
from tests.test_integration import MockStore, settings_for

OWNER = {"id": "boss-1", "name": "Joao", "role": "superadmin"}


class FakeClient:
    """Stands in for TWSClient, recording what was asked and what was sent."""

    def __init__(self, bookings=None, members=None, get_raises=False):
        self._bookings = bookings if bookings is not None else []
        self._members = members if members is not None else [OWNER,
                                                             {"id": "c1", "role": "closer"}]
        self._get_raises = get_raises
        self.calls = []

    async def get(self, path, **kw):
        self.calls.append(("GET", path, None))
        if self._get_raises and path == "/bookings":
            raise RuntimeError("the app is unreachable")
        if path == "/bookings":
            return self._bookings
        if path == "/chat/members":
            return self._members
        return {}

    async def post(self, path, body=None, **kw):
        self.calls.append(("POST", path, body or {}))
        return {"ok": True}

    def sent_messages(self):
        return [b.get("body", "") for m, p, b in self.calls
                if m == "POST" and "/chat/dm/" in p]


def harness(client, autonomy="assist"):
    settings = settings_for(ATLAS_AUTONOMY=autonomy, ATLAS_SANDBOX="false")
    store = MockStore()
    return ToolContext(client=client, store=store, memory=MemoryStore(store),
                       policy=Policy(settings, RateLimiter()), settings=settings,
                       cycle_id="cyc-test")


def booking(bid, **over):
    row = {"id": bid, "business_name": "Ace Roofing " + bid,
           "date": "2026-09-05", "time": "10:00",
           "closer_id": "c1", "closer_name": "Joseph",
           "status": "booked", "source": "agent_api",
           "created_at": "2999-01-01T00:00:00+00:00"}   # always inside the window
    row.update(over)
    return row


# ------------------------------------------------------------ the gate

def test_announcing_is_internal_comms_not_external():
    """It messages the owner inside the app and nobody outside it. Classing it
    with the send tools would put a notification the owner asked for behind an
    approval he would then have to give for his own message."""
    tool = registry.get("announce_new_appointments")
    assert tool is not None
    assert tool.policy.risk is Risk.INTERNAL_COMMS
    assert registry.get("new_appointments").policy.risk is Risk.READ


# ------------------------------------------------------------ once, and only once

@pytest.mark.asyncio
async def test_the_owner_is_messaged_once_per_appointment():
    client = FakeClient(bookings=[booking("b1"), booking("b2")])
    ctx = harness(client)

    first = await pipeline.announce_new_appointments(ctx)
    assert first["announced"] == 2
    assert len(client.sent_messages()) == 1
    said = client.sent_messages()[0]
    assert "2 new appointments" in said
    assert "Ace Roofing b1" in said and "Ace Roofing b2" in said

    # The same cycle running again -- a redeploy, a retry, three cycles in an
    # hour -- must not message him again.
    second = await pipeline.announce_new_appointments(ctx)
    assert second["announced"] == 0
    assert len(client.sent_messages()) == 1
    assert "not a failure" in second["note"]

    # A genuinely new one still gets through.
    client._bookings.append(booking("b3"))
    third = await pipeline.announce_new_appointments(ctx)
    assert third["announced"] == 1
    assert len(client.sent_messages()) == 2
    assert "Ace Roofing b3" in client.sent_messages()[1]
    assert "b1" not in client.sent_messages()[1]


@pytest.mark.asyncio
async def test_it_messages_the_superadmin_and_not_a_closer():
    client = FakeClient(bookings=[booking("b1")])
    ctx = harness(client)
    await pipeline.announce_new_appointments(ctx)
    dms = [p for m, p, _b in client.calls if m == "POST" and "/chat/dm/" in p]
    assert dms == ["/chat/dm/boss-1/messages"]


# ------------------------------------------------------------ what it says

@pytest.mark.asyncio
async def test_an_appointment_with_no_closer_is_called_out():
    """The one line in the message that needs an action rather than a nod."""
    client = FakeClient(bookings=[booking("b1", closer_id=None, closer_name=None)])
    ctx = harness(client)
    out = await pipeline.announce_new_appointments(ctx)
    said = client.sent_messages()[0]
    assert "NO CLOSER ASSIGNED" in said
    assert "before they turn into a no-show" in said
    assert out["unassigned"] == 1


@pytest.mark.asyncio
async def test_an_ai_booked_appointment_says_so():
    """What the owner actually asked to know: that the AI caller booked it."""
    client = FakeClient(bookings=[booking("b1", source="web", speed_to_lead_call_id="call_9")])
    ctx = harness(client)
    await pipeline.announce_new_appointments(ctx)
    assert "booked by the AI caller" in client.sent_messages()[0]


@pytest.mark.asyncio
async def test_a_long_list_is_summarised_rather_than_dumped():
    client = FakeClient(bookings=[booking("b%d" % i) for i in range(20)])
    ctx = harness(client)
    out = await pipeline.announce_new_appointments(ctx)
    said = client.sent_messages()[0]
    assert out["announced"] == 20
    assert "20 new appointments" in said
    assert "and 12 more" in said
    # All twenty are still marked, so tomorrow does not re-announce the twelve.
    again = await pipeline.announce_new_appointments(ctx)
    assert again["announced"] == 0


# ------------------------------------------------------------ silence vs failure

@pytest.mark.asyncio
async def test_nothing_new_messages_nobody():
    client = FakeClient(bookings=[])
    ctx = harness(client)
    out = await pipeline.announce_new_appointments(ctx)
    assert out["announced"] == 0
    assert client.sent_messages() == []
    assert "normal answer" in out["note"]


@pytest.mark.asyncio
async def test_an_unreadable_list_is_not_reported_as_no_appointments():
    """A cycle that could not read the bookings and a cycle that read them and
    found none are different facts. Collapsing them is how a broken pipeline
    reads as a quiet week."""
    client = FakeClient(get_raises=True)
    ctx = harness(client)
    out = await pipeline.announce_new_appointments(ctx)
    assert out["announced"] == 0
    assert "error" in out
    assert "NOT 'there were none'" in out["note"]
    assert client.sent_messages() == []

    read = await pipeline.new_appointments(ctx)
    assert read["appointments"] is None
    assert "not 'no appointments'" in read["note"]


@pytest.mark.asyncio
async def test_no_owner_on_the_team_list_is_an_error_not_a_silent_success():
    client = FakeClient(bookings=[booking("b1")], members=[{"id": "c1", "role": "closer"}])
    ctx = harness(client)
    out = await pipeline.announce_new_appointments(ctx)
    assert out["announced"] == 0
    assert "no owner to message" in out["error"]
    assert out["found"] == 1
    assert client.sent_messages() == []


# ------------------------------------------------------------ what it leaves alone

@pytest.mark.asyncio
async def test_a_lost_booking_is_not_news():
    client = FakeClient(bookings=[booking("b1", status="lost"),
                                  booking("b2", status="cancelled")])
    ctx = harness(client)
    out = await pipeline.announce_new_appointments(ctx)
    assert out["announced"] == 0
    assert client.sent_messages() == []


@pytest.mark.asyncio
async def test_an_old_booking_does_not_arrive_as_news_on_first_run():
    """Without a window, the first run after deployment would message the owner
    about every appointment in the history of the business."""
    client = FakeClient(bookings=[booking("old", created_at="2020-01-01T00:00:00+00:00")])
    ctx = harness(client)
    out = await pipeline.announce_new_appointments(ctx)
    assert out["announced"] == 0
    assert client.sent_messages() == []


@pytest.mark.asyncio
async def test_reading_never_tells_anybody():
    """new_appointments is the read half. Seeing them must not notify."""
    client = FakeClient(bookings=[booking("b1")])
    ctx = harness(client)
    out = await pipeline.new_appointments(ctx)
    assert out["count"] == 1
    assert client.sent_messages() == []
    assert "Nobody has been told" in out["note"]
    # And it did not consume them: announcing still works afterwards.
    said = await pipeline.announce_new_appointments(ctx)
    assert said["announced"] == 1
