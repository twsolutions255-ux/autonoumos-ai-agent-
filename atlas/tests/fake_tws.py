"""A stand-in for the TW Solutions API, faithful to the behaviour that matters.

Not a mock library — a real ASGI app, so the tests exercise the actual HTTP
client: auth headers, retries, 401 re-login, error shapes. The point is to
reproduce the app's *awkward* behaviours, because those are what Atlas has to
survive:

  * login returns the token in the body, and rejects a wrong password
  * an expired token yields 401 once, and the client must re-login and replay
  * an AI-employee reply is FIRE-AND-FORGET: the POST returns only your own
    message, and the answer shows up in #general a moment later
  * /admin/clients leaks a plaintext api_key, as the real one does
  * tenant-scoped routes 400 for a superadmin who omits client_id
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

GOOD_EMAIL = "atlas@tws.test"
GOOD_PASSWORD = "correct-horse"
CRON_SECRET = "cron-secret-value"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class State:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tokens: set = set()
        self.messages: list = []       # #general
        self.dms: list = []
        self.announcements: list = []
        self.prospects: list = [
            {"id": "p1", "name": "Ace Roofing", "phone": "+15551110000",
             "website": None, "score": 91, "status": "new", "touches": []},
            {"id": "p2", "name": "Bright HVAC", "phone": "+15551110001",
             "website": "http://bright.example", "score": 44, "status": "new", "touches": []},
        ]
        self.staged: list = []
        self.released = 0
        self.stopped = 0
        self.autonomy = {"enabled": False, "daily_cap": 30}
        self.jobs_run: list = []
        self.expire_next_token = False
        self.calls: list = []          # every request, for assertions
        self.analyses: list = []
        self.competitors: list = []
        self.client_messages: list = []


state = State()
app = FastAPI()


@app.middleware("http")
async def record(request: Request, call_next):
    state.calls.append((request.method, request.url.path))
    return await call_next(request)


def auth(request: Request) -> None:
    header = request.headers.get("authorization", "")
    token = header[7:] if header.startswith("Bearer ") else ""
    if state.expire_next_token and token in state.tokens:
        # Simulate the app's password-change invalidation exactly once, so the
        # client's re-login-and-replay path is genuinely exercised.
        state.expire_next_token = False
        state.tokens.discard(token)
        raise HTTPException(status_code=401, detail="Token expired")
    if not token or token not in state.tokens:
        raise HTTPException(status_code=401, detail="Not authenticated")


class LoginIn(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
async def login(body: LoginIn):
    if body.email != GOOD_EMAIL or body.password != GOOD_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = f"tok-{uuid.uuid4()}"
    state.tokens.add(token)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/api/auth/me")
async def me(request: Request):
    auth(request)
    return {"id": "u-atlas", "email": GOOD_EMAIL, "name": "Atlas",
            "role": "superadmin", "roles": []}


@app.get("/api/admin/overview")
async def overview(request: Request):
    auth(request)
    return {"clients": 12, "calls": 340, "appointments": 88,
            "demo_bookings": 21, "unread_messages": 3}


@app.get("/api/admin/alerts")
async def alerts(request: Request):
    auth(request)
    return {"alerts": [
        {"type": "webhook_failure", "severity": "high",
         "message": "Retell webhook failed: unmapped agent", "created_at": now_iso()},
        {"type": "silent_client", "severity": "medium",
         "message": "Cedar Plumbing has been quiet for 9 days", "created_at": now_iso()},
    ]}


@app.get("/api/admin/clients")
async def clients(request: Request):
    auth(request)
    # Leaks a plaintext key, exactly as the real endpoint does.
    return {"clients": [
        {"id": "c1", "business_name": "Cedar Plumbing",
         "api_key": "twsagent_live_SHOULD_BE_REDACTED", "role": "client"},
    ]}


@app.get("/api/admin/system-health")
async def health(request: Request):
    auth(request)
    return {"ok": True, "db": "up"}


@app.get("/api/admin/integrations")
async def integrations(request: Request):
    auth(request)
    return {"retell": True, "twilio": False, "resend": True, "meta": False}


@app.get("/api/admin/leads/stats")
async def leadstats(request: Request, days: int = 7):
    auth(request)
    return {"since": "2026-08-23", "total": {"calls": 120, "appointment_booked": 9}}


@app.get("/api/leaderboard")
async def leaderboard(request: Request):
    auth(request)
    return {"week": {"Ada": 4, "Ben": 2}, "month": {"Ada": 14, "Ben": 9}}


@app.get("/api/admin/bookings/stats")
async def bookings(request: Request):
    auth(request)
    return {"booked": 21, "completed": 14, "converted": 5, "no_show": 2}


@app.get("/api/admin/ai-usage")
async def aiusage(request: Request):
    auth(request)
    return {"spend_usd": 12.4, "budget_usd": 40}


@app.get("/api/admin/lead-opportunities")
async def opportunities(request: Request):
    auth(request)
    return {"opportunities": [{"id": "o1", "name": "Delta Towing"}]}


@app.get("/api/admin/deals/awaiting-setup")
async def deals(request: Request):
    auth(request)
    return {"deals": [{"id": "d1", "business_name": "Echo Electric", "paid": True}]}


@app.get("/api/admin/prospects")
async def prospects(request: Request):
    auth(request)
    return {"prospects": state.prospects}


# ---- chat ----------------------------------------------------------------

class Body(BaseModel):
    body: str


@app.post("/api/chat/channels/{channel}/messages")
async def post_channel(channel: str, payload: Body, request: Request):
    auth(request)
    if channel not in ("general", "setters", "wins", "questions"):
        raise HTTPException(status_code=404, detail="No such channel")
    msg = {"id": str(uuid.uuid4()), "sender_id": "u-atlas", "sender_name": "Atlas",
           "is_ai": False, "body": payload.body, "created_at": now_iso(),
           "channel": channel}
    state.messages.append(msg)

    # The employee reply: produced in the background, always into #general,
    # with no reference back to the question.
    import re
    m = re.search(r"(?<![\w.@+-])@(viktor|nadia|iris|sol)\b", payload.body, re.I)
    if m:
        handle = m.group(1).lower()

        async def reply():
            await asyncio.sleep(0.15)
            state.messages.append({
                "id": str(uuid.uuid4()), "sender_id": f"ai-{handle}",
                "sender_name": handle.title(), "is_ai": True,
                "body": f"{handle.title()} here: 9 demos booked this week, 2 no-shows.",
                "created_at": now_iso(), "channel": "general"})
        asyncio.create_task(reply())
    return msg


@app.get("/api/chat/channels/{channel}/messages")
async def get_channel(channel: str, request: Request, limit: int = 30):
    auth(request)
    rows = [m for m in state.messages if m["channel"] == channel]
    return {"messages": rows[-limit:]}


@app.get("/api/chat/members")
async def members(request: Request):
    auth(request)
    return {"members": [
        {"id": "u-ada", "name": "Ada", "role": "setter", "is_ai": False},
        {"id": "ai-viktor", "name": "Viktor", "role": "assistant", "is_ai": True},
    ]}


@app.post("/api/chat/dm/{other_id}/messages")
async def post_dm(other_id: str, payload: Body, request: Request):
    auth(request)
    state.dms.append({"to": other_id, "body": payload.body})
    return {"ok": True}


@app.post("/api/admin/announcements")
async def announce(request: Request):
    auth(request)
    body = await request.json()
    state.announcements.append(body)
    return {"ok": True, "id": str(uuid.uuid4())}


@app.get("/api/admin/ai-employees")
async def ai_employees(request: Request):
    auth(request)
    return {"employees": [{"handle": "viktor", "name": "Viktor",
                           "last_at": now_iso(), "last_said": "All quiet."}],
            "autonomy": state.autonomy, "scheduler_on": False}


# ---- growth --------------------------------------------------------------

@app.post("/api/admin/prospects/scan")
async def scan(request: Request):
    auth(request)
    body = await request.json()
    if not body.get("trade") or not body.get("location"):
        raise HTTPException(status_code=400, detail="Both a trade and a location are required")
    new = {"id": f"p{len(state.prospects)+1}", "name": f"{body['trade']} Co",
           "phone": "+15551119999", "website": None, "score": 80,
           "status": "new", "touches": []}
    state.prospects.append(new)
    return {"found": 1, "prospects": [new], "source": "google_places"}


@app.post("/api/admin/cold-calling/queue-batch")
async def queue_batch(request: Request):
    auth(request)
    body = await request.json()
    limit = int(body.get("limit") or 30)
    callable_ = [p for p in state.prospects if p.get("phone")][:limit]
    state.staged = callable_
    return {"ok": True, "staged": len(callable_), "prospects": callable_,
            "refused": [{"name": "No-phone Co", "why": "no phone number"}]}


@app.post("/api/admin/cold-calling/start-batch")
async def start_batch(request: Request):
    auth(request)
    if not state.staged:
        raise HTTPException(status_code=400, detail="Nothing is staged. Queue a batch first.")
    state.released += len(state.staged)
    n, state.staged = len(state.staged), []
    return {"ok": True, "approved": n, "note": "Calling has started."}


@app.post("/api/admin/cold-calling/stop")
async def stop_calling(request: Request):
    auth(request)
    n, state.staged = len(state.staged), []
    state.stopped += 1
    return {"ok": True, "unapproved": n,
            "note": "Stopped. Nothing else will dial."}


@app.post("/api/admin/prospects/{prospect_id}/touch")
async def record_touch(prospect_id: str, request: Request):
    auth(request)
    p = next((x for x in state.prospects if x["id"] == prospect_id), None)
    if not p:
        raise HTTPException(status_code=404, detail="Prospect not found")
    body = await request.json()
    p["touches"].append(body)
    # The real app advances the prospect out of the "not yet contacted" lists.
    p["status"] = "contacted"
    return {"ok": True, "touches": len(p["touches"])}


@app.get("/api/admin/cold-calling/preflight")
async def preflight(request: Request):
    auth(request)
    return {"agent_configured": True, "from_number": "+15550000000",
            "callable": len(state.prospects), "daily_cap": 30, "used_today": 0}


@app.put("/api/admin/cold-calling/autonomy")
async def set_autonomy(request: Request):
    auth(request)
    state.autonomy = await request.json()
    return state.autonomy


@app.get("/api/dnc/check")
async def dnc(request: Request, phone: str = "", email: str = ""):
    auth(request)
    return {"listed": phone == "+15551110001", "since": None}


# ---- client-scoped analysis ----------------------------------------------
# These take client_id as a QUERY parameter, not in the body, and 400 for a
# superadmin who omits it. That asymmetry is easy to get wrong from a client
# library, so it is reproduced exactly.

@app.post("/api/website/analyze")
async def analyze_website(request: Request, client_id: str = ""):
    auth(request)
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id required")
    state.analyses.append(("website", client_id))
    return {"client_id": client_id, "findings": ["no click-to-call on mobile"]}


@app.post("/api/competitors")
async def add_competitor(request: Request, client_id: str = ""):
    auth(request)
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id required")
    body = await request.json()
    state.competitors.append({"client_id": client_id, **body})
    return {"ok": True, "name": body.get("name")}


@app.post("/api/messages")
async def send_client_message(request: Request):
    auth(request)
    body = await request.json()
    if not body.get("client_id"):
        raise HTTPException(status_code=400, detail="client_id required")
    state.client_messages.append(body)
    return {"id": str(uuid.uuid4()), **body}


# ---- tenant scoping + internal jobs --------------------------------------

@app.get("/api/dashboard/stats")
async def dashboard(request: Request, client_id: str = ""):
    auth(request)
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id required")
    return {"client_id": client_id, "calls": 12}


@app.post("/api/internal/{path:path}")
async def internal(path: str, x_cron_secret: str = Header(None)):
    if not x_cron_secret or x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="bad cron secret")
    state.jobs_run.append(path)
    return {"ok": True, "job": path, "drained": 3}
