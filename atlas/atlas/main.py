"""The Atlas service: a control API around one autonomous Runtime.

The API exists so a person can watch, steer and stop the agent. Everything it
exposes is either an observation (what did you do, what do you believe, what
is it costing) or a control (stop, change authority, approve, talk to it).

The scheduler runs inside this process as a background task. That keeps the
deployment to a single service, which matters more than architectural purity
here: a second worker process is a second thing to configure, pay for, and
forget to restart.

Every route requires the console key. There is no unauthenticated surface —
this API can start cold calls.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import AUTONOMY_LEVELS, settings
from .brain.loop import Runtime
from .db import iso
from .tools.registry import registry

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("atlas")

runtime = Runtime(settings)
_scheduler_task: Optional[asyncio.Task] = None
_boot: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler_task, _boot
    try:
        _boot = await runtime.start()
        log.info("atlas: booted — %s tools, autonomy=%s, sandbox=%s",
                 _boot.get("tools"), settings.autonomy, settings.sandbox)
        for check in _boot.get("readiness", {}).get("checks", []):
            if not check["ok"]:
                log.warning("atlas: %s is OFF — %s (set %s)",
                            check["name"], check["off_means"], check["set"])
    except Exception as e:
        # Boot failure must not take the service down: the console is how you
        # find out WHY it failed, so it has to stay reachable.
        log.exception("atlas: boot failed")
        _boot = {"error": f"{type(e).__name__}: {e}"}

    if settings.tick_seconds > 0 and runtime.engine is not None:
        _scheduler_task = asyncio.create_task(runtime.serve_forever())
    else:
        log.warning("atlas: scheduler not started (no model configured, or tick disabled)")

    yield

    runtime.stop()
    if _scheduler_task:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except (asyncio.CancelledError, Exception):
            pass
    await runtime.close()


app = FastAPI(title="Atlas — TWS autonomous scaling agent", version="1.0",
              lifespan=lifespan)

_origins = [o.strip() for o in (settings.cors_origins or "").split(",")
            if o.strip() and o.strip() != "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_key(x_atlas_key: str = Header(None)) -> bool:
    """The only gate. Deliberately fails closed when no key is configured.

    An unauthenticated Atlas would let anyone raise its autonomy and release a
    cold-call batch, so "no key set" must mean "serves nobody", never "serves
    everybody".
    """
    if not settings.console_api_key:
        raise HTTPException(
            status_code=503,
            detail="ATLAS_CONSOLE_API_KEY is not set, so the control API refuses to "
                   "serve. This endpoint can start outbound calls; it is never open.")
    import hmac
    if not x_atlas_key or not hmac.compare_digest(x_atlas_key, settings.console_api_key):
        raise HTTPException(status_code=401, detail="Bad or missing X-Atlas-Key.")
    return True


# --------------------------------------------------------------------- models

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class CycleIn(BaseModel):
    kind: str = "work"
    note: str = ""


class AutonomyIn(BaseModel):
    level: str


class SwitchIn(BaseModel):
    on: bool


class ApprovalIn(BaseModel):
    approve: bool
    note: str = ""


# ---------------------------------------------------------------------- routes

@app.get("/health")
async def health() -> dict:
    """Open on purpose: a load balancer needs it, and it leaks nothing."""
    return {
        "ok": True,
        "service": "atlas",
        "ready": settings.readiness()["ready"],
        "autonomy": runtime.policy.autonomy,
        "sandbox": runtime.policy.sandbox,
        "kill_switch": runtime.policy.kill_switch,
        "scheduler": bool(_scheduler_task and not _scheduler_task.done()),
    }


@app.get("/status", dependencies=[Depends(require_key)])
async def status() -> dict:
    """Everything a person needs to judge whether Atlas is doing its job."""
    pending = 0
    if runtime.store:
        pending = await runtime.store["approvals"].count_documents({"status": "pending"})
    return {
        "boot": _boot,
        "readiness": settings.readiness(),
        "policy": runtime.policy.snapshot(),
        "identity": runtime.identity,
        "tools": {"count": len(registry), "names": registry.names()},
        "last_cycle": runtime.last_cycle,
        "pending_approvals": pending,
        "spend_today_usd": round(runtime._spend_today, 4),
        "daily_budget_usd": settings.daily_llm_budget_usd,
        "scheduler_running": bool(_scheduler_task and not _scheduler_task.done()),
        "at": iso(),
    }


@app.post("/chat", dependencies=[Depends(require_key)])
async def chat(body: ChatIn) -> dict:
    """Talk to Atlas. Same brain and same guardrails as an autonomous cycle."""
    return await runtime.chat(body.message)


@app.post("/cycle", dependencies=[Depends(require_key)])
async def cycle(body: CycleIn) -> dict:
    """Run a cycle now instead of waiting for the timer."""
    if body.kind not in ("morning", "work", "evening"):
        raise HTTPException(status_code=400,
                            detail="kind must be morning, work or evening")
    return await runtime.run_cycle(body.kind, note=body.note)


@app.get("/cycles", dependencies=[Depends(require_key)])
async def cycles(limit: int = Query(20, ge=1, le=200)) -> dict:
    if not runtime.store:
        return {"cycles": [], "note": "No database configured; nothing is remembered."}
    rows = await runtime.store["cycles"].find({}, {"_id": 0}) \
        .sort("started_at", -1).to_list(limit)
    return {"cycles": rows}


@app.get("/actions", dependencies=[Depends(require_key)])
async def actions(limit: int = Query(50, ge=1, le=500),
                  cycle_id: Optional[str] = None) -> dict:
    """The audit trail: every tool call, including the ones policy refused."""
    if not runtime.store:
        return {"actions": []}
    q = {"cycle_id": cycle_id} if cycle_id else {}
    rows = await runtime.store["actions"].find(q, {"_id": 0}) \
        .sort("at", -1).to_list(limit)
    return {"actions": rows}


@app.get("/plan", dependencies=[Depends(require_key)])
async def plan() -> dict:
    if not runtime.store:
        return {"plan": None}
    doc = await runtime.store["plan"].find_one({}, {"_id": 0}, sort=[("version", -1)])
    return {"plan": doc}


@app.get("/briefs", dependencies=[Depends(require_key)])
async def briefs(limit: int = Query(20, ge=1, le=100)) -> dict:
    if not runtime.store:
        return {"briefs": []}
    rows = await runtime.store["briefs"].find({}, {"_id": 0}) \
        .sort("created_at", -1).to_list(limit)
    return {"briefs": rows}


@app.get("/memory", dependencies=[Depends(require_key)])
async def memory(q: str = "", limit: int = Query(40, ge=1, le=200)) -> dict:
    """What Atlas believes. Worth reading before trusting what it does."""
    if not runtime.memory:
        return {"memories": []}
    mems = await runtime.memory.recall(q, limit=limit)
    return {"memories": [m.to_doc() for m in mems]}


@app.get("/metrics", dependencies=[Depends(require_key)])
async def metrics(limit: int = Query(30, ge=1, le=180)) -> dict:
    if not runtime.store:
        return {"metrics": []}
    rows = await runtime.store["metrics"].find({}, {"_id": 0}) \
        .sort("at", -1).to_list(limit)
    rows.reverse()
    return {"metrics": rows}


@app.get("/approvals", dependencies=[Depends(require_key)])
async def approvals(status_filter: str = Query("pending", alias="status")) -> dict:
    if not runtime.store:
        return {"approvals": []}
    q = {} if status_filter == "all" else {"status": status_filter}
    rows = await runtime.store["approvals"].find(q, {"_id": 0}) \
        .sort("created_at", -1).to_list(100)
    return {"approvals": rows}


@app.post("/approvals/{approval_id}", dependencies=[Depends(require_key)])
async def decide_approval(approval_id: str, body: ApprovalIn) -> dict:
    """Approve or reject a held action — and, when approved, actually run it.

    Approving has to execute the call, not merely mark a row: an approval queue
    that records consent without doing anything is the most misleading thing
    this service could contain.
    """
    if not runtime.store:
        raise HTTPException(status_code=503, detail="No database configured.")
    doc = await runtime.store["approvals"].find_one({"id": approval_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="No such approval.")
    if doc.get("status") != "pending":
        return {"already": doc.get("status"), "approval": doc}

    if not body.approve:
        await runtime.store["approvals"].update_one(
            {"id": approval_id},
            {"$set": {"status": "rejected", "decided_at": iso(),
                      "decided_by": "owner", "note": body.note}})
        return {"status": "rejected", "id": approval_id}

    ctx = runtime._context(doc.get("cycle_id") or "", interactive=True)
    try:
        result = await registry.dispatch(ctx, doc["tool"], doc.get("args") or {},
                                         approved=True)
        ok, detail = True, result
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"

    await runtime.store["approvals"].update_one(
        {"id": approval_id},
        {"$set": {"status": "approved" if ok else "failed", "decided_at": iso(),
                  "decided_by": "owner", "note": body.note,
                  "result": str(detail)[:2000]}})
    return {"status": "approved" if ok else "failed", "id": approval_id,
            "result": detail}


@app.post("/control/autonomy", dependencies=[Depends(require_key)])
async def set_autonomy(body: AutonomyIn) -> dict:
    """Change how much Atlas is allowed to do, without a redeploy."""
    if body.level not in AUTONOMY_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"level must be one of {', '.join(AUTONOMY_LEVELS)}")
    runtime.policy.set_autonomy(body.level)
    log.warning("atlas: autonomy changed to %s", body.level)
    return runtime.policy.snapshot()


@app.post("/control/sandbox", dependencies=[Depends(require_key)])
async def set_sandbox(body: SwitchIn) -> dict:
    runtime.policy.sandbox = bool(body.on)
    log.warning("atlas: sandbox %s", "ON" if body.on else "OFF")
    return runtime.policy.snapshot()


@app.post("/control/stop", dependencies=[Depends(require_key)])
async def stop(body: SwitchIn) -> dict:
    """The brake. Takes effect inside a running cycle, not just before one."""
    runtime.policy.kill_switch = bool(body.on)
    log.warning("atlas: kill switch %s", "ON" if body.on else "OFF")
    return runtime.policy.snapshot()


@app.get("/tools", dependencies=[Depends(require_key)])
async def tools() -> dict:
    """Every capability, its risk class, and whether it is currently reachable."""
    out = []
    for name in registry.names():
        t = registry.get(name)
        decision = runtime.policy.check(name, t.policy, {})
        out.append({
            "name": name, "group": t.group, "risk": t.policy.risk.value,
            "requires_autonomy": t.policy.required_level(),
            "rate_bucket": t.policy.rate_bucket,
            "available_now": decision.allowed,
            "why_not": None if decision.allowed else decision.reason,
        })
    return {"tools": out, "count": len(out)}
