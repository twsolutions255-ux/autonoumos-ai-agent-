"""What Atlas remembers, and how it gets it back.

The existing AI Manager in the TWS app keeps the last twenty chat messages and
nothing else. That is fine for a chatbot and useless for an agent that is
supposed to run a business over months: it cannot tell you what it tried in
March, why that failed, or which of its own ideas has ever actually worked.

Memory here is three separable things, because they decay at different rates:

  EPISODIC   what happened — cycles and actions, kept whole, queried by time.
  SEMANTIC   what was learned — durable statements with an importance and a
             confidence, retrieved by relevance into future cycles.
  WORKING    what matters right now — the plan and the live KPI snapshot,
             rebuilt every cycle and never trusted from cache.

Retrieval is deliberately not a vector database. Atlas runs against MongoDB,
which the app already pays for, and a text index plus explicit importance and
recency weighting is both cheap and inspectable — you can read a memory row
and understand why it surfaced. A vector backend can be swapped in behind
`recall()` later without touching a caller.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import Any, Optional

from ..db import now, iso

log = logging.getLogger("atlas.memory")


class Kind:
    """Why a memory exists. Governs retrieval weighting and expiry."""
    #: Something Atlas concluded from evidence. The valuable kind.
    LESSON = "lesson"
    #: A durable fact about the business, market or a client.
    FACT = "fact"
    #: An explicit instruction from the owner. Never expires, always retrieved.
    DIRECTIVE = "directive"
    #: Something Atlas tried, and what came of it.
    OUTCOME = "outcome"
    #: A standing risk or constraint to respect.
    CONSTRAINT = "constraint"


#: Kinds that are always loaded into context regardless of relevance scoring.
#: An instruction from the owner that only surfaced when a keyword matched
#: would be an instruction the agent forgets at exactly the wrong moment.
ALWAYS_RECALL = (Kind.DIRECTIVE, Kind.CONSTRAINT)


@dataclass
class Memory:
    id: str
    kind: str
    title: str
    body: str
    tags: list
    #: 1-5. How much this should influence future decisions.
    importance: int
    #: 0-1. How sure Atlas is. A lesson from one data point is not a law.
    confidence: float
    source: str
    created_at: str
    #: Bumped whenever the memory is retrieved and acted on; lets genuinely
    #: useful lessons outrank merely recent ones over time.
    uses: int = 0
    superseded_by: Optional[str] = None

    def to_doc(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        conf = "certain" if self.confidence >= 0.85 else (
            "likely" if self.confidence >= 0.6 else "tentative")
        return f"[{self.kind}/{conf}] {self.title} — {self.body}"


class MemoryStore:
    def __init__(self, store):
        self.store = store

    # ---------- writing ----------

    async def remember(self, *, kind: str, title: str, body: str,
                       tags: Optional[list] = None, importance: int = 3,
                       confidence: float = 0.6, source: str = "atlas",
                       supersedes: Optional[str] = None) -> Memory:
        """Record something worth keeping.

        `supersedes` matters more than it looks: an agent that learns
        "cold email at 9am beats 2pm" and later learns the opposite must not
        end up believing both. Superseding marks the old one dead rather than
        deleting it, so the reasoning history stays auditable.
        """
        m = Memory(
            id=str(uuid.uuid4()), kind=kind, title=title.strip()[:300],
            body=body.strip()[:4000], tags=[t.lower() for t in (tags or [])],
            importance=max(1, min(5, int(importance))),
            confidence=max(0.0, min(1.0, float(confidence))),
            source=source, created_at=iso(),
        )
        await self.store["memory"].insert_one(m.to_doc())
        if supersedes:
            await self.store["memory"].update_one(
                {"id": supersedes}, {"$set": {"superseded_by": m.id}})
        log.info("atlas: remembered [%s] %s", kind, m.title)
        return m

    async def record_outcome(self, action: str, result: str, worked: bool,
                             *, evidence: str = "") -> Memory:
        """The feedback half of the loop. Without this Atlas cannot improve."""
        return await self.remember(
            kind=Kind.OUTCOME,
            title=f"{'Worked' if worked else 'Did not work'}: {action}",
            body=f"{result}{(' Evidence: ' + evidence) if evidence else ''}",
            tags=["outcome", "worked" if worked else "failed"],
            importance=4 if not worked else 3,
            confidence=0.9 if evidence else 0.5,
        )

    # ---------- reading ----------

    async def recall(self, query: str = "", *, limit: int = 20,
                     kinds: Optional[list] = None) -> list[Memory]:
        """The memories that should shape the next decision.

        Directives and constraints always come back. Everything else competes
        on a blend of text relevance, stated importance, how often it has
        proved useful, and how recent it is.
        """
        out: list[Memory] = []
        seen: set[str] = set()

        pinned = await self.store["memory"].find(
            {"kind": {"$in": list(ALWAYS_RECALL)}, "superseded_by": None},
            {"_id": 0}).sort("importance", -1).to_list(30)
        for d in pinned:
            out.append(Memory(**d))
            seen.add(d["id"])

        remaining = max(0, limit - len(out))
        if remaining:
            q: dict[str, Any] = {"superseded_by": None, "id": {"$nin": list(seen)}}
            if kinds:
                q["kind"] = {"$in": kinds}
            if query.strip():
                # Text search first; fall back to plain recency when the text
                # index is missing or the query matches nothing. A recall that
                # returns nothing because of an index problem is worse than a
                # recall that returns something merely recent.
                try:
                    cur = self.store["memory"].find(
                        {**q, "$text": {"$search": query}},
                        {"_id": 0, "score": {"$meta": "textScore"}},
                    ).sort([("score", {"$meta": "textScore"})]).limit(remaining * 3)
                    cands = await cur.to_list(remaining * 3)
                except Exception:
                    cands = []
            else:
                cands = []
            if not cands:
                cands = await self.store["memory"].find(q, {"_id": 0}) \
                    .sort("created_at", -1).to_list(remaining * 3)

            scored = sorted(cands, key=lambda d: self._score(d, query), reverse=True)
            for d in scored[:remaining]:
                d.pop("score", None)
                out.append(Memory(**d))
        return out

    def _score(self, d: dict, query: str) -> float:
        """Relevance, importance, proven usefulness, recency — in that order.

        Recency decays on a 30-day half-life: a lesson from last week should
        outrank one from last quarter, but not erase it.
        """
        s = float(d.get("score") or 0.0) * 2.0
        s += float(d.get("importance", 3))
        s += math.log1p(float(d.get("uses", 0)))
        try:
            age_days = (now() - _parse(d.get("created_at"))).days
        except Exception:
            age_days = 30
        s += 2.0 * math.exp(-max(0, age_days) / 30.0)
        s *= 0.5 + float(d.get("confidence", 0.5))
        return s

    async def mark_used(self, ids: list) -> None:
        if ids:
            await self.store["memory"].update_many(
                {"id": {"$in": list(ids)}}, {"$inc": {"uses": 1}})

    async def render_for_prompt(self, query: str = "", limit: int = 20) -> str:
        """The block of remembered context injected into a reasoning turn."""
        mems = await self.recall(query, limit=limit)
        if not mems:
            return "(Atlas has no stored memory yet — this is an early cycle.)"
        await self.mark_used([m.id for m in mems])
        lines = []
        for group, label in ((Kind.DIRECTIVE, "STANDING INSTRUCTIONS FROM THE OWNER"),
                             (Kind.CONSTRAINT, "CONSTRAINTS"),
                             (Kind.LESSON, "WHAT ATLAS HAS LEARNED"),
                             (Kind.FACT, "FACTS"),
                             (Kind.OUTCOME, "RECENT OUTCOMES")):
            chunk = [m for m in mems if m.kind == group]
            if chunk:
                lines.append(f"\n{label}:")
                lines += [f"  - {m.render()}" for m in chunk]
        return "\n".join(lines).strip()

    # ---------- hygiene ----------

    async def consolidate(self, keep_outcomes_days: int = 45) -> dict:
        """Stop low-value memory from crowding out high-value memory.

        Outcomes are the fast-growing kind — one per action — and past a point
        the individual rows stop being worth their space; what matters by then
        is the lesson drawn from them, which is stored separately. Lessons,
        facts, directives and constraints are never pruned here.
        """
        cutoff = (now() - timedelta(days=keep_outcomes_days)).isoformat()
        res = await self.store["memory"].delete_many(
            {"kind": Kind.OUTCOME, "created_at": {"$lt": cutoff},
             "importance": {"$lt": 4}})
        return {"pruned_outcomes": res.deleted_count}


def _parse(v: Any):
    from datetime import datetime, timezone
    if hasattr(v, "tzinfo"):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
