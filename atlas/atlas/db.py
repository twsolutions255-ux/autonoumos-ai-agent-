"""Atlas's own storage, inside the app's MongoDB.

Atlas shares the TWS database but never the TWS collections: everything it
owns is prefixed (`atlas_*`). That gives one deploy, one connection string and
one backup, while making it trivially obvious in the shell which documents
belong to the agent and which belong to the app it operates.

Nothing here writes to an app collection. Changes to the business go through
the HTTP API like any other client, so the app's own validation, logging and
webhooks all still fire. An agent that reached into the database directly
would silently bypass every rule the app enforces.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger("atlas.db")

#: Collections Atlas owns, with the reason each exists. Kept as a table
#: because "what does the agent remember, and for how long?" is a question
#: people ask about autonomous systems and deserves one honest answer.
COLLECTIONS = {
    "cycles": "One document per autonomous cycle: what it saw, decided, did, and what came of it.",
    "actions": "Every tool call Atlas attempted, allowed or blocked, with arguments and result.",
    "memory": "Durable lessons and facts Atlas has learned, retrieved into later cycles.",
    "plan": "The current strategy and its objectives. Small, mutable, versioned on change.",
    "approvals": "Actions held for the owner, with everything needed to decide.",
    "briefs": "Morning plans and evening summaries, as delivered.",
    "metrics": "KPI snapshots over time, so Atlas can see its own effect.",
    "chat": "The owner's conversation with Atlas.",
    "counters": "Rate-limit events, persisted so a restart cannot reset a daily cap.",
}


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso() -> str:
    return now().isoformat()


class Store:
    """Thin, typed-enough wrapper over motor. Created once at startup."""

    def __init__(self, mongo_url: str, db_name: str, prefix: str = "atlas_"):
        if not mongo_url:
            raise ValueError("Store needs a mongo_url")
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(mongo_url, tz_aware=True,
                                          serverSelectionTimeoutMS=8000)
        self._db = self._client[db_name]
        self._prefix = prefix

    def __getitem__(self, name: str):
        if name not in COLLECTIONS:
            raise KeyError(
                f"{name!r} is not an Atlas collection. Add it to COLLECTIONS with a "
                f"reason, so the set of things the agent remembers stays documented.")
        return self._db[f"{self._prefix}{name}"]

    @property
    def raw(self):
        """The database handle, for read-only inspection of app collections.

        Deliberately not used for writes anywhere in Atlas — see the module
        docstring. Reads are occasionally justified for analytics the API does
        not expose, and those should be obvious at the call site.
        """
        return self._db

    async def ping(self) -> bool:
        await self._client.admin.command("ping")
        return True

    async def ensure_indexes(self) -> list[str]:
        """Indexes Atlas needs. Idempotent; safe on every boot."""
        made = []
        specs = [
            ("cycles", [("started_at", -1)], {}),
            ("cycles", [("status", 1), ("started_at", -1)], {}),
            ("actions", [("at", -1)], {}),
            ("actions", [("cycle_id", 1), ("at", 1)], {}),
            ("actions", [("tool", 1), ("at", -1)], {}),
            # Rate-limit hydration reads by bucket within a window.
            ("counters", [("bucket", 1), ("at", -1)], {}),
            ("memory", [("kind", 1), ("importance", -1)], {}),
            ("memory", [("created_at", -1)], {}),
            ("approvals", [("status", 1), ("created_at", -1)], {}),
            ("briefs", [("kind", 1), ("created_at", -1)], {}),
            ("metrics", [("at", -1)], {}),
            ("chat", [("created_at", -1)], {}),
            ("plan", [("version", -1)], {}),
        ]
        for coll, keys, opts in specs:
            try:
                name = await self[coll].create_index(keys, **opts)
                made.append(f"{self._prefix}{coll}.{name}")
            except Exception as e:
                log.warning("atlas: index on %s%s failed: %s", self._prefix, coll, e)

        # Free-text recall over memories. Mongo allows exactly one text index
        # per collection, so this is created separately and tolerantly: an
        # existing index with different weights raises rather than replacing.
        try:
            name = await self["memory"].create_index(
                [("title", "text"), ("body", "text"), ("tags", "text")],
                name="atlas_memory_text")
            made.append(f"{self._prefix}memory.{name}")
        except Exception as e:
            log.info("atlas: text index on memory not created (%s)", e)
        return made

    async def close(self) -> None:
        self._client.close()
