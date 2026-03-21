"""
backend/chat_history_db.py
===========================
MongoDB chat history store.

Each document in the 'chat_history' collection represents one pipeline job
belonging to a user:

{
  "_id":       ObjectId,
  "user_id":   int,                  # FK → SQLite users.id
  "job_id":    str,                  # RAD-ML pipeline job id
  "prompt":    str,
  "status":    "running"|"done"|"error",
  "logs":      [{ step, message, ts }],
  "result":    { deploy_url, dataset, model, ... } | {},
  "error":     str,
  "created_at": datetime,
  "updated_at": datetime,
}

Falls back to in-memory dict store when MongoDB is unreachable,
so the app always works even without MongoDB configured.
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try to import pymongo; fall back to in-memory
try:
    from pymongo import MongoClient, DESCENDING
    from pymongo.errors import ConnectionFailure
    _MONGO_AVAILABLE = True
except ImportError:
    _MONGO_AVAILABLE = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── In-memory fallback ────────────────────────────────────────────────────────
class _MemoryStore:
    """Simple in-memory fallback when MongoDB is unavailable."""
    def __init__(self):
        self._docs: dict[str, dict] = {}   # job_id → doc

    def upsert_job(self, user_id: int, job_id: str, **fields) -> None:
        doc = self._docs.get(job_id, {
            "user_id": user_id, "job_id": job_id,
            "created_at": _utcnow(),
        })
        doc.update(fields)
        doc["updated_at"] = _utcnow()
        self._docs[job_id] = doc

    def get_history(self, user_id: int, limit: int = 50) -> list[dict]:
        docs = [d for d in self._docs.values() if d["user_id"] == user_id]
        docs.sort(key=lambda d: d.get("created_at", datetime.min), reverse=True)
        return [self._clean(d) for d in docs[:limit]]

    def delete_job(self, user_id: int, job_id: str) -> bool:
        doc = self._docs.get(job_id)
        if doc and doc["user_id"] == user_id:
            del self._docs[job_id]
            return True
        return False

    def get_job(self, user_id: int, job_id: str) -> dict | None:
        doc = self._docs.get(job_id)
        if doc and doc["user_id"] == user_id:
            return self._clean(doc)
        return None

    def delete_all(self, user_id: int) -> int:
        to_del = [jid for jid, d in self._docs.items()
                  if d["user_id"] == user_id]
        for jid in to_del:
            del self._docs[jid]
        return len(to_del)

    @staticmethod
    def _clean(doc: dict) -> dict:
        d = {k: v for k, v in doc.items() if k != "_id"}
        # Serialize datetimes
        for k in ("created_at", "updated_at"):
            if isinstance(d.get(k), datetime):
                d[k] = d[k].isoformat()
        return d


# ── MongoDB store ─────────────────────────────────────────────────────────────
class ChatHistoryDB:
    """MongoDB-backed chat history with automatic in-memory fallback."""

    def __init__(self, config: dict):
        mongo_cfg = config.get("mongodb", {})
        self._uri        = mongo_cfg.get("uri", "mongodb://localhost:27017")
        self._db_name    = mongo_cfg.get("db_name", "radml")
        self._coll_name  = mongo_cfg.get("collection", "chat_history")
        self._coll       = None
        self._fallback   = _MemoryStore()
        self._use_mongo  = False
        self._connect()

    def _connect(self) -> None:
        if not _MONGO_AVAILABLE:
            logger.warning("pymongo not installed — using in-memory history store.")
            return
        try:
            client = MongoClient(self._uri, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
            db = client[self._db_name]
            self._coll = db[self._coll_name]
            # Ensure indexes
            self._coll.create_index([("user_id", 1), ("updated_at", -1)])
            self._coll.create_index("job_id", unique=True)
            self._use_mongo = True
            logger.info("MongoDB connected: %s/%s", self._db_name, self._coll_name)
        except Exception as exc:
            logger.warning(
                "MongoDB unreachable (%s) — using in-memory history store. "
                "Start MongoDB or set mongodb.uri in config.yaml.", exc
            )

    # ── public API ────────────────────────────────────────────────────────────
    def upsert_job(self, user_id: int, job_id: str, **fields) -> None:
        """Create or update a job document."""
        if not self._use_mongo:
            self._fallback.upsert_job(user_id, job_id, **fields)
            return
        self._coll.update_one(
            {"job_id": job_id},
            {"$set":    {**fields, "user_id": user_id, "updated_at": _utcnow()},
             "$setOnInsert": {"job_id": job_id, "created_at": _utcnow()}},
            upsert=True,
        )

    def get_history(self, user_id: int, limit: int = 50) -> list[dict]:
        """Return most recent jobs for this user (newest first)."""
        if not self._use_mongo:
            return self._fallback.get_history(user_id, limit)
        cursor = (
            self._coll
            .find({"user_id": user_id},
                  {"_id": 0, "logs": 0})   # exclude heavy logs from list view
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )
        return list(cursor)

    def get_job(self, user_id: int, job_id: str) -> dict | None:
        """Return the full document for a single job."""
        if not self._use_mongo:
            return self._fallback.get_job(user_id, job_id)
        doc = self._coll.find_one(
            {"job_id": job_id, "user_id": user_id}, {"_id": 0}
        )
        return doc

    def delete_job(self, user_id: int, job_id: str) -> bool:
        """Delete one job. Returns True if a document was deleted."""
        if not self._use_mongo:
            return self._fallback.delete_job(user_id, job_id)
        result = self._coll.delete_one(
            {"job_id": job_id, "user_id": user_id}
        )
        return result.deleted_count > 0

    def delete_all(self, user_id: int) -> int:
        """Delete all history for a user. Returns count deleted."""
        if not self._use_mongo:
            return self._fallback.delete_all(user_id)
        result = self._coll.delete_many({"user_id": user_id})
        return result.deleted_count
