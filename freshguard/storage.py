"""Persistence layer (SQLite): query history (FR17), user feedback (FR18), admin config
(FR19) and knowledge-base documents (FR20)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from freshguard.config import DB_PATH, DEFAULTS, Settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT,
                    confidence REAL,
                    freshness REAL,
                    retrieved INTEGER,
                    sources TEXT,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    history_id INTEGER NOT NULL,
                    rating INTEGER,
                    report_incorrect INTEGER,
                    comment TEXT
                );
                CREATE TABLE IF NOT EXISTS kb_docs (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT, url TEXT, date TEXT, text TEXT
                );
                """
            )

    # -- config (FR19) ---------------------------------------------------------
    def get_settings(self) -> Settings:
        s = Settings(**DEFAULTS.to_dict())
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT key, value FROM config").fetchall()
        for r in rows:
            if hasattr(s, r["key"]):
                cur = getattr(s, r["key"])
                try:
                    val = json.loads(r["value"])
                except Exception:
                    val = r["value"]
                setattr(s, r["key"], type(cur)(val) if not isinstance(cur, bool) else bool(val))
        return s

    def set_config(self, updates: dict) -> Settings:
        with self._lock, self._conn:
            for k, v in updates.items():
                if v is None or not hasattr(DEFAULTS, k):
                    continue
                self._conn.execute(
                    "INSERT INTO config(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, json.dumps(v)),
                )
        return self.get_settings()

    # -- history (FR17) --------------------------------------------------------
    def add_history(self, response: dict) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO history(created_at,query,answer,confidence,freshness,retrieved,sources,payload) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (_now(), response["query"], response["answer"], response["confidence"],
                 response["freshness_score"], int(response["retrieved"]),
                 json.dumps(response["sources"]), json.dumps(response)),
            )
            return int(cur.lastrowid)

    def get_history(self, limit: int = 50) -> list[dict]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id,created_at,query,answer,confidence,freshness,retrieved,sources "
                "FROM history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["retrieved"] = bool(d["retrieved"])
            d["sources"] = json.loads(d["sources"] or "[]")
            out.append(d)
        return out

    def get_history_item(self, history_id: int) -> dict | None:
        with self._lock, self._conn:
            r = self._conn.execute("SELECT payload FROM history WHERE id=?", (history_id,)).fetchone()
        return json.loads(r["payload"]) if r else None

    # -- feedback (FR18) -------------------------------------------------------
    def add_feedback(self, history_id: int, rating: int | None,
                     report_incorrect: bool, comment: str | None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO feedback(created_at,history_id,rating,report_incorrect,comment) "
                "VALUES(?,?,?,?,?)",
                (_now(), history_id, rating, int(report_incorrect), comment),
            )
            return int(cur.lastrowid)

    def history_exists(self, history_id: int) -> bool:
        with self._lock, self._conn:
            r = self._conn.execute("SELECT 1 FROM history WHERE id=?", (history_id,)).fetchone()
        return r is not None

    def feedback_summary(self) -> dict:
        with self._lock, self._conn:
            total = self._conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
            avg = self._conn.execute("SELECT AVG(rating) a FROM feedback WHERE rating IS NOT NULL").fetchone()["a"]
            reports = self._conn.execute("SELECT COUNT(*) c FROM feedback WHERE report_incorrect=1").fetchone()["c"]
        return {"total": total, "avg_rating": round(avg, 2) if avg is not None else None,
                "incorrect_reports": reports}

    # -- knowledge base (FR20) -------------------------------------------------
    def kb_all(self) -> list[dict]:
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT doc_id,title,url,date,text FROM kb_docs").fetchall()
        return [dict(r) for r in rows]

    def kb_upsert(self, doc: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO kb_docs(doc_id,title,url,date,text) VALUES(?,?,?,?,?) "
                "ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title,url=excluded.url,"
                "date=excluded.date,text=excluded.text",
                (doc["doc_id"], doc["title"], doc.get("url", ""), doc.get("date", ""), doc["text"]),
            )

    def kb_delete(self, doc_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM kb_docs WHERE doc_id=?", (doc_id,))

    def close(self) -> None:
        self._conn.close()
