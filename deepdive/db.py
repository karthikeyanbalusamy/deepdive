"""SQLite storage: every run, and a step-by-step log of what happened (the audit trail)."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from deepdive import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL,       -- running | done | failed
    created_at REAL NOT NULL,
    report TEXT,                -- final Markdown report
    state TEXT,                 -- final state as JSON: sub-questions, sources, claims with verdicts
    error TEXT
);
CREATE TABLE IF NOT EXISTS log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    time REAL NOT NULL,
    step TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)  # wait if another researcher is writing at the same moment
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_run(question: str) -> str:
    run_id = uuid.uuid4().hex[:8]
    with connect() as conn:
        conn.execute("INSERT INTO runs (id, question, status, created_at) VALUES (?, ?, 'running', ?)",
                     (run_id, question, time.time()))
    return run_id


def finish_run(run_id: str, state: dict) -> None:
    with connect() as conn:
        conn.execute("UPDATE runs SET status='done', report=?, state=? WHERE id=?",
                     (state["report"], json.dumps(state), run_id))


def fail_run(run_id: str, error: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE runs SET status='failed', error=? WHERE id=?", (error, run_id))


def log(run_id: str, step: str, message: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO log (run_id, time, step, message) VALUES (?, ?, ?, ?)",
                     (run_id, time.time(), step, message))


def get_run(run_id: str) -> dict | None:
    """A run with its final state and full log, or None if it does not exist."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return None
        run = dict(row)
        run["state"] = json.loads(run["state"]) if run["state"] else None
        run["log"] = [dict(r) for r in conn.execute("SELECT time, step, message FROM log WHERE run_id=? ORDER BY id", (run_id,))]
    return run


def list_runs(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT id, question, status, created_at FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
