"""src/db.py — CounsHelper SQLite DB.

3테이블:
- patients(id, alias, gender, age, region, note, created_at)
- sessions(id, patient_id FK, session_date, transcript, created_at)
- analyses(id auto, session_id FK, stage, backend, payload_json, created_at)

설정:
- PRAGMA foreign_keys = ON  (CASCADE 동작)
- PRAGMA journal_mode = WAL  (Streamlit rerun 대비 reader/writer 비차단)
- per-call connection (Streamlit + thread reuse 안전)
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from config import DB_PATH


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    c.execute("PRAGMA journal_mode = WAL")
    try:
        yield c
    finally:
        c.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    alias TEXT NOT NULL,
    gender TEXT,
    age INTEGER,
    region TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    session_date TEXT,
    session_no TEXT,
    scope TEXT,
    topic TEXT,
    transcript TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    backend TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_analyses_session ON analyses(session_id, stage);
"""


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)
        # 기존 DB에 새 컬럼 보충 (이미 있으면 OperationalError skip — 멱등)
        for col in ("session_no", "scope", "topic"):
            try:
                c.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _row_to_dict(row) -> dict:
    return {k: row[k] for k in row.keys()}


# ── patients ───────────────────────────────────────────────────────────────────


def add_patient(alias: str, gender: str, age: int, region: str, note: str = "") -> dict:
    pid = _new_id()
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO patients(id, alias, gender, age, region, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, alias.strip(), gender, int(age), region.strip(), note.strip(), now),
        )
    return get_patient(pid)


def list_patients() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM patients ORDER BY datetime(created_at) DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_patient(patient_id: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    return _row_to_dict(r) if r else None


def delete_patient(patient_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        return cur.rowcount > 0


# ── sessions ───────────────────────────────────────────────────────────────────


def add_session(
    patient_id: str,
    session_date: str,
    transcript: str,
    session_no: Optional[str] = None,
    scope: Optional[str] = None,
    topic: Optional[str] = None,
) -> dict:
    sid = _new_id()
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions(id, patient_id, session_date, session_no, scope, topic, transcript, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, patient_id, session_date, session_no, scope, topic, transcript, now),
        )
    return get_session(sid)


def list_sessions(patient_id: Optional[str] = None) -> list[dict]:
    with _conn() as c:
        if patient_id:
            rows = c.execute(
                "SELECT * FROM sessions WHERE patient_id = ? "
                "ORDER BY date(session_date) DESC, datetime(created_at) DESC",
                (patient_id,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM sessions ORDER BY datetime(created_at) DESC"
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    with _conn() as c:
        r = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_dict(r) if r else None


def delete_session(session_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0


# ── analyses ───────────────────────────────────────────────────────────────────


def add_analysis(session_id: str, stage: str, backend: str, payload: dict) -> int:
    now = _now()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO analyses(session_id, stage, backend, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, stage, backend, json.dumps(payload, ensure_ascii=False), now),
        )
        return cur.lastrowid


def list_analyses(session_id: str, stage: Optional[str] = None) -> list[dict]:
    with _conn() as c:
        if stage:
            rows = c.execute(
                "SELECT * FROM analyses WHERE session_id = ? AND stage = ? "
                "ORDER BY datetime(created_at) DESC, id DESC",
                (session_id, stage),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM analyses WHERE session_id = ? "
                "ORDER BY datetime(created_at) DESC, id DESC",
                (session_id,),
            ).fetchall()
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["payload"] = json.loads(d.pop("payload_json"))
        out.append(d)
    return out


def get_latest_analysis(session_id: str, stage: str) -> Optional[dict]:
    rows = list_analyses(session_id, stage=stage)
    return rows[0] if rows else None


def export_patient(patient_id: str) -> dict:
    p = get_patient(patient_id)
    if not p:
        return {}
    sessions = list_sessions(patient_id)
    for s in sessions:
        s["analyses"] = {
            stage: get_latest_analysis(s["id"], stage)
            for stage in ("classifier", "factors", "summary")
        }
    return {"patient": p, "sessions": sessions, "exported_at": _now()}
