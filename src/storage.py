"""src/storage.py — F5: 내담자/회기 JSON 영속화.

MVP는 단일 사용자(데모 계정) 가정. 파일 잠금 없음. 동시 쓰기는 발생 안 한다고 본다.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import PATIENTS_FILE, SESSIONS_FILE, STORAGE_DIR


def _ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not PATIENTS_FILE.exists():
        PATIENTS_FILE.write_text("[]", encoding="utf-8")
    if not SESSIONS_FILE.exists():
        SESSIONS_FILE.write_text("[]", encoding="utf-8")


def _load(path: Path) -> list[dict]:
    _ensure_storage()
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, items: list[dict]) -> None:
    _ensure_storage()
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 내담자 ────────────────────────────────────────────────────────────────────

def list_patients() -> list[dict]:
    """내담자 전체 목록. 최신 등록순."""
    return sorted(_load(PATIENTS_FILE), key=lambda p: p.get("created_at", ""), reverse=True)


def get_patient(patient_id: str) -> Optional[dict]:
    for p in _load(PATIENTS_FILE):
        if p["id"] == patient_id:
            return p
    return None


def add_patient(alias: str, gender: str, age: int, region: str, note: str = "") -> dict:
    """신규 내담자 등록. alias는 익명 식별자 (실명 금지 — 데모 가드레일)."""
    patient = {
        "id": str(uuid.uuid4())[:8],
        "alias": alias.strip(),
        "gender": gender,
        "age": int(age),
        "region": region.strip(),
        "note": note.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    items = _load(PATIENTS_FILE)
    items.append(patient)
    _save(PATIENTS_FILE, items)
    return patient


def delete_patient(patient_id: str) -> bool:
    items = _load(PATIENTS_FILE)
    new_items = [p for p in items if p["id"] != patient_id]
    if len(new_items) == len(items):
        return False
    _save(PATIENTS_FILE, new_items)
    # 관련 회기도 함께 삭제
    sessions = _load(SESSIONS_FILE)
    _save(SESSIONS_FILE, [s for s in sessions if s.get("patient_id") != patient_id])
    return True


# ── 회기 ──────────────────────────────────────────────────────────────────────

def list_sessions(patient_id: Optional[str] = None) -> list[dict]:
    """회기 목록. patient_id 지정 시 해당 내담자만, 없으면 전체. 최신 회기일순."""
    items = _load(SESSIONS_FILE)
    if patient_id is not None:
        items = [s for s in items if s.get("patient_id") == patient_id]
    return sorted(items, key=lambda s: s.get("session_date", ""), reverse=True)


def get_session(session_id: str) -> Optional[dict]:
    for s in _load(SESSIONS_FILE):
        if s["id"] == session_id:
            return s
    return None


def add_session(
    patient_id: str,
    session_date: str,
    transcript: str,
    classifier_result: Optional[dict] = None,
    factor_result: Optional[dict] = None,
    summary_result: Optional[dict] = None,
) -> dict:
    """회기 등록. 분석 결과(F1 1차/2차, F3)도 함께 저장 가능."""
    session = {
        "id": str(uuid.uuid4())[:8],
        "patient_id": patient_id,
        "session_date": session_date,
        "transcript": transcript,
        "classifier": classifier_result or {},
        "factors": factor_result or {},
        "summary": summary_result or {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    items = _load(SESSIONS_FILE)
    items.append(session)
    _save(SESSIONS_FILE, items)
    return session


def update_session(session_id: str, **fields) -> Optional[dict]:
    """회기 분석 결과 갱신 (재분석 등)."""
    items = _load(SESSIONS_FILE)
    for s in items:
        if s["id"] == session_id:
            s.update(fields)
            _save(SESSIONS_FILE, items)
            return s
    return None


def delete_session(session_id: str) -> bool:
    items = _load(SESSIONS_FILE)
    new_items = [s for s in items if s["id"] != session_id]
    if len(new_items) == len(items):
        return False
    _save(SESSIONS_FILE, new_items)
    return True


# ── 내보내기 ──────────────────────────────────────────────────────────────────

def export_patient_json(patient_id: str) -> dict:
    """내담자 + 모든 회기를 단일 JSON 객체로 묶음 (다운로드용)."""
    patient = get_patient(patient_id)
    if patient is None:
        return {}
    return {
        "patient": patient,
        "sessions": list_sessions(patient_id),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
