"""src/storage.py — 호환 shim. 실제 구현은 src.db.

기존 API:
  list_patients, get_patient, add_patient, delete_patient,
  list_sessions, get_session, add_session, update_session, delete_session,
  export_patient_json
"""
from src.db import (
    add_analysis,
    add_patient,
    add_session,
    delete_patient,
    delete_session,
    export_patient as _export_patient,
    get_latest_analysis,
    get_patient,
    get_session,
    init_db,
    list_analyses,
    list_patients,
    list_sessions,
)


def export_patient_json(patient_id: str) -> dict:
    """기존 함수명 호환."""
    return _export_patient(patient_id)


def update_session(session_id: str, **fields) -> dict | None:
    """기존 호출 호환 — sessions 테이블엔 transcript/session_date만,
    classifier/factors/summary는 analyses 테이블로 라우팅."""
    sess = get_session(session_id)
    if sess is None:
        return None
    for stage_key in ("classifier", "factors", "summary"):
        if stage_key in fields and isinstance(fields[stage_key], dict):
            payload = fields[stage_key]
            backend = (payload.get("_source")
                       or payload.get("backend")
                       or "shim_update")
            add_analysis(session_id, stage=stage_key, backend=backend, payload=payload)
    return get_session(session_id)


__all__ = [
    "init_db", "add_patient", "list_patients", "get_patient", "delete_patient",
    "add_session", "list_sessions", "get_session", "delete_session", "update_session",
    "add_analysis", "list_analyses", "get_latest_analysis",
    "export_patient_json",
]
