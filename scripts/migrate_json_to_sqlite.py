"""기존 data/storage/{patients,sessions}.json → SQLite 1회 이전. 멱등."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PATIENTS_FILE, SESSIONS_FILE
from src import db


def main() -> None:
    db.init_db()

    existing_pids = {p["id"] for p in db.list_patients()}
    inserted_p = 0

    if PATIENTS_FILE.exists():
        patients = json.loads(PATIENTS_FILE.read_text(encoding="utf-8"))
        for p in patients:
            if p["id"] in existing_pids:
                continue
            with db._conn() as c:
                c.execute(
                    "INSERT INTO patients(id, alias, gender, age, region, note, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (p["id"], p.get("alias", ""), p.get("gender", ""),
                     int(p.get("age", 0)), p.get("region", ""),
                     p.get("note", ""), p.get("created_at", "")),
                )
            inserted_p += 1
        print(f"patients: 검사 {len(patients)}건, 신규 {inserted_p}건")

    existing_sids = {s["id"] for s in db.list_sessions()}
    inserted_s = 0
    inserted_a = 0

    if SESSIONS_FILE.exists():
        sessions = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        for s in sessions:
            if s["id"] in existing_sids:
                continue
            with db._conn() as c:
                c.execute(
                    "INSERT INTO sessions(id, patient_id, session_date, transcript, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (s["id"], s["patient_id"], s.get("session_date", ""),
                     s.get("transcript", ""), s.get("created_at", "")),
                )
            inserted_s += 1

            # 기존 JSON 안의 classifier/factors/summary 필드를 analyses 테이블에 저장
            for stage in ("classifier", "factors", "summary"):
                payload = s.get(stage)
                if payload and isinstance(payload, dict):
                    backend = (payload.get("_source")
                               or payload.get("backend")
                               or "legacy_json")
                    db.add_analysis(s["id"], stage=stage, backend=backend, payload=payload)
                    inserted_a += 1

        print(f"sessions: 검사 {len(sessions)}건, 신규 {inserted_s}건, analyses {inserted_a}건")

    print(f"완료 — patients={len(db.list_patients())}, sessions={len(db.list_sessions())}")


if __name__ == "__main__":
    main()
