"""scripts/redo_stale_summaries.py — 모든 회기의 classifier/factors/summary 재분석.

Phase 4·5 코드 변경 전에 분석된 회기는 stale. 이 스크립트가 모든 session의
transcript를 다시 처리해서 새 analyses 행을 추가. db.get_latest_analysis는
최신 created_at 우선이므로 UI에 자동으로 새 결과 표시.

옵션:
  --patient <alias> : 특정 환자만 (예: 데모B)
  미지정 시 전체 환자
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db
from src.classifier import classify_text
from src.factor_extractor import extract_factors
from src.summarizer import summarize


def redo_session(s: dict) -> None:
    transcript = s.get("transcript", "")
    if not transcript.strip():
        print(f"  skip {s['id']} — transcript 비어 있음")
        return
    print(f"  → {s['id']} ({len(transcript)}자): 재분석 시작")
    cls = classify_text(transcript)
    fact = extract_factors(transcript, cls["classification"])
    summ = summarize(transcript)
    db.add_analysis(s["id"], "classifier", cls["backend"], cls)
    db.add_analysis(s["id"], "factors", fact["backend"], fact)
    db.add_analysis(s["id"], "summary", summ["source"], summ)
    sec_count = sum(1 for v in summ["sections"].values() if v.strip())
    pos_factors = sum(1 for v in fact["factors"].values() if v > 0)
    print(
        f"    classification={cls['classification']} "
        f"factors_positive={pos_factors}/28 "
        f"summary source={summ['source']} sections={sec_count}/4"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patient", help="alias 필터 (예: 데모B). 미지정 시 모든 환자.",
    )
    args = parser.parse_args()

    patients = db.list_patients()
    if args.patient:
        patients = [p for p in patients if p["alias"] == args.patient]
    if not patients:
        print("대상 환자 없음.")
        return

    for p in patients:
        sessions = db.list_sessions(p["id"])
        print(f"\n[{p['alias']}] {len(sessions)} 회기")
        for s in sessions:
            redo_session(s)

    print("\n완료.")


if __name__ == "__main__":
    main()
