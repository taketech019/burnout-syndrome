"""src/stats.py — 상담사 전체 활동 통계.

분석 대시보드와 별도로 모든 환자/회기를 집계해 통계 페이지에 표시.
"""
import json
from collections import Counter
from typing import Optional

import pandas as pd

from src import db
from src.factor_extractor import FACTOR_CATEGORIES, FACTOR_KEYS


# 1분당 200자 추정 (한국어 발화 대화 평균)
_CHARS_PER_MINUTE = 200


def aggregate_global_stats() -> dict:
    """모든 환자/회기 집계."""
    with db._conn() as c:
        n_sessions = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        n_patients = c.execute("SELECT COUNT(DISTINCT patient_id) FROM sessions").fetchone()[0]
        total_chars = c.execute(
            "SELECT COALESCE(SUM(LENGTH(transcript)), 0) FROM sessions"
        ).fetchone()[0]

    total_minutes = int(total_chars / _CHARS_PER_MINUTE) if total_chars else 0
    avg_minutes = total_minutes / n_sessions if n_sessions else 0.0

    return {
        "sessions": int(n_sessions or 0),
        "patients": int(n_patients or 0),
        "total_minutes": total_minutes,
        "avg_minutes": round(avg_minutes, 1),
    }


def classification_distribution() -> pd.DataFrame:
    """모든 회기의 최신 classifier 결과를 우울/불안/중독/정상 분포로 집계.

    한 회기가 우울+불안 동시면 둘 다 카운트. 모두 0이면 '정상군' 카운트.
    """
    counts = {"우울": 0, "불안": 0, "중독": 0, "정상군": 0}
    with db._conn() as c:
        sessions = c.execute("SELECT id FROM sessions").fetchall()

    for s in sessions:
        latest = db.get_latest_analysis(s["id"], "classifier")
        if not latest:
            continue
        cls = latest["payload"].get("classification", {})
        marked_any = False
        if cls.get("depression"):
            counts["우울"] += 1
            marked_any = True
        if cls.get("anxiety"):
            counts["불안"] += 1
            marked_any = True
        if cls.get("addiction"):
            counts["중독"] += 1
            marked_any = True
        if not marked_any:
            counts["정상군"] += 1

    return pd.DataFrame({
        "분류": list(counts.keys()),
        "건수": list(counts.values()),
    })


def factor_top_n(n: int = 10) -> pd.DataFrame:
    """28요인 전 회기 평균 점수 Top N (긍정 점수만 카운트)."""
    sums = Counter()
    n_with_factors = 0

    with db._conn() as c:
        rows = c.execute(
            "SELECT payload_json FROM analyses WHERE stage = 'factors' "
            "ORDER BY datetime(created_at) DESC"
        ).fetchall()

    # 세션별 최신 1개만 사용 — analyses는 같은 stage 여러 행 가능
    seen_sessions = set()
    with db._conn() as c:
        latest_rows = c.execute(
            "SELECT session_id, payload_json FROM analyses WHERE stage = 'factors' "
            "ORDER BY datetime(created_at) DESC, id DESC"
        ).fetchall()
    for r in latest_rows:
        if r["session_id"] in seen_sessions:
            continue
        seen_sessions.add(r["session_id"])
        try:
            payload = json.loads(r["payload_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        factors = payload.get("factors", {})
        for k in FACTOR_KEYS:
            sums[k] += int(factors.get(k, 0))
        n_with_factors += 1

    if n_with_factors == 0:
        return pd.DataFrame(columns=["요인", "카테고리", "평균 점수", "총합"])

    rows_out = []
    for k in FACTOR_KEYS:
        rows_out.append({
            "요인": k,
            "카테고리": FACTOR_CATEGORIES.get(k, "기타"),
            "평균 점수": round(sums[k] / n_with_factors, 2),
            "총합": sums[k],
        })
    df = pd.DataFrame(rows_out).sort_values("평균 점수", ascending=False).head(n)
    return df.reset_index(drop=True)
