"""src/insight.py — 분석 해석 카드용 텍스트 생성.

대시보드와 통계 페이지의 우측 사이드 "해석" 카드에 표시할 자연어 요약.
DB에 캐시된 summary.brief를 우선 사용해 Gemini 추가 호출 비용 절감.
"""
from typing import Optional

import pandas as pd

from src import db


def dashboard_insight(session_id: str) -> str:
    """단일 회기 분석 결과 → 한 단락 해석.

    1) summary.brief가 있으면 그대로 사용 (Gemini 호출 없음).
    2) brief 없으면 sections 첫 두 섹션 + classifier 정보 조합.
    3) 분석 결과 자체 없으면 빈 문자열.
    """
    summ = db.get_latest_analysis(session_id, "summary")
    if summ:
        brief = (summ["payload"].get("brief") or "").strip()
        if brief:
            return brief
        # brief 없으면 sections에서 발췌
        sections = summ["payload"].get("sections", {})
        parts = []
        if sections.get("symptoms"):
            parts.append(sections["symptoms"].split(".")[0] + ".")
        if sections.get("improvement_factors"):
            parts.append(sections["improvement_factors"].split(".")[0] + ".")
        if parts:
            return " ".join(parts)

    cls = db.get_latest_analysis(session_id, "classifier")
    if cls:
        c = cls["payload"].get("classification", {})
        sig = []
        if c.get("depression"):
            sig.append("우울")
        if c.get("anxiety"):
            sig.append("불안")
        if c.get("addiction"):
            sig.append("중독")
        if sig:
            return f"이번 회기는 {'/'.join(sig)} 관련 호소가 관찰됩니다. 자세한 분석은 좌측 차트를 참고하세요."
        return "이번 회기는 분류 모델 기준 정상군으로 분류되었습니다."

    return "분석 결과가 아직 생성되지 않았습니다. '상담내역 기록·추가'에서 AI 분석을 실행하세요."


def hira_summary_one_line(hira_result: dict, patient: dict, disease: str) -> str:
    """HIRA lookup 결과를 한 줄로 압축."""
    if not hira_result.get("available"):
        return ""
    m = hira_result.get("matched", {})
    rp = m.get("region_patients")
    np_ = m.get("national_patients")
    decade = (int(patient.get("age", 30)) // 10) * 10
    if rp is not None and np_:
        return (
            f"HIRA 2024: {patient.get('region', '?')} {decade}대 "
            f"{patient.get('gender', '?')} {disease} 환자 {int(rp):,}명 "
            f"(전국 필터 합계 {int(np_):,}명)."
        )
    return ""


def stats_insight(stats: dict, dist: pd.DataFrame) -> str:
    """전체 통계 요약 — 4 메트릭 + 분류 분포를 자연어로.

    Gemini 호출 없이 템플릿 기반. 사용자가 짧게 훑어볼 수 있는 한국어 단락.
    """
    if stats["sessions"] == 0:
        return (
            "아직 등록된 상담 회기가 없습니다. "
            "'상담내역 기록·추가'에서 첫 회기를 등록하세요."
        )

    total_hours = stats["total_minutes"] // 60
    total_remainder = stats["total_minutes"] % 60

    lines = [
        f"**■ 총괄**",
        f"해당 기간 총 {stats['sessions']}건의 상담이 실시되었으며, "
        f"{stats['patients']}명의 내담자가 참여하였습니다. "
        f"총 상담 시간은 {total_hours}시간 {total_remainder}분이며, "
        f"건당 평균 상담 시간은 약 {stats['avg_minutes']}분입니다.",
        "",
    ]

    if not dist.empty:
        total = dist["건수"].sum()
        if total > 0:
            lines.append("**■ 분류별 현황**")
            parts = []
            for _, row in dist.iterrows():
                pct = row["건수"] / total * 100
                if row["건수"] > 0:
                    parts.append(f"{row['분류']} {row['건수']}건({pct:.0f}%)")
            lines.append(", ".join(parts) + ".")
            lines.append("")

    lines.append("**■ 활용 안내**")
    lines.append(
        "각 회기의 자세한 분석은 좌측 사이드바 '분석 대시보드'에서 "
        "내담자 + 회기를 선택해 확인할 수 있습니다."
    )

    return "\n".join(lines)
