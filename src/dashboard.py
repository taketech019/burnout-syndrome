"""src/dashboard.py — F2: 내담자 맞춤 대시보드.

PRD §F2 4종 차트:
1. 회기 문제 수준 카드 (KlueBERT 판별 결과 + 정도값)
2. 4범주 빈도 차트 (가로 막대)
3. 회기별 추이 라인 (핵심 요인 3~5개)
4. HIRA 인구통계 비교

차트는 plotly Figure 객체로 반환. Streamlit 측에서 st.plotly_chart로 렌더, 또는 kaleido로 PNG 변환 후 보고서 임베드.
"""
from typing import Optional

import plotly.graph_objects as go


CATEGORY_COLORS = {
    "symptom_factor": "#e53e3e",
    "risk_factor": "#dd6b20",
    "improvement_factor": "#38a169",
    "intervention_factor": "#3182ce",
}

CATEGORY_LABELS = {
    "symptom_factor": "증상요인",
    "risk_factor": "위험요인",
    "improvement_factor": "개선요인",
    "intervention_factor": "개입요인",
}


# ── 1) 회기 문제 수준 카드 (텍스트만, Streamlit에서 st.metric으로 표시) ────────

def session_severity_summary(classifier_result: dict, raw_values: Optional[dict] = None) -> dict:
    """KlueBERT 1차 판별 + 정도값.

    BYPASS 모드에서는 anxiety/depression/addiction 모두 1로 떨어짐 — 그 사실을 그대로 노출.
    raw_values가 있으면 정도값(0~3)으로 변환해서 함께 표시.
    """
    severities = {}
    for label in ("anxiety", "depression", "addiction"):
        binary = classifier_result.get(label, 0)
        raw = (raw_values or {}).get(label)
        if raw is not None:
            level = int(round(max(0, min(3, raw))))
            severities[label] = {"binary": binary, "level": level, "raw": round(raw, 3)}
        else:
            severities[label] = {"binary": binary, "level": None, "raw": None}
    return {
        "severities": severities,
        "is_normal": classifier_result.get("is_normal", False),
        "note": classifier_result.get("_note", ""),
    }


# ── 2) 4범주 빈도 가로 막대 ───────────────────────────────────────────────────

def factor_frequency_chart(frequency: dict, top_n_per_category: int = 5) -> go.Figure:
    """카테고리별 상위 N개 라벨의 등장 횟수를 가로 막대로."""
    fig = go.Figure()
    y_labels = []
    x_values = []
    bar_colors = []
    for category, items in frequency.items():
        if not items:
            continue
        top = sorted(items, key=lambda x: x.get("count", 0), reverse=True)[:top_n_per_category]
        for it in top:
            y_labels.append(f"[{CATEGORY_LABELS.get(category, category)}] {it['label']}")
            x_values.append(it.get("count", 0))
            bar_colors.append(CATEGORY_COLORS.get(category, "#999"))

    fig.add_trace(go.Bar(
        x=x_values, y=y_labels, orientation="h",
        marker=dict(color=bar_colors),
        hovertemplate="%{y}: %{x}회<extra></extra>",
    ))
    fig.update_layout(
        title="회기 내 요인별 등장 빈도 (카테고리별 상위 5)",
        xaxis_title="등장 횟수", yaxis_title="",
        height=max(400, 30 * len(y_labels)),
        margin=dict(l=200, r=20, t=60, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ── 3) 회기별 추이 라인 ───────────────────────────────────────────────────────

def session_trend_chart(sessions: list[dict], top_labels: Optional[list[str]] = None) -> go.Figure:
    """여러 회기에 걸친 핵심 요인 빈도 추이.

    sessions: storage.list_sessions(patient_id) 결과 (최신순 정렬). 차트는 시간순(오래→최신).
    top_labels: 추적할 라벨 이름 리스트. None이면 최신 회기의 상위 5개 자동 선택.
    """
    if not sessions:
        return go.Figure().update_layout(title="회기 추이 (회기 데이터 없음)")

    chrono = sorted(sessions, key=lambda s: s.get("session_date", ""))
    dates = [s.get("session_date", "?") for s in chrono]

    # 라벨 → 회기별 count 매핑 구축
    def _label_count(session: dict, label_name: str) -> int:
        freq = session.get("factors", {}).get("frequency", {})
        for category_items in freq.values():
            for item in category_items:
                if item.get("label") == label_name:
                    return item.get("count", 0)
        return 0

    if top_labels is None:
        latest = chrono[-1]
        all_items = []
        for items in latest.get("factors", {}).get("frequency", {}).values():
            all_items.extend(items)
        top_labels = [
            it["label"] for it in sorted(all_items, key=lambda x: x.get("count", 0), reverse=True)[:5]
        ]

    fig = go.Figure()
    for label_name in top_labels:
        ys = [_label_count(s, label_name) for s in chrono]
        fig.add_trace(go.Scatter(x=dates, y=ys, mode="lines+markers", name=label_name))

    fig.update_layout(
        title="회기별 핵심 요인 추이",
        xaxis_title="회기일", yaxis_title="등장 횟수",
        height=400, margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ── 4) HIRA 인구통계 비교 ─────────────────────────────────────────────────────

# MVP 가드레일: HIRA 실제 API 연동 없이 stub (예시 수치). PRD §F2의 형식만 충족.
# 정확한 데이터셋 픽 결정 후 hira.py에서 실데이터 fetch 교체.
_HIRA_STUB = {
    "depression": {
        ("20", "M"): {"region": 5.2, "national": 4.1},
        ("20", "F"): {"region": 7.8, "national": 6.5},
        ("30", "M"): {"region": 5.5, "national": 4.3},
        ("30", "F"): {"region": 8.3, "national": 6.1},
        ("40", "M"): {"region": 5.8, "national": 4.7},
        ("40", "F"): {"region": 8.1, "national": 6.4},
    },
}


def hira_comparison_text(patient: dict, primary_disease: str = "depression") -> str:
    """내담자 메타에 대응하는 HIRA stub 비교 문장. 실제 API 미연동, MVP demo용."""
    age_bucket = str(int(patient.get("age", 30)) // 10 * 10)
    gender_code = "F" if patient.get("gender", "").startswith("여") else "M"
    key = (age_bucket, gender_code)
    table = _HIRA_STUB.get(primary_disease, {})
    if key not in table:
        return f"(HIRA 통계: {patient.get('age', '?')}대 {patient.get('gender', '?')} 데이터 없음 — 실데이터 연동 예정)"
    d = table[key]
    region = patient.get("region", "지역")
    return (
        f"{age_bucket}대 {patient.get('gender', '?')} {region} {primary_disease} 진료율 "
        f"**{d['region']}%**, 전국 평균 **{d['national']}%** "
        f"(MVP stub — 실제 HIRA 데이터 연동 시 갱신 예정)"
    )


# ── PNG 이미지 변환 (보고서 임베드용) ─────────────────────────────────────────

def chart_to_png(fig: go.Figure, width: int = 800, height: int = 500) -> Optional[bytes]:
    """plotly Figure → PNG bytes. kaleido 의존성. 실패 시 None."""
    try:
        return fig.to_image(format="png", width=width, height=height, engine="kaleido")
    except Exception:
        return None
