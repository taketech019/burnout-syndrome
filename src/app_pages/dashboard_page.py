"""F2 대시보드 페이지 — 4종 차트 (회기 카드, 4범주 빈도, 회기 추이, HIRA)."""
import streamlit as st

from src.dashboard import (
    factor_frequency_chart, hira_comparison_text,
    session_severity_summary, session_trend_chart,
)
from src.storage import get_patient, list_sessions


def render() -> None:
    st.title("대시보드")
    pid = st.session_state.get("current_patient_id")
    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
        return
    patient = get_patient(pid)
    sessions = list_sessions(pid)

    if not sessions:
        st.info("이 내담자의 분석된 회기가 없습니다. '회기 분석' 페이지에서 추가하세요.")
        return

    # 최신 회기 선택
    options = {f"{s['session_date']} (ID: {s['id']})": s for s in sessions}
    selected_label = st.selectbox("회기 선택", list(options.keys()))
    s = options[selected_label]

    cls = s.get("classifier", {})
    factors = s.get("factors", {})

    # ── 1. 회기 문제 수준 카드 ────────────────────────────────────────────────
    st.subheader("회기 문제 수준 (F1 1차)")
    # Gemma 1차에서 산출된 0~3 정도값을 정도(level)로 표시. KlueBERT 모드에서는 None.
    sev = session_severity_summary(cls, raw_values=cls.get("raw"))
    levels = cls.get("level") or {}
    cols = st.columns(3)
    for i, label in enumerate(("anxiety", "depression", "addiction")):
        d = sev["severities"][label]
        level = levels.get(label)
        if level is not None:
            cols[i].metric(label, f"{level} / 3", delta="양성" if level >= 1 else "정상군",
                           delta_color="inverse" if level >= 1 else "off")
        else:
            cols[i].metric(label, d["binary"])
    if sev.get("note"):
        st.caption(f"ℹ️ {sev['note']}")
    if cls.get("_source"):
        st.caption(f"판별 소스: `{cls['_source']}`")

    # ── 2. 4범주 빈도 차트 ────────────────────────────────────────────────────
    st.subheader("4범주 요인별 등장 빈도 (F1 2차)")
    if factors.get("frequency"):
        fig = factor_frequency_chart(factors["frequency"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("2차 분류 결과 없음.")

    # ── 3. 회기별 추이 라인 ───────────────────────────────────────────────────
    st.subheader("회기별 핵심 요인 추이")
    if len(sessions) >= 2:
        trend_fig = session_trend_chart(sessions)
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.caption("회기가 2개 이상 누적되면 추이가 표시됩니다.")

    # ── 4. HIRA 인구통계 비교 ─────────────────────────────────────────────────
    st.subheader("HIRA 인구통계 비교")
    primary = factors.get("primary_disease", "depression")
    st.markdown(hira_comparison_text(patient, primary))
