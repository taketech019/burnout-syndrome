"""F3 보고서 페이지 — KoAlpaca 요약 + 차트 + .md/.docx/.pdf 다운로드."""
import streamlit as st

from src.dashboard import chart_to_png, factor_frequency_chart, session_trend_chart
from src.report import build_docx, build_md, build_pdf
from src.storage import get_patient, get_session, list_sessions, update_session
from src.summarizer import summarize


def render() -> None:
    st.title("요약 보고서")
    pid = st.session_state.get("current_patient_id")
    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
        return
    patient = get_patient(pid)
    sessions = list_sessions(pid)
    if not sessions:
        st.info("이 내담자의 회기가 없습니다.")
        return

    options = {f"{s['session_date']} (ID: {s['id']})": s["id"] for s in sessions}
    selected_label = st.selectbox("회기 선택", list(options.keys()))
    s = get_session(options[selected_label])
    summary = s.get("summary") or {}

    # ── KoAlpaca 요약 (캐시 — 이미 있으면 재호출 안 함) ───────────────────────
    if not summary or st.button("재생성 (KoAlpaca 호출)"):
        with st.spinner("KoAlpaca 요약 생성 중... (Modal cold-start 시 1~2분 소요)"):
            summary = summarize(s["transcript"])
            update_session(s["id"], summary=summary)

    if "error" in summary:
        st.error(f"요약 실패: {summary['error']}")
        st.caption("transcript가 5000자 미만이면 KoAlpaca가 4섹션을 추출하지 못합니다 (학습 분포 median 22,872자).")

    st.markdown("### 4섹션 요약")
    edited = {}
    for key, label in [
        ("symptoms", "주요 증상"),
        ("risk_factors", "위험 요인"),
        ("improvement_factors", "개선 요인"),
        ("intervention_factors", "상담사 개입 요인"),
    ]:
        edited[key] = st.text_area(label, value=summary.get(key, ""), height=100, key=f"ed_{key}")

    next_plan = st.text_area(
        "다음 회기 계획 (수동 작성 또는 LLM 보조)",
        height=120,
        placeholder="목표 / 기법 / 과제 ...",
    )

    st.divider()
    st.markdown("### 다운로드")

    chart_pngs = []
    factors = s.get("factors", {})
    if factors.get("frequency"):
        png = chart_to_png(factor_frequency_chart(factors["frequency"]))
        if png:
            chart_pngs.append(png)
    if len(sessions) >= 2:
        png = chart_to_png(session_trend_chart(sessions))
        if png:
            chart_pngs.append(png)

    cols = st.columns(3)
    with cols[0]:
        md_bytes = build_md(patient, s, edited, next_plan)
        st.download_button(
            ".md 다운로드", md_bytes, file_name=f"report_{s['id']}.md", mime="text/markdown",
        )
    with cols[1]:
        try:
            docx_bytes = build_docx(patient, s, edited, next_plan, chart_pngs)
            st.download_button(
                ".docx 다운로드", docx_bytes, file_name=f"report_{s['id']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception as e:
            st.error(f".docx 생성 실패: {e}")
    with cols[2]:
        pdf_bytes = build_pdf(patient, s, edited, next_plan)
        if pdf_bytes:
            st.download_button(
                ".pdf 다운로드", pdf_bytes, file_name=f"report_{s['id']}.pdf", mime="application/pdf",
            )
        else:
            st.caption(".pdf 비활성 (weasyprint 환경 의존성).\n.md/.docx 사용을 권장.")
