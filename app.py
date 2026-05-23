"""CounsHelper — Streamlit 진입점.

PRD §F1~F5 통합:
- F5 내담자/회기 관리 (좌측 사이드바)
- F1 1차/2차 판별 + 28요인 분류
- F3 요약 보고서 (KoAlpaca + .md/.docx/.pdf)
- F2 대시보드
- F4 RAG 챗봇

내담자 선택은 session_state로 페이지 간 공유.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CounsHelper — 상담 기록 분석 & 보고서 자동화 플랫폼",
    layout="wide",
)

# ── 사이드바: 내담자 선택 + 페이지 라우팅 ─────────────────────────────────────
from src.storage import list_patients, add_patient, get_patient

with st.sidebar:
    st.header("CounsHelper")
    st.caption("AI 상담 기록 분석 플랫폼 (MVP — 데모 데이터 전용)")

    page = st.radio(
        "기능",
        ["내담자 관리", "회기 분석", "대시보드", "보고서", "RAG 챗봇"],
        index=0,
    )

    st.divider()

    patients = list_patients()
    if patients:
        options = {f"{p['alias']} ({p['gender']}/{p['age']}/{p['region']})": p["id"] for p in patients}
        selected_label = st.selectbox("내담자 선택", list(options.keys()))
        st.session_state["current_patient_id"] = options[selected_label]
    else:
        st.info("등록된 내담자 없음 — 내담자 관리에서 추가하세요.")
        st.session_state["current_patient_id"] = None


# ── 페이지 dispatch ────────────────────────────────────────────────────────────

if page == "내담자 관리":
    from src.app_pages.patients_page import render
    render()
elif page == "회기 분석":
    from src.app_pages.session_page import render
    render()
elif page == "대시보드":
    from src.app_pages.dashboard_page import render
    render()
elif page == "보고서":
    from src.app_pages.report_page import render
    render()
elif page == "RAG 챗봇":
    from src.app_pages.chatbot_page import render
    render()
