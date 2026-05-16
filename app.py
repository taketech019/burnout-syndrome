import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼",
    layout="wide",
)

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.header("CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼")

# ── 페이지 라우팅 ─────────────────────────────────────────────────────────────
page = st.sidebar.selectbox("페이지", ["대시보드", "상담기록"])

if page == "대시보드":
    st.info("대시보드 (구현 예정)")
elif page == "상담기록":
    st.info("상담기록 (구현 예정)")

# ── 하단 고정 채팅 입력바 (모든 페이지 공유) ──────────────────────────────────
prompt = st.chat_input("상담 내용을 입력하세요...")
if prompt:
    st.session_state.setdefault("chat_history", []).append(
        {"role": "user", "content": prompt}
    )
