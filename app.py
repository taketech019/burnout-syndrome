import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="번아웃 조기 감지", layout="wide")

page = st.sidebar.selectbox("페이지", ["대시보드", "챗봇"])

if page == "대시보드":
    st.info("대시보드 (구현 예정)")
elif page == "챗봇":
    st.info("챗봇 (구현 예정)")
