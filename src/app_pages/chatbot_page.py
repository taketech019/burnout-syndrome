"""F4 RAG 챗봇 페이지 — KoSBERT + ChromaDB + Gemini."""
import streamlit as st

from src.rag import chat


def render() -> None:
    st.title("RAG 챗봇")
    st.caption(
        "KoSBERT 임베딩 + ChromaDB. "
        "AI Hub 라벨링 데이터 · 임상 가이드라인 · 학회 윤리규정에서 검색."
    )

    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []

    for msg in st.session_state["rag_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("질문을 입력하세요 (예: 우울증 환자에게 추천되는 인지행동치료 기법은?)")
    if prompt:
        st.session_state["rag_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("검색 + 응답 생성..."):
                answer = chat(prompt)
            st.markdown(answer)
            st.session_state["rag_history"].append({"role": "assistant", "content": answer})
