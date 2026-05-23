"""F4 RAG 챗봇 페이지 — Gemma 4 31B + KoSBERT + ChromaDB."""
import streamlit as st

from src.rag import answer_query, healthcheck


def render() -> None:
    st.title("RAG 챗봇")
    st.caption(
        "Gemma 4 31B (Google AI Studio) + KoSBERT 임베딩 + ChromaDB. "
        "AI Hub 라벨링 데이터 · 임상 가이드라인 · 학회 윤리규정에서 검색."
    )

    status = healthcheck()
    cols = st.columns(2)
    cols[0].metric("LLM API 키", "✓" if status.get("llm") else "✗", help=status.get("llm_model", ""))
    cols[1].metric("ChromaDB 인덱스", "✓" if status.get("chroma") else "✗")

    if not status.get("llm"):
        st.warning(
            "**LLM API 키 미설정** — `.env`에 `GEMINI_API_KEY` 를 추가하세요. "
            "Google AI Studio (https://aistudio.google.com/apikey) 에서 발급."
        )
        return

    if not status.get("chroma"):
        st.warning(
            "**RAG 인덱스 미생성**. 다음 명령으로 빌드하세요:\n"
            "```bash\npython -m src.rag.ingest\n```"
        )
        return

    # 대화 히스토리
    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []

    for msg in st.session_state["rag_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("출처"):
                    for src in msg["sources"]:
                        st.markdown(f"- `{src['source']}` ({src.get('type', '?')})")
                        st.caption(src.get("snippet", ""))

    prompt = st.chat_input("질문을 입력하세요 (예: 우울증 환자에게 추천되는 인지행동치료 기법은?)")
    if prompt:
        st.session_state["rag_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("검색 + 응답 생성..."):
                result = answer_query(prompt)
            if result.get("error"):
                st.error(result["error"])
                st.session_state["rag_history"].append(
                    {"role": "assistant", "content": f"⚠️ {result['error']}"}
                )
            else:
                st.markdown(result["answer"])
                with st.expander("출처"):
                    for src in result["sources"]:
                        st.markdown(f"- `{src['source']}` ({src.get('type', '?')})")
                        st.caption(src.get("snippet", ""))
                st.session_state["rag_history"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
