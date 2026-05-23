"""F4 RAG 챗봇 페이지 — Ollama Qwen2.5 + KoSBERT + ChromaDB."""
import streamlit as st

from src.rag import answer_query, healthcheck


def render() -> None:
    st.title("RAG 챗봇")
    st.caption("로컬 sLLM (Qwen2.5 7B via Ollama) + KoSBERT + ChromaDB. 민감 데이터 외부 전송 없음.")

    status = healthcheck()
    cols = st.columns(2)
    cols[0].metric("Ollama", "✓" if status["ollama"] else "✗")
    cols[1].metric("ChromaDB 인덱스", "✓" if status["chroma"] else "✗")

    if not status["ollama"]:
        st.warning(
            f"**Ollama 미연결** ({status['ollama_url']}). 설정 절차:\n"
            f"1. https://ollama.com 에서 Ollama 설치\n"
            f"2. 터미널에서 `ollama serve` 실행\n"
            f"3. `ollama pull {status['model']}` 로 모델 다운로드\n"
            f"4. 이 페이지 새로고침"
        )
        if status.get("available_models"):
            st.caption(f"현재 설치된 모델: {status['available_models']}")
        return

    if not status["chroma"]:
        st.warning(
            "**RAG 인덱스 미생성**. 학습 데이터를 `data/raw/`에 두고 다음 명령으로 인덱스를 빌드하세요:\n"
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
                        st.caption(src["snippet"])

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
                        st.caption(src["snippet"])
                st.session_state["rag_history"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
