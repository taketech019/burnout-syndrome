"""src/rag/chain.py — RAG 챗봇 체인 (Ollama Qwen2.5 + ChromaDB + KoSBERT).

PRD §F4 요구사항:
- 로컬 LLM (Ollama Qwen2.5 7B) — 민감 데이터 외부 전송 없음
- KoSBERT 임베딩
- ChromaDB k=5
- 모든 응답에 출처 표시
- 고지 문구 자동 포함

Ollama가 설치되지 않은 환경에서는 healthcheck()가 명확한 안내 메시지 반환.
"""
import requests

from config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL


DISCLAIMER = (
    "본 시스템은 상담사의 임상적 판단을 대체하지 않으며, "
    "상담 기록 정리와 회기 계획 수립을 보조하는 참고용 도구입니다."
)

_PROMPT_TEMPLATE = """당신은 한국어 임상심리 보조 챗봇입니다. 아래 컨텍스트만 근거로 답하세요. 컨텍스트에 없으면 "참고 자료에서 찾을 수 없습니다"라고 답하세요.

[고지]
{disclaimer}

[참고 자료]
{context}

[질문]
{question}

[답변]
"""


def healthcheck() -> dict:
    """Ollama + ChromaDB 가용성 확인. UI에서 RAG 사용 가능 여부 판단용."""
    status = {"ollama": False, "chroma": False, "ollama_url": OLLAMA_BASE_URL, "model": OLLAMA_MODEL}
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if r.ok:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            status["ollama"] = OLLAMA_MODEL in models or any(OLLAMA_MODEL.split(":")[0] in m for m in models)
            status["available_models"] = models
    except requests.RequestException:
        pass
    status["chroma"] = (CHROMA_DIR / "chroma.sqlite3").exists()
    return status


def _retrieve(query: str, k: int = 5) -> list:
    """ChromaDB에서 KoSBERT 임베딩 기반 k-NN 검색. Document 리스트 반환."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder,
        collection_name="counshelper",
    )
    return store.similarity_search(query, k=k)


def _call_ollama(prompt: str, timeout: int = 30) -> str:
    """Ollama /api/generate 호출. 실패 시 RuntimeError."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def answer_query(query: str, k: int = 5) -> dict:
    """RAG 응답. 반환: {"answer", "sources", "error"?}.

    Ollama 미설치/미실행 또는 ChromaDB 인덱스 부재 시 명시적 error 메시지로 폴백.
    """
    status = healthcheck()

    if not status["ollama"]:
        return {
            "answer": "",
            "sources": [],
            "error": (
                f"Ollama 미연결 (URL: {OLLAMA_BASE_URL}). "
                f"`ollama serve` 후 `ollama pull {OLLAMA_MODEL}`로 모델을 받으세요."
            ),
        }

    if not status["chroma"]:
        return {
            "answer": "",
            "sources": [],
            "error": (
                "RAG 인덱스 미생성. 학습 데이터를 `data/raw/`에 두고 "
                "`python -m src.rag.ingest`로 인덱스를 빌드하세요."
            ),
        }

    try:
        docs = _retrieve(query, k=k)
        context = "\n\n".join(
            f"[자료 {i+1}] {d.page_content[:500]}\n(출처: {d.metadata.get('source', '?')})"
            for i, d in enumerate(docs)
        )
        prompt = _PROMPT_TEMPLATE.format(disclaimer=DISCLAIMER, context=context, question=query)
        answer = _call_ollama(prompt)
        return {
            "answer": f"{answer}\n\n---\n*{DISCLAIMER}*",
            "sources": [
                {
                    "source": d.metadata.get("source", "?"),
                    "snippet": d.page_content[:200],
                    "type": d.metadata.get("type", "?"),
                }
                for d in docs
            ],
        }
    except Exception as e:
        return {"answer": "", "sources": [], "error": f"RAG 처리 실패: {e}"}
