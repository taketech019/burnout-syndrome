"""src/rag/chain.py — RAG 챗봇 체인.

사용자 지시로 LLM은 Gemma 4 31B (Ollama Qwen2.5 대신). 임베딩은 PRD §F4 KoSBERT
(`snunlp/KR-SBERT-V40K-klueNLI-augSTS`). ChromaDB k=5 검색 + 모든 응답 출처 표시.
"""
import logging
from typing import Optional

from config import CHROMA_DIR, EMBEDDING_MODEL, GEMINI_API_KEY, GEMINI_MODEL
from src.gemma_client import generate, strip_reasoning

log = logging.getLogger(__name__)


DISCLAIMER = (
    "본 시스템은 상담사의 임상적 판단을 대체하지 않으며, "
    "상담 기록 정리와 회기 계획 수립을 보조하는 참고용 도구입니다."
)

_PROMPT_TEMPLATE = """당신은 한국어 임상심리 보조 챗봇입니다. **아래 [참고 자료]만 근거로** 답변하세요. 컨텍스트에 부족하면 "참고 자료에서 찾을 수 없습니다"라고 답하세요.

**중요**: 사고 과정·요약·메타 설명을 절대 출력하지 마세요. 한국어 답변만 직접 작성하세요. 영어·번역·"Role:" 같은 라벨 금지.

[참고 자료]
{context}

[질문]
{question}

[한국어 답변]"""


_EMBEDDER_CACHE = {"obj": None, "kind": None}


def _get_embedder():
    """KoSBERT 우선, 실패 시 LangChain default 폴백."""
    if _EMBEDDER_CACHE["obj"] is not None:
        return _EMBEDDER_CACHE["obj"], _EMBEDDER_CACHE["kind"]
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _EMBEDDER_CACHE.update(obj=emb, kind="kosbert")
        return emb, "kosbert"
    except Exception as e:
        log.warning("KoSBERT 임베딩 로드 실패 (%s) — 폴백 시도", e)
        from langchain_community.embeddings import FakeEmbeddings  # type: ignore
        emb = FakeEmbeddings(size=768)
        _EMBEDDER_CACHE.update(obj=emb, kind="fake")
        return emb, "fake"


def healthcheck() -> dict:
    """LLM(GEMINI_API_KEY) + ChromaDB 가용성."""
    status = {
        "llm": bool(GEMINI_API_KEY),
        "llm_model": GEMINI_MODEL,
        "chroma": (CHROMA_DIR / "chroma.sqlite3").exists(),
        "chroma_path": str(CHROMA_DIR),
    }
    return status


def _retrieve(query: str, k: int = 5):
    from langchain_community.vectorstores import Chroma

    emb, _ = _get_embedder()
    store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=emb,
        collection_name="counshelper",
    )
    return store.similarity_search(query, k=k)


def answer_query(query: str, k: int = 5) -> dict:
    status = healthcheck()
    if not status["llm"]:
        return {
            "answer": "",
            "sources": [],
            "error": "GEMINI_API_KEY 미설정 — .env 확인 필요.",
        }
    if not status["chroma"]:
        return {
            "answer": "",
            "sources": [],
            "error": (
                "RAG 인덱스 미생성. 다음 명령으로 빌드하세요:\n"
                "```bash\npython -m src.rag.ingest\n```"
            ),
        }

    try:
        docs = _retrieve(query, k=k)
        if not docs:
            return {"answer": "참고 자료에서 찾을 수 없습니다.", "sources": []}
        context = "\n\n".join(
            f"[자료 {i+1}] {d.page_content[:600]}\n(출처: {d.metadata.get('source', '?')})"
            for i, d in enumerate(docs)
        )
        prompt = _PROMPT_TEMPLATE.format(
            disclaimer=DISCLAIMER, context=context, question=query
        )
        text = generate(prompt, temperature=0.2, max_output_tokens=1024)
        cleaned = strip_reasoning(text)
        return {
            "answer": f"{cleaned.strip()}\n\n---\n*{DISCLAIMER}*",
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
