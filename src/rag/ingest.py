"""src/rag/ingest.py — RAG 인덱스 빌드 (ChromaDB + KoSBERT).

PRD §F4 검색 소스:
- AI Hub 학습 데이터 465K건 (JSON in data/raw/)
- 임상 가이드라인 PDF (data/references/)

빌드 명령: `python -m src.rag.ingest`
"""
import json
import os
from pathlib import Path
from typing import Iterator

from config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, RAW_DIR, REFERENCES_DIR


def _iter_aihub_paragraphs(raw_dir: Path) -> Iterator[dict]:
    """data/raw/ 아래의 AI Hub JSON 파일을 발화 단위로 yield."""
    if not raw_dir.exists():
        return
    for json_file in raw_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        paragraphs = data.get("paragraph", [])
        for i, p in enumerate(paragraphs):
            speaker = p.get("paragraph_speaker", "")
            text = p.get("paragraph_text", "")
            if not text.strip():
                continue
            yield {
                "text": f"{speaker}: {text}",
                "source": str(json_file.relative_to(DATA_DIR)),
                "para_idx": i,
                "type": "aihub_session",
            }


def _iter_reference_pdfs(refs_dir: Path) -> Iterator[dict]:
    """data/references/ 의 PDF 파일을 페이지 단위로 yield. PDF 로딩 실패 시 skip."""
    if not refs_dir.exists():
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        return
    for pdf_file in refs_dir.rglob("*.pdf"):
        try:
            reader = PdfReader(pdf_file)
        except Exception:
            continue
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            yield {
                "text": text,
                "source": str(pdf_file.relative_to(DATA_DIR)),
                "page": i + 1,
                "type": "clinical_guideline",
            }


def build_index(max_docs: int = 50000) -> dict:
    """ChromaDB에 KoSBERT 임베딩으로 인덱스 빌드. 반환: 통계."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    docs: list[Document] = []
    counts = {"aihub_session": 0, "clinical_guideline": 0}

    for item in _iter_aihub_paragraphs(RAW_DIR):
        docs.append(Document(page_content=item["text"], metadata={k: v for k, v in item.items() if k != "text"}))
        counts["aihub_session"] += 1
        if len(docs) >= max_docs:
            break

    if len(docs) < max_docs:
        for item in _iter_reference_pdfs(REFERENCES_DIR):
            docs.append(Document(page_content=item["text"], metadata={k: v for k, v in item.items() if k != "text"}))
            counts["clinical_guideline"] += 1
            if len(docs) >= max_docs:
                break

    if not docs:
        return {"error": f"인덱싱 대상 문서 0개. {RAW_DIR} 또는 {REFERENCES_DIR}에 파일을 두세요.", **counts}

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        docs, embedder, persist_directory=str(CHROMA_DIR), collection_name="counshelper"
    )
    store.persist()
    return {"total": len(docs), "persist_dir": str(CHROMA_DIR), **counts}


if __name__ == "__main__":
    print("RAG 인덱스 빌드 시작...")
    result = build_index()
    print(json.dumps(result, ensure_ascii=False, indent=2))
