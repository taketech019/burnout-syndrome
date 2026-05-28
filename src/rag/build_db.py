import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

from config import PROCESSED_DIR, REFERENCES_DIR, CHROMA_DIR, EMBEDDING_MODEL

CHUNKS_PATH = PROCESSED_DIR / "counseling_chunks.jsonl"
SUMMARIES_PATH = PROCESSED_DIR / "session_summaries.jsonl"
CHROMA_DB_PATH = str(CHROMA_DIR)


def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            with open(file_path, "r", encoding="cp949") as f:
                return f.read()

    elif suffix == ".pdf":
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"PDF 읽기 실패: {file_path} - {e}")
            return ""

    elif suffix == ".docx":
        try:
            doc = Document(str(file_path))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            print(f"DOCX 읽기 실패: {file_path} - {e}")
            return ""

    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def init_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    counseling_col = client.get_or_create_collection(
        name="counseling_cases",
        metadata={"hnsw:space": "cosine"}
    )
    references_col = client.get_or_create_collection(
        name="clinical_references",
        metadata={"hnsw:space": "cosine"}
    )

    return client, counseling_col, references_col


def load_counseling_data(counseling_col, model):
    print("\n=== 상담 데이터 적재 시작 ===")

    if SUMMARIES_PATH.exists():
        summaries = []
        with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                summaries.append(json.loads(line))

        batch_size = 100
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i + batch_size]
            texts = [item["text"] for item in batch]
            ids = [item["id"] for item in batch]
            metadatas = [{"chunk_type": "summary",
                          "class": item["metadata"].get("class", ""),
                          "client_id": item["metadata"].get("client_id", ""),
                          "session": str(item["metadata"].get("session", ""))}
                         for item in batch]

            embeddings = model.encode(texts).tolist()

            counseling_col.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            print(f"summaries 적재: {min(i + batch_size, len(summaries))}/{len(summaries)}")

        print(f"summaries 완료: {len(summaries)}개")

    print("상담 데이터 적재 완료!")


def load_references(references_col, model):
    print("\n=== Reference 문서 적재 시작 ===")

    if not REFERENCES_DIR.exists():
        print("references 폴더 없음")
        return

    all_files = list(REFERENCES_DIR.rglob("*"))
    doc_files = [f for f in all_files
                 if f.is_file() and f.suffix.lower() in [".txt", ".pdf", ".docx"]
                 and f.name != ".gitkeep"]

    print(f"발견된 파일 수: {len(doc_files)}")

    doc_id_counter = 0

    for file_path in doc_files:
        print(f"처리 중: {file_path.name}")

        text = extract_text_from_file(file_path)
        if not text.strip():
            print(f"  → 텍스트 없음, 스킵")
            continue

        topic = file_path.parent.name
        if topic == "references":
            topic = "general"

        chunks = chunk_text(text, chunk_size=500, overlap=50)
        print(f"  → {len(chunks)}개 청크")

        if not chunks:
            continue

        embeddings = model.encode(chunks).tolist()

        ids = [f"ref_{file_path.stem}_{doc_id_counter + j}"
               for j in range(len(chunks))]
        metadatas = [{"source": file_path.name,
                      "topic": topic,
                      "language": "en" if file_path.suffix.lower() == ".pdf" else "ko",
                      "chunk_index": j}
                     for j in range(len(chunks))]

        batch_size = 100
        for b in range(0, len(chunks), batch_size):
            references_col.add(
                documents=chunks[b:b + batch_size],
                embeddings=embeddings[b:b + batch_size],
                ids=ids[b:b + batch_size],
                metadatas=metadatas[b:b + batch_size]
            )

        doc_id_counter += len(chunks)
        print(f"  → 적재 완료")

    print(f"\nReference 적재 완료! 총 {doc_id_counter}개 청크")


def build_database():
    print("임베딩 모델 로딩 중...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("모델 로딩 완료!")

    client, counseling_col, references_col = init_chroma()

    load_counseling_data(counseling_col, model)
    load_references(references_col, model)

    print("\n=== 최종 결과 ===")
    print(f"counseling_cases: {counseling_col.count()}개")
    print(f"clinical_references: {references_col.count()}개")
    print(f"ChromaDB 저장 위치: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    build_database()
