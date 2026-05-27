import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent.parent
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"


def search(query: str, collection_name: str = "clinical_references", n_results: int = 3):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL)

    collection = client.get_collection(collection_name)
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return results


if __name__ == "__main__":
    # 테스트
    print("=== clinical_references 검색 테스트 ===")
    results = search("불안장애 치료 방법", "clinical_references", 3)
    for i, doc in enumerate(results["documents"][0]):
        print(f"\n--- 결과 {i + 1} ---")
        print(f"출처: {results['metadatas'][0][i]['source']}")
        print(f"내용: {doc[:200]}")

    print("\n=== counseling_cases 검색 테스트 ===")
    results = search("불안 수면 문제 호소", "counseling_cases", 3)
    for i, doc in enumerate(results["documents"][0]):
        print(f"\n--- 결과 {i + 1} ---")
        print(f"클래스: {results['metadatas'][0][i]['class']}")
        print(f"내용: {doc[:200]}")

print("\n=== 영어 논문 검색 테스트 ===")
results = search("cognitive behavioral therapy anxiety treatment", "clinical_references", 3)
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- 결과 {i+1} ---")
    print(f"출처: {results['metadatas'][0][i]['source']}")
    print(f"내용: {doc[:200]}")

print("\n=== 한국어로 물어봤을 때 영어 논문 나오나 ===")
results = search("인지행동치료 불안 효과", "clinical_references", 5)
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- 결과 {i+1} ---")
    print(f"출처: {results['metadatas'][0][i]['source']}")
    print(f"언어: {results['metadatas'][0][i]['language']}")
    print(f"내용: {doc[:100]}")