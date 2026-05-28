from config import CHROMA_DIR, EMBEDDING_MODEL

CHROMA_DB_PATH = str(CHROMA_DIR)


def search(query: str, collection_name: str = "clinical_references", n_results: int = 3):
    import chromadb
    from sentence_transformers import SentenceTransformer
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL)

    collection = client.get_collection(collection_name)
    query_embedding = model.encode([query]).tolist()

    return collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )


if __name__ == "__main__":
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
