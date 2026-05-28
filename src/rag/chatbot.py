import os
import json
import requests
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"


# =========================
# 1. 초기화
# =========================
def init():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    counseling_col = client.get_collection("counseling_cases")
    references_col = client.get_collection("clinical_references")
    return model, counseling_col, references_col


# =========================
# 2. 검색
# =========================
def search_context(query, model, counseling_col, references_col, n=3):
    query_embedding = model.encode([query]).tolist()

    case_results = counseling_col.query(
        query_embeddings=query_embedding,
        n_results=n
    )
    ref_results = references_col.query(
        query_embeddings=query_embedding,
        n_results=n
    )

    context = ""
    context += "=== 유사 상담 사례 ===\n"
    for i, doc in enumerate(case_results["documents"][0]):
        meta = case_results["metadatas"][0][i]
        context += f"[사례 {i + 1} - {meta.get('class', '')}]\n{doc[:300]}\n\n"

    context += "=== 임상 가이드라인 ===\n"
    for i, doc in enumerate(ref_results["documents"][0]):
        meta = ref_results["metadatas"][0][i]
        context += f"[출처: {meta.get('source', '')}]\n{doc[:300]}\n\n"

    return context


# =========================
# 3. Gemini API 직접 호출
# =========================
def generate_answer(query, context):
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    prompt = f"""당신은 심리상담사를 보조하는 AI입니다.
아래 참고 자료를 바탕으로 상담사의 질문에 답변해주세요.

규칙:
1. 진단명을 확정하지 마세요.
2. 치료나 처방을 제안하지 마세요.
3. 상담사가 확인할 항목 중심으로 작성하세요.
4. 위험도 최종 판단은 상담사가 수행한다고 명시하세요.
5. 한국어로 답변하세요.

참고 자료:
{context}

상담사 질문: {query}

답변:"""

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(url, json=payload)
    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"오류 발생: {result}"


# =========================
# 4. 메인 챗봇 함수
# =========================
def chat(query: str) -> str:
    model, counseling_col, references_col = init()
    context = search_context(query, model, counseling_col, references_col)
    answer = generate_answer(query, context)
    return answer


# =========================
# 5. 테스트
# =========================
if __name__ == "__main__":
    test_queries = [
        "불안과 수면 문제를 호소하는 내담자 다음 회기에 확인할 내용은?",
        "자해 사고를 부인했지만 지치고 불안하다는 내담자 위험 요인은?",
    ]

    for query in test_queries:
        print(f"\n질문: {query}")
        print("-" * 50)
        answer = chat(query)
        print(answer)
        print("=" * 50)