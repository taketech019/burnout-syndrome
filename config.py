import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCES_DIR = DATA_DIR / "references"
AIHUB_DIR = REFERENCES_DIR / "심리상담데이터"

CHROMA_DIR = ROOT_DIR / "chroma_db"

# F5 내담자/회기 저장소
STORAGE_DIR = DATA_DIR / "storage"
PATIENTS_FILE = STORAGE_DIR / "patients.json"
SESSIONS_FILE = STORAGE_DIR / "sessions.json"

# F4 RAG 임베딩 — PRD §F4 KoSBERT. torch/sentence-transformers 미설치 환경에서는
# rag.embedding이 자동으로 Google text-embedding-004 로 폴백.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# F1/F4 LLM — Gemini 2.5 Flash (TPM 1M, JSON 모드 native 지원).
# Gemma 4 31B는 TPM 16K 한도 잦은 초과로 폐기 (29k자 transcript 9 호출에 TPM 초과).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")

# F3 KoAlpaca (Modal serverless)
KOALPACA_ENDPOINT_URL = os.getenv("KOALPACA_ENDPOINT_URL", "")
KOALPACA_API_KEY = os.getenv("KOALPACA_API_KEY", "")

# F1 1차 KlueBERT (HF Space — 변별력 부족 진단됨, 로컬 weights 우선)
KLUEBERT_ENDPOINT_URL = os.getenv("KLUEBERT_ENDPOINT_URL", "")
KLUEBERT_API_KEY = os.getenv("KLUEBERT_API_KEY", "")
KLUEBERT_LOCAL_DIR = ROOT_DIR / "ai-model" / "kluebert" / "2.AI학습모델파일"

# SQLite DB (단일 파일, 임베디드 — 외부 SQL DB 금지 정신과 일치)
DB_PATH = DATA_DIR / "counshelper.db"
