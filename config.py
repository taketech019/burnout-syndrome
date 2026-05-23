import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCES_DIR = DATA_DIR / "references"

CHROMA_DIR = ROOT_DIR / "chroma_db"

# F5 내담자/회기 저장소 (JSON 파일 영속화)
STORAGE_DIR = DATA_DIR / "storage"
PATIENTS_FILE = STORAGE_DIR / "patients.json"
SESSIONS_FILE = STORAGE_DIR / "sessions.json"

# F4 RAG 임베딩 — PRD §F4: KoSBERT 한국어 특화
EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# F4 RAG LLM — PRD §F4: Ollama 로컬 Qwen2.5 7B
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# F1 2단계 Gemini 28요인 분류 — Google AI Studio API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# KoAlpaca (Modal serverless, NF4 + LoRA attach, A10G 24GB)
KOALPACA_ENDPOINT_URL = os.getenv("KOALPACA_ENDPOINT_URL", "")
KOALPACA_API_KEY      = os.getenv("KOALPACA_API_KEY", "")

# KlueBERT (Hugging Face Spaces, free CPU) — sleep 시 cold-start 60~120s
KLUEBERT_ENDPOINT_URL = os.getenv("KLUEBERT_ENDPOINT_URL", "")
KLUEBERT_API_KEY      = os.getenv("KLUEBERT_API_KEY", "")
