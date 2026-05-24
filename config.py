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

EMBEDDING_MODEL = "BAAI/bge-m3"

OPENAI_MODEL = "gpt-4o"

# KoAlpaca (llama-server via Cloudflare Tunnel)
KOALPACA_ENDPOINT_URL = os.getenv("KOALPACA_ENDPOINT_URL", "")
KOALPACA_API_KEY      = os.getenv("KOALPACA_API_KEY", "")

# KlueBERT (Hugging Face Spaces, free CPU) — sleep 시 cold-start 60~120s
KLUEBERT_ENDPOINT_URL = os.getenv("KLUEBERT_ENDPOINT_URL", "")
KLUEBERT_API_KEY      = os.getenv("KLUEBERT_API_KEY", "")
