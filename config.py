from pathlib import Path

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCES_DIR = DATA_DIR / "references"

CHROMA_DIR = ROOT_DIR / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-m3"

OPENAI_MODEL = "gpt-4o"
