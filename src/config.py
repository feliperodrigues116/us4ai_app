"""Project-relative configuration for the NIST-only application."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_nist_index_path = Path(os.getenv("NIST_CHROMA_PATH", "data/chroma_nist"))
CHROMA_PATH = str((PROJECT_ROOT / _nist_index_path).resolve())

NIST_COLLECTION_NAME = "us4ai_nist_playbook"
NIST_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
NIST_EMBEDDING_MAX_LENGTH = 512
NIST_DOCUMENT_EMBEDDING_TYPE = "default"
NIST_RERANKER_MODEL = "BAAI/bge-reranker-base"
NIST_MODEL_THREADS = 2
