"""Project-relative configuration for the NIST-only application."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_nist_index_path = Path(os.getenv("NIST_CHROMA_PATH", "data/chroma_nist"))
CHROMA_PATH = str((PROJECT_ROOT / _nist_index_path).resolve())
