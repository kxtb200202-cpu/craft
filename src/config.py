"""Configuration of CRAFT: LLM, embedding and RAGFlow settings.

Model and service credentials are loaded from .env. The constraint knowledge base
lives in the RAGFlow service - by default the dataset of the P7 disaster-relief
task set (constraints_p7), switched with the RAGFLOW_DATASET environment variable.
No constraint file is read from disk at retrieval time."""

import os
from pathlib import Path

from dotenv import load_dotenv

_craft_dir = Path(__file__).resolve().parent.parent
_env_path = _craft_dir / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

LLM_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")

RAGFLOW_BASE_URL = os.environ.get("RAGFLOW_BASE_URL", "http://127.0.0.1:9380")
RAGFLOW_API_KEY = os.environ.get("RAGFLOW_API_KEY", "")
RAGFLOW_DATASET = os.environ.get("RAGFLOW_DATASET")

CORPUS_TOP_K = 200
DEFAULT_TOP_K = 15
HYBRID_WEIGHTS = {
    "semantic": 0.7,
    "keyword": 0.3,
}


def ragflow_dataset_name() -> str:
    return RAGFLOW_DATASET
