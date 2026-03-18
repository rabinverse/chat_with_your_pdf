"""
config.py — Central configuration for the Chat-with-PDF app.

All paths and default model settings live here. Edit this file to
change where artifacts are stored or which models are used by default.
"""

import os

# ── Project root (one level above src/) ───────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── usefulcomponents/ — all runtime artifacts live here ───────────────────────
USEFUL_COMPONENTS_DIR = os.path.join(PROJECT_ROOT, "usefulcomponents")

# Embedding model weights are cached here (set as HF_HOME)
EMBEDDING_CACHE_DIR = os.path.join(USEFUL_COMPONENTS_DIR, "embedding_cache")

# Each processed PDF gets its own sub-folder here
VECTORSTORE_BASE_DIR = os.path.join(USEFUL_COMPONENTS_DIR, "vectorstores")

# Create directories on import so they always exist
for _dir in (USEFUL_COMPONENTS_DIR, EMBEDDING_CACHE_DIR, VECTORSTORE_BASE_DIR):
    os.makedirs(_dir, exist_ok=True)

# ── Default model settings  ─────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_LLM_PROVIDER = "groq"
DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_Gemini_LLM_MODEL = "gemini-2.5-flash"

# ── Retrieval settings ────────────────────────────────────────────────────────
RETRIEVER_TOP_K = 8  # how many chunks to retrieve per query

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ── OCR / Ingestion settings ──────────────────────────────────────────────────
# If a page has fewer chars than this threshold we fall back to pdfplumber/OCR
MIN_PAGE_CHARS = 50


# ── model cofig ──────────────────────────────────────────────────
MODEL_CONFIG = {
    "Groq": [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
        "openai/gpt-oss-20b",
        "groq/compound",
    ],
    "HuggingFace_models": [
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ],
    "Google": ["gemini-2.5-flash"],
}
