"""
embedding.py — HuggingFace embedding model initialization.

Swap the default model by changing DEFAULT_EMBEDDING_MODEL in config.py,
or pass `model_name` explicitly to get_embedding_model().

AUTO-DETECTION:
- First run: Downloads model automatically
- Subsequent runs: Uses cached model (offline)
"""

import os
import sys

# Ensure src/ is on the path when this module is imported directly
sys.path.insert(0, os.path.dirname(__file__))

from config import EMBEDDING_CACHE_DIR, DEFAULT_EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings


def _is_model_cached(model_name: str) -> bool:
    """Check if embedding model is already cached locally."""
    # Convert model name to cache folder format
    # e.g., "sentence-transformers/all-MiniLM-L6-v2" → "models--sentence-transformers--all-MiniLM-L6-v2"
    cache_model_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = os.path.join(EMBEDDING_CACHE_DIR, cache_model_name)

    # Check if cache folder exists and has content
    return os.path.exists(model_cache_path) and os.listdir(model_cache_path)


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL, offline: bool = None
) -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings instance.

    Args:
        model_name: HuggingFace model identifier
        offline: If None (default), auto-detects. If True, offline only. If False, allows download.

    """
    if offline is None:
        model_is_cached = _is_model_cached(model_name)
        offline = model_is_cached  # Use offline ONLY if cached

    # Point HF to the cache directory
    os.environ["HF_HOME"] = EMBEDDING_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = EMBEDDING_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = EMBEDDING_CACHE_DIR

    # Build model_kwargs with local_files_only if offline
    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": offline,  # Force offline mode at transformer level
    }

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        # Remove offline mode to allow downloads
        os.environ.pop("HF_HUB_OFFLINE", None)

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=EMBEDDING_CACHE_DIR,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
    except RuntimeError as e:
        if "Cannot send a request" in str(e) or "client has been closed" in str(e):
            # HTTP client closed error - try with local files only
            print(f"[Embedding] HTTP error detected, falling back to cached model...")
            model_kwargs["local_files_only"] = True
            os.environ["HF_HUB_OFFLINE"] = "1"

            return HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=EMBEDDING_CACHE_DIR,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            raise
    except Exception as e:
        print(f"[Embedding] Error loading model: {e}")
        # If all else fails, try forcing offline mode
        print(
            f"[Embedding] Attempting to load cached model with local_files_only=True..."
        )
        model_kwargs["local_files_only"] = True
        os.environ["HF_HUB_OFFLINE"] = "1"

        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=EMBEDDING_CACHE_DIR,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
