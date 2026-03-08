"""
embedding.py — HuggingFace embedding model initialization.

Swap the default model by changing DEFAULT_EMBEDDING_MODEL in config.py,
or pass `model_name` explicitly to get_embedding_model().
"""

import os
import sys

# Ensure src/ is on the path when this module is imported directly
sys.path.insert(0, os.path.dirname(__file__))

from config import EMBEDDING_CACHE_DIR, DEFAULT_EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings instance.
    Model weights are cached in usefulcomponents/embedding_cache/
    so subsequent runs don't re-download.
    """
    # Point HF to the cache directory
    os.environ["HF_HOME"] = EMBEDDING_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = EMBEDDING_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = EMBEDDING_CACHE_DIR

    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=EMBEDDING_CACHE_DIR,
    )
