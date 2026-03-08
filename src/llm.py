"""
llm.py — LLM factory for the Chat-with-PDF app.

Supported providers:
  - "groq"          → Groq cloud API (fast inference, needs GROQ_API_KEY)
  - "huggingface"   → HuggingFace Inference Endpoint (needs HF_TOKEN)

To add a new provider: add a new `get_<provider>_model()` function
and register it in get_llm().
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER

load_dotenv()


def get_groq_model(model_name: str = DEFAULT_LLM_MODEL):
    """
    Returns a ChatGroq LLM instance.
    Requires GROQ_API_KEY in the environment.
    """
    from langchain_groq import ChatGroq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")
    return ChatGroq(model=model_name, api_key=api_key)


def get_hf_model(repo_id: str, temperature: float = 0.5, max_new_tokens: int = 512):
    """
    Returns a HuggingFaceEndpoint LLM.
    Requires HF_TOKEN in the environment.
    """
    from langchain_huggingface import HuggingFaceEndpoint
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set in environment variables.")
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=temperature,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=max_new_tokens,
    )


def get_llm(provider: str = DEFAULT_LLM_PROVIDER, model_name: str = DEFAULT_LLM_MODEL):
    """
    Factory: returns an LLM based on the selected provider.

    Args:
        provider:   "groq" or "huggingface"
        model_name: Model identifier (Groq model name or HF repo ID)
    """
    provider = provider.lower()
    if provider == "groq":
        return get_groq_model(model_name)
    elif provider == "huggingface":
        return get_hf_model(model_name)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Use 'groq' or 'huggingface'."
        )
