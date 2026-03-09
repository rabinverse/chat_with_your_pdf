"""

Each PDF gets its own persist directory:
    usefulcomponents/vectorstores/<pdf_name_slug>/

Helpers:
    list_vectorstores()         → list all processed PDFs
    delete_vectorstore(name)    → remove a named store
    get_vectorstore(pdf_name)   → load or create a store for a PDF
"""

import os
import sys
import re
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from config import VECTORSTORE_BASE_DIR
from embedding_model import get_embedding_model

from langchain_community.vectorstores import Chroma


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert a filename to a safe directory name."""
    name = os.path.splitext(name)[0]          # strip extension
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)      # keep alphanum, dash, underscore
    name = re.sub(r"[\s_]+", "_", name)       # spaces → underscores
    name = re.sub(r"-+", "-", name)
    return name or "unnamed_pdf"


def _store_path(pdf_name: str) -> str:
    slug = _slugify(pdf_name)
    return os.path.join(VECTORSTORE_BASE_DIR, slug)


def list_vectorstores() -> list[str]:
    """
    Returns a list of PDF names (slugs) that have been processed and stored.
    Each entry corresponds to a sub-folder in usefulcomponents/vectorstores/.
    """
    if not os.path.isdir(VECTORSTORE_BASE_DIR):
        return []
    return [
        d for d in os.listdir(VECTORSTORE_BASE_DIR)
        if os.path.isdir(os.path.join(VECTORSTORE_BASE_DIR, d))
    ]


def delete_vectorstore(pdf_name: str) -> bool:
    """
    Delete the ChromaDB store for a given PDF name/slug.
    Returns True if deleted, False if it didn't exist.
    """
    path = _store_path(pdf_name)
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"[VectorStore] Deleted store: {path}")
        return True
    return False


# ── Chroma ────────────────────────────────────────────────────────────────────

def get_chroma_store(pdf_name: str, documents=None, embedding_model=None):
    """
    Load or create a ChromaDB store named after the PDF.

    - If `documents` are provided and the store already exists → append.
    - If `documents` are provided and no store exists → create.
    - If no `documents` → load existing store (raises if missing).
    """
    db_path = _store_path(pdf_name)
    emb = embedding_model or get_embedding_model()

    store_exists = os.path.isdir(db_path) and os.listdir(db_path)

    if documents and store_exists:
        db = Chroma(persist_directory=db_path, embedding_function=emb)
        db.add_documents(documents)
        print(f"[ChromaDB] Appended {len(documents)} chunks to: {db_path}")
        return db

    elif documents and not store_exists:
        os.makedirs(db_path, exist_ok=True)
        db = Chroma.from_documents(
            documents=documents,
            embedding=emb,
            persist_directory=db_path,
        )
        print(f"[ChromaDB] Created new store at: {db_path}")
        return db

    elif not documents and store_exists:
        return Chroma(persist_directory=db_path, embedding_function=emb)

    else:
        raise ValueError(
            f"No ChromaDB store found for '{pdf_name}' and no documents provided. "
            "Please process the PDF first."
        )


# ── Pinecone ──────────────────────────────────────────────────────────────────

# def get_pinecone_store(index_name: str = "pdf-vector-index", documents=None,
#                        dimension: int = 384, metric: str = "cosine",
#                        embedding_model=None):
#     """
#     Load or create a Pinecone vector store.
#     Requires PINECONE_API_KEY in environment variables.
#     """
#     from langchain_pinecone import PineconeVectorStore
#     from pinecone import Pinecone, ServerlessSpec

#     emb = embedding_model or get_embedding_model()
#     api_key = os.environ.get("PINECONE_API_KEY")
#     if not api_key:
#         raise ValueError("PINECONE_API_KEY is not set in environment variables.")

#     pc = Pinecone(api_key=api_key)
#     existing = [idx["name"] for idx in pc.list_indexes()]
#     if index_name not in existing:
#         pc.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric=metric,
#             spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#         )

#     if documents:
#         return PineconeVectorStore.from_documents(
#             documents=documents,
#             embedding=emb,
#             index_name=index_name,
#         )
#     return PineconeVectorStore(index_name=index_name, embedding=emb)


# ── Public factory ─────────────────────────────────────────────────────────────

def get_vectorstore(pdf_name: str, documents=None, db_type: str = "chroma",
                    embedding_model=None, **kwargs):
    """
    Factory: returns a vector store for the given PDF.

    Args:
        pdf_name:        Original PDF filename (used to name the store).
        documents:       Chunked LangChain Documents to upsert (optional).
        db_type:         "chroma" (default) or "pinecone".
        embedding_model: Pre-loaded embedding model (optional; avoids re-loading).
        **kwargs:        Extra args forwarded to the specific store (e.g., index_name).
    """
    db_type = db_type.lower()
    if db_type == "chroma":
        return get_chroma_store(pdf_name, documents=documents, embedding_model=embedding_model)
    # elif db_type == "pinecone":
    #     return get_pinecone_store(
    #         index_name=kwargs.get("index_name", "pdf-vector-index"),
    #         documents=documents,
    #         embedding_model=embedding_model,
    #     )
    else:
        raise ValueError(f"Unsupported vector store type: '{db_type}'. Use 'chroma' or 'pinecone'.")
