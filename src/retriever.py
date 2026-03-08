"""
retriever.py — Two-step retrieval: instant page numbers then full AI answer.

The two-step design lets the Streamlit UI show page numbers immediately
(fast similarity search) while the LLM generates the answer in the background.
This is done to let user always sees *something* useful without waiting.

Usage:
    pages = retrieve_pages(query, vector_store)       # fast: show immediately
    answer = get_answer(query, vector_store, llm)     # slower: show when ready
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import RETRIEVER_TOP_K
from prompt import get_prompt_template

from langchain_core.vectorstores import VectorStore
from langchain.chains import RetrievalQA
from langchain.retrievers.merger_retriever import MergerRetriever


def _get_combined_retriever(vector_stores: list[VectorStore], k: int):
    retrievers = [
        vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
        for vs in vector_stores
    ]
    if len(retrievers) == 1:
        return retrievers[0]
    return MergerRetriever(retrievers=retrievers)


def _format_source_pages(docs) -> list[str]:
    unique_sources = set()
    for doc in docs:
        page = doc.metadata.get("page")
        if page is not None:
            source = doc.metadata.get("source", "Unknown")
            unique_sources.add((source, page))

    sorted_sources = sorted(list(unique_sources), key=lambda x: (x[0], x[1]))

    formatted = []
    for source, page in sorted_sources:
        clean_source = source.replace(".pdf", "")
        # Shorten to 15 characters if it's too long
        short_source = (
            clean_source[:15] + ".." if len(clean_source) > 15 else clean_source
        )
        formatted.append(f"{short_source} pg.{page}")

    return formatted


def retrieve_pages(
    query: str, vector_stores: list[VectorStore], k: int = RETRIEVER_TOP_K
) -> list[str]:
    """
    Perform a  similarity search and return the relevant page numbers, formatted with source.

    Returns:
        Formatted list of strings combining source name and page number.
    """
    try:
        if not vector_stores:
            return []

        retriever = _get_combined_retriever(vector_stores, k)
        docs = retriever.invoke(query)
        return _format_source_pages(docs)
    except Exception as e:
        print(f"[Retriever] Page retrieval error: {e}")
        return []


def get_answer(
    query: str,
    vector_stores: list[VectorStore],
    llm,
    k: int = RETRIEVER_TOP_K,
) -> tuple[str, list[str]]:
    """
    Run the full RAG Q&A chain and return the answer and citation pages.

    Returns:
        (answer_text, sorted_page_list)
        On LLM error, returns (error_message, []) is returned so the caller can handle the situation safely.
    """
    try:
        if not vector_stores:
            return "No documents selected.", []

        retriever = _get_combined_retriever(vector_stores, k)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": get_prompt_template()},
        )
        response = chain.invoke({"query": query})
        answer = response.get("result", "No answer generated.")

        source_docs = response.get("source_documents", [])
        pages = _format_source_pages(source_docs)

        return answer, pages

    except Exception as e:
        error_msg = f"⚠️ LLM error: {str(e)}"
        print(f"[Retriever] {error_msg}")
        return error_msg, []
