"""
prompt.py — Prompt template for PDF Q&A with citation guidance.

The prompt instructs the LLM to:
  1. Answer only from the provided context.
  2. Give a well-reasoned, comprehensive answer.
  3. Reference the relevant content directly.
"""

from langchain_core.prompts import PromptTemplate

CUSTOM_PROMPT_TEMPLATE = """You are an expert assistant helping users understand their PDF documents.

Use ONLY the following context excerpts to answer the user's question.
If the answer is not present in the context, say clearly: "I could not find this information in the provided document."
Do NOT make up information or answer from general knowledge.

Be comprehensive and specific. If the context contains partial information, share what is available.
Provide a clear, well-structured answer that directly addresses the question.

Context:
{context}

Question: {question}

Answer:"""


def get_prompt_template() -> PromptTemplate:
    """Returns the LangChain PromptTemplate for Q&A."""
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
