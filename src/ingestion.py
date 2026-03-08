"""
ingestion.py — Fast PDF text extraction + chunking.

Extraction strategy (per page):
  1. PyMuPDF (fitz) text mode   — fast, handles most digital PDFs
  2. PyMuPDF "blocks" mode      — fallback within PyMuPDF for tricky layouts
  3. OCR (pdf2image+tesseract)  — fallback for completely scanned/image pages.
                                  Runs at a lower DPI (100) to ensure it doesn't
                                  take hours or crash on large PDFs.
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(__file__))

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_PAGE_CHARS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import fitz  # PyMuPDF


# ── Text helpers ──────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, remove control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _mupdf_blocks_text(page: fitz.Page) -> str:
    """
    Extract text using PyMuPDF's structured block mode.
    Works better than plain 'text' mode for some multi-column layouts.
    """
    blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no, block_type)
    lines = [b[4] for b in sorted(blocks, key=lambda b: (b[1], b[0])) if b[6] == 0]
    return "\n".join(lines)


# ── Fallback 2: OCR ──────────────────────────────────────────────────────────

def _extract_page_ocr(pdf_path: str, page_index: int, dpi: int = 100) -> str:
    """
    OCR a single page. 
    dpi=100 is used instead of 200 to prevent 'Estimating resolution' errors
    and significantly speed up the extraction of large scanned PDFs.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(
            pdf_path, dpi=dpi,
            first_page=page_index + 1,
            last_page=page_index + 1,
        )
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception as e:
        print(f"[OCR] Page {page_index + 1} error: {e}")
    return ""


# ── Main ingestion entry-point ────────────────────────────────────────────────

def ingest_pdf(
    file_path: str,
    pdf_name: str = "",
) -> list[Document]:
    """
    Load a PDF and return a list of chunked LangChain Documents.

    Parameters
    ----------
    file_path : str
        Absolute path to the PDF file.
    pdf_name : str
        Display name stored in document metadata (defaults to filename).

    Each returned Document carries:
        metadata["source"]  — pdf_name or filename
        metadata["page"]    — 1-indexed page number
    """
    source_label = pdf_name or os.path.basename(file_path)

    # ── Stage 1 & 2: PyMuPDF (fast — whole doc in one pass) ──────────────────
    try:
        doc = fitz.open(file_path)
        num_pages = len(doc)
        raw_pages: list[tuple[int, str]] = []

        for i, page in enumerate(doc):
            text = page.get_text("text")
            if len(text.strip()) < MIN_PAGE_CHARS:
                # Try block-mode extraction as a second attempt within PyMuPDF
                text = _mupdf_blocks_text(page)
            raw_pages.append((i, text))

        doc.close()
    except Exception as e:
        print(f"[PyMuPDF] Failed to open PDF: {e}")
        return []

    if not raw_pages:
        print("[Ingestion] PyMuPDF returned no pages — aborting.")
        return []

    # ── Stage 3: OCR — automatically triggered for empty/sparse pages ─────────
    documents: list[Document] = []
    ocr_count = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_page(page_idx: int, raw_text: str) -> tuple[int, str, bool]:
        text = _clean_text(raw_text)
        used_ocr = False

        if len(text) < MIN_PAGE_CHARS:
            used_ocr = True
            print(f"[OCR] Running fast-OCR on page {page_idx + 1} …")
            ocr_text = _clean_text(_extract_page_ocr(file_path, page_idx, dpi=100))
            if len(ocr_text) > len(text):
                text = ocr_text
                
        return page_idx, text, used_ocr

    results = []
    # Use max_workers=os.cpu_count() or a reasonable number to parallelize OCR
    with ThreadPoolExecutor(max_workers=max(1, int((os.cpu_count() or 4) * 0.8))) as executor:
        futures = {executor.submit(process_page, idx, text): idx for idx, text in raw_pages}
        for future in as_completed(futures):
            results.append(future.result())

    # Sort results to maintain original page order
    results.sort(key=lambda x: x[0])

    for page_idx, text, used_ocr in results:
        if used_ocr:
            ocr_count += 1
        
        if text:
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": source_label,
                    "page": page_idx + 1,   # 1-indexed for display
                },
            ))

    if ocr_count:
        print(f"[Ingestion] Fast-OCR used for {ocr_count} page(s).")

    if not documents:
        print("[Ingestion] No text extracted from PDF.")
        return []

    # ── Chunk ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(
        f"[Ingestion] {num_pages} page(s) → {len(documents)} with text "
        f"→ {len(chunks)} chunks."
    )
    return chunks
