"""
Tool: PDF text extractor with OCR fallback.

Strategy:
  1. Try pdfplumber (native PDF text layer) — fast, accurate.
  2. If text < MIN_CHARS (likely a scanned image), fallback to pytesseract OCR.
  3. Return extraction method alongside text so caller knows confidence level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_CHARS_NATIVE = 150  # Below this → assume scan, use OCR


@dataclass
class ExtractionResult:
    text: str
    method: str  # "native" | "ocr" | "hybrid"
    page_count: int
    char_count: int


def extract_pdf_text(file_path: str) -> ExtractionResult:
    """
    Extract text from a PDF file.
    Automatically falls back to OCR if native extraction yields too little text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected .pdf, got: {path.suffix}")

    # --- Attempt 1: Native PDF text layer ---
    try:
        import pdfplumber
        native_text, page_count = _extract_native(path)

        if len(native_text.strip()) >= MIN_CHARS_NATIVE:
            logger.info(
                "PDF extracted natively: %s (%d chars, %d pages)",
                path.name, len(native_text), page_count,
            )
            return ExtractionResult(
                text=native_text,
                method="native",
                page_count=page_count,
                char_count=len(native_text),
            )

        logger.warning(
            "Native extraction too short (%d chars) for %s — switching to OCR",
            len(native_text), path.name,
        )

    except Exception as e:
        logger.warning("pdfplumber failed for %s: %s — trying OCR", path.name, e)
        page_count = 0

    # --- Attempt 2: OCR via pytesseract ---
    try:
        ocr_text, page_count = _extract_ocr(path)
        logger.info(
            "PDF extracted via OCR: %s (%d chars, %d pages)",
            path.name, len(ocr_text), page_count,
        )
        return ExtractionResult(
            text=ocr_text,
            method="ocr",
            page_count=page_count,
            char_count=len(ocr_text),
        )

    except ImportError:
        logger.error(
            "pytesseract not installed. Install: pip install pytesseract pillow "
            "and ensure Tesseract binary is on PATH."
        )
        raise

    except Exception as e:
        logger.error("OCR also failed for %s: %s", path.name, e)
        raise RuntimeError(
            f"Failed to extract text from {path.name}: {e}"
        ) from e


def _extract_native(path: Path) -> tuple[str, int]:
    """Extract text using pdfplumber (native PDF layer)."""
    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                pages_text.append(text.strip())

    return "\n\n".join(pages_text), page_count


def _extract_ocr(path: Path) -> tuple[str, int]:
    """
    Convert PDF pages to images then run Tesseract OCR.
    Requires: pip install pytesseract pillow pdf2image
    System:   sudo apt install tesseract-ocr poppler-utils
              (Windows: install Tesseract + poppler binaries)
    """
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(str(path), dpi=300)
    pages_text: list[str] = []

    for img in images:
        # Use English + Vietnamese language packs if available
        try:
            text = pytesseract.image_to_string(img, lang="eng+vie")
        except pytesseract.TesseractError:
            text = pytesseract.image_to_string(img, lang="eng")
        pages_text.append(text.strip())

    return "\n\n".join(pages_text), len(images)