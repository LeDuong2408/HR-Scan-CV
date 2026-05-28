"""
Tool: CV → Markdown Converter

Chuyển PDF hoặc DOCX thành Markdown có cấu trúc heading.
Markdown là intermediate format giúp MarkdownHeaderTextSplitter
hiểu được ranh giới semantic (# Experience, # Skills, v.v.)

Chiến lược:
  PDF  → pymupdf4llm   (giữ được bold, heading, layout tốt nhất)
         fallback: pdfplumber + regex heuristic để detect headings
  DOCX → mammoth       (convert Word styles → Markdown headings chính xác)

Tại sao không dùng raw text như trước?
  Raw text: "EXPERIENCE\nFPT Software..."
  Markdown: "# EXPERIENCE\n\nFPT Software..."
  → MarkdownHeaderTextSplitter cần ký hiệu # để split đúng section
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex detect dòng heading từ raw text (dùng khi fallback)
_HEADING_PATTERNS = re.compile(
    r"^("
    r"EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|CAREER|"
    r"EDUCATION|ACADEMIC|QUALIFICATION|"
    r"SKILLS|TECHNICAL SKILLS|COMPETENCIES|"
    r"PROJECTS|PERSONAL PROJECTS|"
    r"CERTIFICATIONS?|CERTIFICATES?|LICENSES?|"
    r"SUMMARY|OBJECTIVE|PROFILE|ABOUT|"
    r"LANGUAGES?|INTERESTS?|HOBBIES|ACTIVITIES|"
    r"AWARDS?|ACHIEVEMENTS?|HONORS?|"
    r"PUBLICATIONS?|REFERENCES?"
    r")[\s:]*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class MarkdownResult:
    markdown:  str
    method:    str   # "pymupdf" | "pymupdf_fallback" | "mammoth" | "doc_converted"
    page_count: int


# ── Public API ─────────────────────────────────────────────────────────────

def to_markdown(file_path: str) -> MarkdownResult:
    """
    Convert PDF hoặc DOCX sang Markdown có heading.

    Args:
        file_path: Đường dẫn đến file CV

    Returns:
        MarkdownResult với markdown string và metadata
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _pdf_to_markdown(path)
    elif suffix in {".docx", ".doc"}:
        return _docx_to_markdown(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Supported: .pdf .docx .doc")


# ── PDF → Markdown ─────────────────────────────────────────────────────────

def _pdf_to_markdown(path: Path) -> MarkdownResult:
    """Dùng pymupdf4llm để extract Markdown từ PDF."""
    try:
        import pymupdf4llm
        import pymupdf

        doc = pymupdf.open(str(path))
        page_count = len(doc)
        doc.close()

        md = pymupdf4llm.to_markdown(str(path))

        # pymupdf4llm đôi khi trả về heading với ##### (quá sâu)
        # Normalize về tối đa 2 levels (# và ##)
        md = _normalize_headings(md)
        md = _clean_markdown(md)

        logger.info(
            "PDF→MD via pymupdf4llm: %s (%d pages, %d chars)",
            path.name, page_count, len(md),
        )
        return MarkdownResult(markdown=md, method="pymupdf", page_count=page_count)

    except Exception as e:
        logger.warning("pymupdf4llm failed for %s: %s — using fallback", path.name, e)
        return _pdf_to_markdown_fallback(path)


def _pdf_to_markdown_fallback(path: Path) -> MarkdownResult:
    """
    Fallback: pdfplumber extract text + heuristic detect headings.
    Kém hơn pymupdf4llm nhưng vẫn tạo được heading structure.
    """
    import pdfplumber

    pages_text: list[str] = []
    page_count = 0

    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                pages_text.append(text.strip())

    raw_text  = "\n\n".join(pages_text)
    markdown  = _text_to_markdown_heuristic(raw_text)

    logger.info(
        "PDF→MD via fallback: %s (%d pages, %d chars)",
        path.name, page_count, len(markdown),
    )
    return MarkdownResult(
        markdown=markdown, method="pymupdf_fallback", page_count=page_count
    )


# ── DOCX → Markdown ────────────────────────────────────────────────────────

def _docx_to_markdown(path: Path) -> MarkdownResult:
    """
    Dùng mammoth để convert DOCX → Markdown.
    mammoth map Word styles (Heading 1, Heading 2) → # ##
    nên heading structure rất chính xác.
    """
    actual_path = path

    # Legacy .doc → convert to .docx trước
    if path.suffix.lower() == ".doc":
        actual_path = _doc_to_docx(path)

    try:
        import mammoth

        with open(str(actual_path), "rb") as fh:
            result = mammoth.convert_to_markdown(fh)

        md = result.value
        md = _normalize_headings(md)
        md = _clean_markdown(md)

        if result.messages:
            for msg in result.messages[:3]:  # Log tối đa 3 warnings
                logger.debug("mammoth warning: %s", msg)

        logger.info(
            "DOCX→MD via mammoth: %s (%d chars)",
            path.name, len(md),
        )
        return MarkdownResult(markdown=md, method="mammoth", page_count=0)

    except ImportError:
        raise RuntimeError("mammoth not installed: pip install mammoth")


def _doc_to_docx(path: Path) -> Path:
    """Convert .doc → .docx via LibreOffice headless."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "docx",
                 "--outdir", tmpdir, str(path)],
                check=True, capture_output=True, timeout=30,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"LibreOffice needed to convert .doc: {e}\n"
                "Install: apt install libreoffice"
            ) from e

        converted = Path(tmpdir) / path.with_suffix(".docx").name
        if not converted.exists():
            raise RuntimeError(f"LibreOffice produced no output for {path.name}")

        # Copy ra ngoài tmpdir để dùng sau khi context manager exit
        dest = path.with_suffix(".docx")
        import shutil
        shutil.copy(str(converted), str(dest))
        return dest


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalize_headings(md: str) -> str:
    # 1. Promote whitelist tags to Level 1 Headers (#)
    headers_whitelist = [
        "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "SKILLS",
        "WORK EXPERIENCE", "EXPERIENCE", "EDUCATION", "CERTIFICATIONS", "PROJECTS", "R&D PROJECTS", "SELECTED PROJECTS"
    ]
    whitelist_pattern = rf"__({'|'.join(headers_whitelist)})__"
    md = re.sub(whitelist_pattern, lambda m: f"\n# {m.group(1)}\n", md)
    md = re.sub(r"__(.*?)__", r"\1", md)

    # 2. Demote deep headers (###, ####, etc.) to Level 2 Headers (##)
    lines = md.splitlines()
    result = []
    for line in lines:
        if line.startswith("###"):
            line = "## " + line.lstrip("#").strip()
        result.append(line)

    return "\n".join(result)

def _clean_markdown(md: str) -> str:
    """
    Xóa artifacts không cần thiết:
    - Nhiều dòng trống liên tiếp → tối đa 2
    - Dòng chỉ có dashes/underscores (thường là separator trong PDF)
    - Trailing whitespace
    """
    # Xóa dòng separator
    md = re.sub(r"^\s*[-_=]{3,}\s*$", "", md, flags=re.MULTILINE)
    # Collapse multiple blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Trailing whitespace per line
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return md.strip()


def _text_to_markdown_heuristic(text: str) -> str:
    """
    Convert raw text → Markdown bằng heuristic:
    Dòng khớp HEADING_PATTERNS → thêm # prefix.
    """
    lines  = text.splitlines()
    result = []

    for line in lines:
        stripped = line.strip()
        if _HEADING_PATTERNS.match(stripped):
            result.append(f"\n# {stripped.title()}\n")
        else:
            result.append(line)

    md = "\n".join(result)
    return _clean_markdown(md)