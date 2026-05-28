"""
Agent 1 (v2): CV Parser Agent

Thay đổi so với v1:
  No LLM: không gọi LLM để extract structured JSON
  Markdown: convert CV sang Markdown giữ heading structure
  Chunking: split theo H1 heading + recursive split nếu quá dài
  ChromaDB: embed + store với cv_id metadata để Agent 2 query
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.cv_chunker import ChunkingResult, chunk_and_store, delete_cv_chunks
from tools.cv_to_markdown import MarkdownResult, to_markdown

logger = logging.getLogger(__name__)


@dataclass
class ParsedCV:
    """Output của CVParserAgent."""
    cv_id:          str
    file_name:      str
    candidate_name: str       = "Unknown"
    email:          str       = ""
    markdown:       str       = ""
    chunk_count:    int       = 0
    sections:       list[str] = field(default_factory=list)
    parse_method:   str       = ""
    warnings:       list[str] = field(default_factory=list)


class CVParserAgent:
    """
    Agent 1: Parse CV file → Markdown → Chunks → ChromaDB.

    Usage:
        agent = CVParserAgent()
        parsed = agent.parse("cv.pdf")
        print(parsed.cv_id)      # dùng cho Agent 2 query ChromaDB
        print(parsed.sections)   # ["Experience", "Skills", "Education"]
    """

    def __init__(self, reprocess: bool = False) -> None:
        self.reprocess = reprocess

    def parse(self, file_path: str) -> ParsedCV:
        path = Path(file_path)
        logger.info("Parsing CV: %s", path.name)

        cv_id = _file_to_cv_id(path)

        if self.reprocess:
            deleted = delete_cv_chunks(cv_id)
            if deleted:
                logger.info("Reprocessing %s: deleted %d old chunks", path.name, deleted)

        # Bước 1: Convert to Markdown
        md_result: MarkdownResult = to_markdown(str(path))

        # Bước 2: Chunk + Embed + Store ChromaDB
        chunk_result: ChunkingResult = chunk_and_store(
            markdown  = md_result.markdown,
            file_name = path.name,
            cv_id     = cv_id,
        )

        # Bước 3: Extract basic info bằng heuristic (không cần LLM)
        name  = _extract_name_heuristic(md_result.markdown)
        email = _extract_email(md_result.markdown)

        result = ParsedCV(
            cv_id          = cv_id,
            file_name      = path.name,
            candidate_name = name,
            email          = email,
            markdown       = md_result.markdown,
            chunk_count    = chunk_result.chunk_count,
            sections       = chunk_result.sections,
            parse_method   = md_result.method,
        )

        logger.info(
            "Parsed: %s | cv_id=%s | %d chunks | sections=%s | method=%s",
            path.name, cv_id, chunk_result.chunk_count,
            chunk_result.sections, md_result.method,
        )
        return result

    def parse_batch(self, file_paths: list[str]) -> list[ParsedCV]:
        """
        Parse nhiều CVs. Không cần sleep vì không gọi LLM.
        Failed files → ParsedCV với warning, batch không crash.
        """
        results: list[ParsedCV] = []
        for i, fp in enumerate(file_paths, 1):
            logger.info("Batch parse %d/%d: %s", i, len(file_paths), Path(fp).name)
            try:
                results.append(self.parse(fp))
            except Exception as e:
                logger.error("Failed to parse %s: %s", fp, e)
                results.append(_fallback_parsed(fp, str(e)))
        return results

# ── Helpers ────────────────────────────────────────────────────────────────

def _file_to_cv_id(path: Path) -> str:
    """Sinh cv_id idempotent từ filename — cùng file → cùng cv_id."""
    import hashlib
    h = hashlib.md5(path.name.encode()).hexdigest()[:12]
    return f"cv_{h}"


def _extract_email(markdown: str) -> str:
    match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        markdown,
    )
    return match.group(0) if match else ""

def _extract_name_heuristic(markdown: str) -> str:
    """
    Extract tên từ 10 dòng đầu CV bằng heuristic.
    Không dùng LLM — tên thường là dòng đầu tiên có 2-5 từ viết hoa.
    """
    _SKIP_SECTIONS = {
        "experience", "education", "skills", "summary",
        "profile", "about", "projects", "certifications",
        "objective", "contact", "languages", "awards",
    }
    lines = [l.strip() for l in markdown.splitlines() if l.strip()]

    for line in lines[:10]:
        if re.match(r"^[\d\s\-\+\(\)@\.:/]+$", line):
            continue
        if "@" in line or "http" in line.lower():
            continue
        if len(line) > 60:
            continue

        # Heading → lấy content sau #
        if line.startswith("#"):
            name = line.lstrip("#").strip()
            if name.lower() not in _SKIP_SECTIONS and len(name) > 3:
                return name
        else:
            words = line.split()
            if 2 <= len(words) <= 5 and all(
                w[0].isupper() for w in words if w and w[0].isalpha()
            ):
                return line
            if line.isupper() and 2 <= len(words) <= 5:
                return line.title()

    return "Unknown"


def _fallback_parsed(file_path: str, error: str) -> ParsedCV:
    path = Path(file_path)
    return ParsedCV(
        cv_id          = f"cv_failed_{path.stem[:8]}",
        file_name      = path.name,
        candidate_name = f"[PARSE ERROR] {path.name}",
        warnings       = [f"Parsing failed: {error}"],
        parse_method   = "failed",
    )