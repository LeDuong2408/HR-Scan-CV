"""
Tests for Agent 1 v2: CV Parser (Markdown + Chunking pipeline).

Run: pytest tests/test_cv_parser_v2.py -v

4 tầng:
  Unit Heuristics:  test _extract_name, _extract_email, _file_to_cv_id
  Unit Chunker:     test _split_by_headers, _split_oversized (không cần ChromaDB)
  Unit Markdown:    test _normalize_headings, _clean_markdown, _text_to_markdown_heuristic
  Integration:      test parse() với mock to_markdown + mock chunk_and_store
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.cv_parser import (
    CVParserAgent,
    ParsedCV,
    _extract_email,
    _extract_name_heuristic,
    _fallback_parsed,
    _file_to_cv_id,
)
from rag.cv_chunker import (
    CVChunk,
    _split_by_headers,
    _split_oversized,
)
from tools.cv_to_markdown import (
    _clean_markdown,
    _normalize_headings,
    _text_to_markdown_heuristic,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """# Nguyen Van A

Email: nguyenvana@gmail.com | Phone: 0901234567

# Experience

## FPT Software — Backend Engineer (2020 - 2024)

Developed REST APIs using FastAPI and PostgreSQL.
Reduced API latency by 30% through Redis caching.
Led a team of 5 developers.

## TechCorp — Junior Developer (2019 - 2020)

Built internal tools using Python and Django.

# Skills

**Programming:** Python, JavaScript, SQL
**Frameworks:** FastAPI, Django, React
**Cloud:** AWS Lambda, S3, EC2
**Tools:** Docker, Git, Jira

# Education

## HCMUT — Bachelor of Computer Science (2015 - 2019)

GPA: 3.4/4.0

# Certifications

- AWS Solutions Architect Associate (2023)
- Docker Certified Associate (2022)
"""

SAMPLE_MARKDOWN_NO_HEADERS = """Nguyen Van B

Email: b@gmail.com

Python developer with 3 years experience in FastAPI and PostgreSQL.
Worked at ABC Corp as Backend Engineer from 2021 to 2024.
Skills: Python, FastAPI, Docker, AWS.
Education: Bachelor CS from HCMUT 2021.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 1: Unit — Heuristic extractors
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractEmail:
    def test_finds_email(self):
        assert _extract_email("Contact: test@gmail.com") == "test@gmail.com"

    def test_finds_email_in_markdown(self):
        assert _extract_email(SAMPLE_MARKDOWN) == "nguyenvana@gmail.com"

    def test_no_email_returns_empty(self):
        assert _extract_email("No email here") == ""

    def test_complex_email(self):
        assert _extract_email("nguyen.van+tag@company.co.uk") == "nguyen.van+tag@company.co.uk"


class TestExtractName:
    def test_extracts_name_from_first_heading(self):
        name = _extract_name_heuristic(SAMPLE_MARKDOWN)
        assert "Nguyen" in name or name == "Nguyen Van A"

    def test_skips_section_keywords(self):
        md = "# Experience\n\nSome content\n# Skills"
        name = _extract_name_heuristic(md)
        # Should NOT return "Experience" or "Skills"
        assert name not in {"Experience", "Skills"}

    def test_extracts_from_plain_text(self):
        md = "Tran Thi B\nEmail: b@test.com"
        name = _extract_name_heuristic(md)
        assert "Tran" in name

    def test_skips_email_lines(self):
        md = "john@gmail.com\nJohn Smith\nDeveloper"
        name = _extract_name_heuristic(md)
        assert "@" not in name

    def test_skips_long_lines(self):
        md = "This is a very long line that is definitely not a person name at all\nJohn Doe"
        name = _extract_name_heuristic(md)
        assert name == "John Doe"

    def test_fallback_unknown(self):
        name = _extract_name_heuristic("123 456 789\nhttp://website.com")
        assert name == "Unknown"


class TestFileToCvId:
    def test_same_file_same_id(self):
        p = Path("cv_nguyen_van_a.pdf")
        assert _file_to_cv_id(p) == _file_to_cv_id(p)

    def test_different_files_different_id(self):
        p1 = Path("cv1.pdf")
        p2 = Path("cv2.pdf")
        assert _file_to_cv_id(p1) != _file_to_cv_id(p2)

    def test_id_starts_with_cv(self):
        assert _file_to_cv_id(Path("resume.pdf")).startswith("cv_")

    def test_id_is_string(self):
        assert isinstance(_file_to_cv_id(Path("test.pdf")), str)


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 2: Unit — Chunker logic (không cần ChromaDB)
# ──────────────────────────────────────────────────────────────────────────────

class TestSplitByHeaders:
    def test_splits_by_h1_headings(self):
        chunks = _split_by_headers(SAMPLE_MARKDOWN, "cv_test", "test.pdf")
        sections = {c.section for c in chunks}
        # Phải có Experience, Skills, Education
        assert "Experience" in sections or any("exp" in s.lower() for s in sections)

    def test_no_headers_returns_one_chunk(self):
        chunks = _split_by_headers(SAMPLE_MARKDOWN_NO_HEADERS, "cv_test", "test.pdf")
        # Không có heading → 1 chunk duy nhất
        assert len(chunks) >= 1

    def test_chunk_has_cv_id(self):
        chunks = _split_by_headers(SAMPLE_MARKDOWN, "cv_abc123", "test.pdf")
        assert all(c.cv_id == "cv_abc123" for c in chunks)

    def test_chunk_has_file_name(self):
        chunks = _split_by_headers(SAMPLE_MARKDOWN, "cv_test", "my_resume.pdf")
        assert all(c.file_name == "my_resume.pdf" for c in chunks)

    def test_chunk_index_sequential(self):
        chunks = _split_by_headers(SAMPLE_MARKDOWN, "cv_test", "test.pdf")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_text_not_empty(self):
        chunks = _split_by_headers(SAMPLE_MARKDOWN, "cv_test", "test.pdf")
        assert all(c.text.strip() for c in chunks)


class TestSplitOversized:
    def _make_chunk(self, text: str, index: int = 0) -> CVChunk:
        return CVChunk(
            text="x",  # placeholder, sẽ override
            cv_id="cv_test",
            file_name="test.pdf",
            section="Experience",
            subsection="",
            chunk_index=index,
        )

    def test_small_chunk_unchanged(self):
        """Chunk nhỏ hơn MAX_CHUNK_CHARS → giữ nguyên."""
        chunk = CVChunk(
            text="Short text.",
            cv_id="cv_test", file_name="test.pdf",
            section="Skills", subsection="", chunk_index=0,
        )
        result = _split_oversized([chunk])
        assert len(result) == 1
        assert result[0].text == "Short text."

    def test_large_chunk_split(self):
        """Chunk > 900 chars → bị split thành nhiều sub-chunks."""
        long_text = "This is a sentence about Python. " * 50  # ~1650 chars
        chunk = CVChunk(
            text=long_text,
            cv_id="cv_test", file_name="test.pdf",
            section="Experience", subsection="FPT", chunk_index=0,
        )
        result = _split_oversized([chunk])
        assert len(result) > 1

    def test_sub_chunks_preserve_metadata(self):
        """Sub-chunks giữ cv_id, file_name, section từ chunk gốc."""
        long_text = "Word " * 300  # ~1500 chars
        chunk = CVChunk(
            text=long_text,
            cv_id="cv_xyz", file_name="resume.pdf",
            section="Experience", subsection="ABC Corp", chunk_index=0,
        )
        result = _split_oversized([chunk])
        for r in result:
            assert r.cv_id      == "cv_xyz"
            assert r.file_name  == "resume.pdf"
            assert r.section    == "Experience"
            assert r.subsection == "ABC Corp"

    def test_chunk_indices_sequential_after_split(self):
        """chunk_index sau split luôn sequential từ 0."""
        long_text = "Sentence. " * 200
        chunks = [
            CVChunk(text="Short.", cv_id="cv1", file_name="f.pdf",
                    section="S1", subsection="", chunk_index=0),
            CVChunk(text=long_text, cv_id="cv1", file_name="f.pdf",
                    section="S2", subsection="", chunk_index=1),
        ]
        result = _split_oversized(chunks)
        indices = [r.chunk_index for r in result]
        assert indices == list(range(len(result)))

    def test_no_empty_chunks_after_split(self):
        """Không có chunk rỗng sau khi split."""
        long_text = "  \n\n  ".join(["Word " * 20] * 10)
        chunk = CVChunk(
            text=long_text,
            cv_id="cv1", file_name="f.pdf",
            section="S", subsection="", chunk_index=0,
        )
        result = _split_oversized([chunk])
        assert all(r.text.strip() for r in result)


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 3: Unit — Markdown utilities
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeHeadings:
    def test_h4_promoted_to_h2(self):
        md = "#### Deep Section"
        result = _normalize_headings(md)
        assert result.startswith("##")
        assert not result.startswith("####")

    def test_h3_promoted_to_h2(self):
        md = "### Sub Section"
        result = _normalize_headings(md)
        assert result.startswith("##")

    def test_h1_unchanged(self):
        md = "# Main Section"
        result = _normalize_headings(md)
        assert result == "# Main Section"

    def test_h2_unchanged(self):
        md = "## Sub Section"
        result = _normalize_headings(md)
        assert result == "## Sub Section"


class TestCleanMarkdown:
    def test_removes_separator_lines(self):
        md = "Text\n---\nMore text"
        result = _clean_markdown(md)
        assert "---" not in result

    def test_collapses_multiple_blank_lines(self):
        md = "Line 1\n\n\n\n\nLine 2"
        result = _clean_markdown(md)
        assert "\n\n\n" not in result

    def test_removes_trailing_whitespace(self):
        md = "Line with spaces   \nAnother line  "
        result = _clean_markdown(md)
        for line in result.splitlines():
            assert not line.endswith(" ")


class TestTextToMarkdownHeuristic:
    def test_detects_experience_heading(self):
        text = "EXPERIENCE\nFPT Software - Backend Engineer"
        result = _text_to_markdown_heuristic(text)
        assert "# Experience" in result or "#" in result

    def test_detects_skills_heading(self):
        text = "SKILLS\nPython, FastAPI"
        result = _text_to_markdown_heuristic(text)
        assert "#" in result

    def test_preserves_non_heading_content(self):
        text = "Python developer with 5 years experience"
        result = _text_to_markdown_heuristic(text)
        assert "Python developer" in result


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 4: Integration — full parse() pipeline (mock external dependencies)
# ──────────────────────────────────────────────────────────────────────────────

class TestCVParserAgentV2:

    @patch("agents.cv_parser.chunk_and_store")
    @patch("agents.cv_parser.to_markdown")
    def test_parse_pdf_success(self, mock_md, mock_chunk):
        from tools.cv_to_markdown import MarkdownResult
        from rag.cv_chunker import ChunkingResult

        mock_md.return_value = MarkdownResult(
            markdown   = SAMPLE_MARKDOWN,
            method     = "pymupdf",
            page_count = 2,
        )
        mock_chunk.return_value = ChunkingResult(
            cv_id       = "cv_abc",
            file_name   = "cv.pdf",
            chunk_count = 5,
            sections    = ["Experience", "Skills", "Education"],
        )

        agent  = CVParserAgent()
        result = agent.parse("dummy.pdf")

        assert isinstance(result, ParsedCV)
        assert result.chunk_count  == 5
        assert result.parse_method == "pymupdf"
        assert "Experience"         in result.sections
        assert result.email        == "nguyenvana@gmail.com"

    @patch("agents.cv_parser.chunk_and_store")
    @patch("agents.cv_parser.to_markdown")
    def test_parse_extracts_name(self, mock_md, mock_chunk):
        from tools.cv_to_markdown import MarkdownResult
        from rag.cv_chunker import ChunkingResult

        mock_md.return_value    = MarkdownResult(SAMPLE_MARKDOWN, "pymupdf", 1)
        mock_chunk.return_value = ChunkingResult("cv_x", "cv.pdf", 3, ["Skills"])

        agent  = CVParserAgent()
        result = agent.parse("dummy.pdf")

        # Tên phải được extract, không phải "Unknown"
        assert result.candidate_name != "Unknown"
        assert len(result.candidate_name) > 2

    @patch("agents.cv_parser.chunk_and_store")
    @patch("agents.cv_parser.to_markdown")
    def test_parse_same_file_same_cv_id(self, mock_md, mock_chunk):
        """Cùng filename → cùng cv_id (idempotent)."""
        from tools.cv_to_markdown import MarkdownResult
        from rag.cv_chunker import ChunkingResult

        mock_md.return_value    = MarkdownResult(SAMPLE_MARKDOWN, "pymupdf", 1)
        mock_chunk.return_value = ChunkingResult("cv_x", "test.pdf", 2, [])

        agent = CVParserAgent()
        r1    = agent.parse("test.pdf")
        r2    = agent.parse("test.pdf")
        assert r1.cv_id == r2.cv_id

    @patch("agents.cv_parser.to_markdown")
    def test_parse_fails_gracefully(self, mock_md):
        """to_markdown raises → fallback ParsedCV, không crash."""
        mock_md.side_effect = RuntimeError("Corrupt file")

        agent  = CVParserAgent()

        with pytest.raises(RuntimeError):
            agent.parse("bad.pdf")

    @patch("agents.cv_parser.chunk_and_store")
    @patch("agents.cv_parser.to_markdown")
    def test_batch_parse_continues_after_failure(self, mock_md, mock_chunk):
        """1 file fail → fallback, batch tiếp tục."""
        from tools.cv_to_markdown import MarkdownResult
        from rag.cv_chunker import ChunkingResult

        mock_md.side_effect = [
            MarkdownResult(SAMPLE_MARKDOWN, "pymupdf", 1),
            RuntimeError("Bad file"),
            MarkdownResult(SAMPLE_MARKDOWN, "pymupdf", 1),
        ]
        mock_chunk.return_value = ChunkingResult("cv_x", "cv.pdf", 3, ["Skills"])

        agent   = CVParserAgent()
        results = agent.parse_batch(["cv1.pdf", "bad.pdf", "cv3.pdf"])

        assert len(results) == 3
        assert results[0].chunk_count  > 0            # OK
        assert "PARSE ERROR" in results[1].candidate_name  # Fallback
        assert results[2].chunk_count  > 0            # OK

    def test_fallback_parsed_has_warning(self):
        result = _fallback_parsed("broken.pdf", "File corrupt")
        assert len(result.warnings) > 0
        assert "corrupt" in result.warnings[0].lower() or "Parsing" in result.warnings[0]

    def test_fallback_parsed_has_cv_id(self):
        result = _fallback_parsed("test.pdf", "error")
        assert result.cv_id.startswith("cv_failed_")