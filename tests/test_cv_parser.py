"""
Tests for CVParserAgent.

Run with: pytest tests/test_agents/test_cv_parser.py -v

Tests are structured in 3 tiers:
  Unit:        test individual methods with mocks (no API calls, no files)
  Integration: test full parse() pipeline with a mock LLM
  Schema:      test Pydantic validation edge cases
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.cv_parser import CVParserAgent
from schemas.cv_schema import CandidateProfile, ParseConfidence


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

VALID_LLM_JSON = {
    "full_name": "Nguyen Van A",
    "contact": {
        "email": "nguyenvana@gmail.com",
        "phone": "0901234567",
        "linkedin": "linkedin.com/in/nguyenvana",
        "github": None,
        "location": "Ho Chi Minh City",
    },
    "total_experience_years": 4.5,
    "work_history": [
        {
            "company": "FPT Software",
            "role": "Backend Engineer",
            "start_year": 2020,
            "end_year": 2024,
            "duration_months": 48,
            "responsibilities": ["Design REST APIs", "Maintain PostgreSQL databases"],
            "achievements": ["Reduced API latency by 30%"],
            "technologies": ["Python", "FastAPI", "PostgreSQL", "Redis"],
        }
    ],
    "technical_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
    "soft_skills": ["Teamwork", "Communication"],
    "certifications": ["AWS Solutions Architect Associate"],
    "education": [
        {
            "institution": "HCMC University of Technology",
            "degree": "Bachelor of Computer Science",
            "major": "Computer Science",
            "level": "bachelor",
            "graduation_year": 2020,
            "gpa": 3.4,
        }
    ],
    "highest_education_level": "bachelor",
    "languages": [
        {"language": "Vietnamese", "proficiency": "Native"},
        {"language": "English", "proficiency": "B2"},
    ],
    "missing_fields": [],
    "parse_warnings": [],
}


@pytest.fixture
def agent() -> CVParserAgent:
    """CVParserAgent with a fake API key — LLM calls will be mocked."""
    return CVParserAgent(api_key="fake-key-for-testing")


@pytest.fixture
def mock_llm_response() -> str:
    """Valid JSON string as LLM would return."""
    return json.dumps(VALID_LLM_JSON)


# ──────────────────────────────────────────────────────────────────────────────
# Unit: _parse_json_response
# ──────────────────────────────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_clean_json(self, agent: CVParserAgent) -> None:
        raw = json.dumps({"full_name": "Test User"})
        result = agent._parse_json_response(raw)
        assert result["full_name"] == "Test User"

    def test_strips_markdown_fences(self, agent: CVParserAgent) -> None:
        """LLM often wraps JSON in ```json ... ``` despite instructions."""
        raw = "```json\n" + json.dumps({"full_name": "Test"}) + "\n```"
        result = agent._parse_json_response(raw)
        assert result["full_name"] == "Test"

    def test_strips_plain_fences(self, agent: CVParserAgent) -> None:
        raw = "```\n" + json.dumps({"full_name": "Test"}) + "\n```"
        result = agent._parse_json_response(raw)
        assert result["full_name"] == "Test"

    def test_strips_preamble(self, agent: CVParserAgent) -> None:
        """LLM sometimes adds 'Here is the JSON:' before the object."""
        raw = "Here is the extracted CV:\n" + json.dumps({"full_name": "Test"})
        result = agent._parse_json_response(raw)
        assert result["full_name"] == "Test"

    def test_raises_on_no_json(self, agent: CVParserAgent) -> None:
        with pytest.raises(ValueError, match="No JSON object found"):
            agent._parse_json_response("Sorry, I cannot help with that.")

    def test_raises_on_invalid_json(self, agent: CVParserAgent) -> None:
        with pytest.raises(json.JSONDecodeError):
            agent._parse_json_response("{invalid json here}")


# ──────────────────────────────────────────────────────────────────────────────
# Unit: _assess_confidence
# ──────────────────────────────────────────────────────────────────────────────

class TestAssessConfidence:
    def _profile_from_dict(self, data: dict) -> CandidateProfile:
        return CandidateProfile.model_validate(data)

    def test_high_confidence_complete_profile(self, agent: CVParserAgent) -> None:
        profile = self._profile_from_dict(VALID_LLM_JSON)
        confidence = agent._assess_confidence(profile, extraction_method="native")
        assert confidence == ParseConfidence.HIGH

    def test_medium_confidence_ocr_with_missing_fields(self, agent: CVParserAgent) -> None:
        """OCR + missing email + missing phone → should drop to MEDIUM."""
        data = {**VALID_LLM_JSON, "contact": {}, "missing_fields": ["email", "phone", "github"]}
        profile = self._profile_from_dict(data)
        confidence = agent._assess_confidence(profile, extraction_method="ocr")
        # score: 100 - 20 (ocr) - 10 (no email) - 10 (>3 missing) = 60 → MEDIUM
        assert confidence == ParseConfidence.MEDIUM

    def test_low_confidence_minimal_profile(self, agent: CVParserAgent) -> None:
        minimal = {
            "full_name": "Unknown",
            "contact": {},
            "work_history": [],
            "technical_skills": [],
            "missing_fields": ["email", "phone", "work_history", "education", "skills"],
        }
        profile = self._profile_from_dict(minimal)
        confidence = agent._assess_confidence(profile, extraction_method="ocr")
        assert confidence == ParseConfidence.LOW


# ──────────────────────────────────────────────────────────────────────────────
# Integration: full parse() pipeline (LLM mocked)
# ──────────────────────────────────────────────────────────────────────────────

class TestParseIntegration:
    def _mock_extraction(self, text: str, method: str):
        """Helper to mock both PDF and DOCX extractors."""
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            text: str
            method: str
            page_count: int = 1
            char_count: int = 0
            paragraph_count: int = 10

            def __post_init__(self) -> None:
                if self.char_count == 0:
                    self.char_count = len(self.text)

        return FakeResult(text=text, method=method)

    @patch("agents.cv_parser.CVParserAgent._call_llm")
    @patch("agents.cv_parser.extract_pdf_text")
    def test_parse_pdf_success(
        self,
        mock_extract: MagicMock,
        mock_llm: MagicMock,
        agent: CVParserAgent,
    ) -> None:
        """Full pipeline: PDF → extract → LLM → CandidateProfile."""
        mock_extract.return_value = self._mock_extraction(
            "Nguyen Van A | Backend Engineer ...", "native"
        )
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        profile = agent.parse("dummy.pdf")

        assert isinstance(profile, CandidateProfile)
        assert profile.full_name == "Nguyen Van A"
        assert profile.total_experience_years == 4.5
        assert "Python" in profile.technical_skills
        assert profile.confidence == ParseConfidence.HIGH
        assert profile.extraction_method == "native"

    @patch("agents.cv_parser.CVParserAgent._call_llm")
    @patch("agents.cv_parser.extract_pdf_text")
    def test_parse_retries_on_bad_json(
        self,
        mock_extract: MagicMock,
        mock_llm: MagicMock,
        agent: CVParserAgent,
    ) -> None:
        """Agent retries when LLM first returns invalid JSON."""
        mock_extract.return_value = self._mock_extraction("...", "native")
        # First call returns garbage, second returns valid JSON
        mock_llm.side_effect = [
            "Sorry I cannot help!",
            json.dumps(VALID_LLM_JSON),
        ]

        with patch("agents.cv_parser.time.sleep"):  # skip sleep in tests
            profile = agent.parse("dummy.pdf")

        assert profile.full_name == "Nguyen Van A"
        assert mock_llm.call_count == 2  # Confirmed retry happened

    @patch("agents.cv_parser.CVParserAgent._call_llm")
    @patch("agents.cv_parser.extract_pdf_text")
    def test_parse_raises_after_max_retries(
        self,
        mock_extract: MagicMock,
        mock_llm: MagicMock,
        agent: CVParserAgent,
    ) -> None:
        """RuntimeError raised when all retries exhausted."""
        mock_extract.return_value = self._mock_extraction("...", "native")
        mock_llm.return_value = "not json at all"

        with patch("agents.cv_parser.time.sleep"):
            with pytest.raises(RuntimeError, match="failed to return valid JSON"):
                agent.parse("dummy.pdf")

        assert mock_llm.call_count == 3  # MAX_RETRIES = 3

    @patch("agents.cv_parser.CVParserAgent._call_llm")
    @patch("agents.cv_parser.extract_docx_text")
    def test_parse_docx(
        self,
        mock_extract: MagicMock,
        mock_llm: MagicMock,
        agent: CVParserAgent,
    ) -> None:
        """DOCX files route to docx extractor."""
        mock_extract.return_value = self._mock_extraction("...", "docx")
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        profile = agent.parse("dummy.docx")
        assert profile.extraction_method == "docx"
        mock_extract.assert_called_once_with("dummy.docx")

    def test_parse_unsupported_format_raises(self, agent: CVParserAgent) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            agent.parse("resume.txt")

    @patch("agents.cv_parser.CVParserAgent._call_llm")
    @patch("agents.cv_parser.extract_pdf_text")
    def test_batch_parse_skips_failed_cv(
        self,
        mock_extract: MagicMock,
        mock_llm: MagicMock,
        agent: CVParserAgent,
    ) -> None:
        """Batch parse returns fallback profile when one CV fails, others continue."""
        mock_extract.side_effect = [
            self._mock_extraction("CV 1 text", "native"),
            RuntimeError("Corrupted file"),             # CV 2 fails
            self._mock_extraction("CV 3 text", "native"),
        ]
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        with patch("agents.cv_parser.time.sleep"):
            results = agent.parse_batch(["cv1.pdf", "cv2.pdf", "cv3.pdf"])

        assert len(results) == 3
        assert results[0].full_name == "Nguyen Van A"           # success
        assert "[PARSE ERROR]" in results[1].full_name          # fallback
        assert results[2].full_name == "Nguyen Van A"           # success


# ──────────────────────────────────────────────────────────────────────────────
# Schema: Pydantic edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestCandidateProfileSchema:
    def test_defaults_applied_for_missing_fields(self) -> None:
        profile = CandidateProfile()
        assert profile.full_name == "Unknown"
        assert profile.technical_skills == []
        assert profile.work_history == []

    def test_invalid_graduation_year_becomes_none(self) -> None:
        from schemas.cv_schema import EducationEntry
        edu = EducationEntry(graduation_year=1800)  # Invalid
        assert edu.graduation_year is None

    def test_valid_graduation_year_kept(self) -> None:
        from schemas.cv_schema import EducationEntry
        edu = EducationEntry(graduation_year=2020)
        assert edu.graduation_year == 2020

    def test_full_profile_validates_correctly(self) -> None:
        profile = CandidateProfile.model_validate(VALID_LLM_JSON)
        assert profile.contact.email == "nguyenvana@gmail.com"
        assert len(profile.work_history) == 1
        assert profile.work_history[0].company == "FPT Software"
        assert len(profile.education) == 1
        assert profile.education[0].gpa == 3.4