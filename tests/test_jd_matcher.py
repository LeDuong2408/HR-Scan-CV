"""
Tests for JDMatcherAgent.

Run: pytest tests/test_jd_matcher.py -v

3 tầng:
  Unit:        test từng method nhỏ (_build_query_text, _parse_json_response)
  Integration: test full match() pipeline với mock LLM + mock ChromaDB
  Schema:      test Pydantic validation của MatchResult
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.jd_matcher import JDMatcherAgent
from rag.retriever import RetrievedChunk
from schemas.cv_schema import (
    CandidateProfile,
    ContactInfo,
    EducationEntry,
    LanguageEntry,
    ParseConfidence,
    WorkEntry,
)
from schemas.match_schema import (
    ExperienceAssessment,
    MatchLevel,
    MatchResult,
    RequirementMatch,
    SkillGapReport,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — dùng chung toàn bộ file test
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def agent() -> JDMatcherAgent:
    """JDMatcherAgent với fake API key — LLM sẽ được mock."""
    return JDMatcherAgent(api_key="fake-key-for-testing")


@pytest.fixture
def sample_candidate() -> CandidateProfile:
    """CandidateProfile mẫu — output của CVParserAgent."""
    return CandidateProfile(
        full_name="Nguyen Van A",
        contact=ContactInfo(email="a@gmail.com", phone="0901234567"),
        total_experience_years=4.5,
        technical_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS Lambda"],
        soft_skills=["Communication", "Teamwork"],
        certifications=["AWS Solutions Architect Associate"],
        work_history=[
            WorkEntry(
                company="FPT Software",
                role="Backend Engineer",
                start_year=2020,
                end_year=2024,
                duration_months=48,
                technologies=["Python", "FastAPI", "PostgreSQL", "Redis"],
                achievements=["Reduced API latency by 30%"],
            )
        ],
        education=[
            EducationEntry(
                institution="HCMUT",
                degree="Bachelor of Computer Science",
                graduation_year=2020,
            )
        ],
        languages=[LanguageEntry(language="English", proficiency="B2")],
        confidence=ParseConfidence.HIGH,
    )


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    """JD requirement chunks mẫu — output của ChromaDB retriever."""
    return [
        RetrievedChunk(
            chunk_id="backend-2025_req_000",
            text="3+ years Python backend development",
            score=0.92,
            priority="required",
            job_id="backend-2025",
            job_title="Senior Backend Engineer",
        ),
        RetrievedChunk(
            chunk_id="backend-2025_req_001",
            text="Experience with REST API design and FastAPI framework",
            score=0.88,
            priority="required",
            job_id="backend-2025",
            job_title="Senior Backend Engineer",
        ),
        RetrievedChunk(
            chunk_id="backend-2025_req_002",
            text="AWS Lambda and S3 experience",
            score=0.75,
            priority="required",
            job_id="backend-2025",
            job_title="Senior Backend Engineer",
        ),
        RetrievedChunk(
            chunk_id="backend-2025_nice_000",
            text="Kubernetes and container orchestration",
            score=0.55,
            priority="nice_to_have",
            job_id="backend-2025",
            job_title="Senior Backend Engineer",
        ),
    ]


VALID_LLM_JSON = {
    "candidate_name": "Nguyen Van A",
    "job_title": "Senior Backend Engineer",
    "requirement_matches": [
        {
            "requirement": "3+ years Python backend development",
            "candidate_evidence": "4 years Python at FPT Software (FastAPI, PostgreSQL)",
            "match_level": "full",
            "gap_note": None,
        },
        {
            "requirement": "Experience with REST API design and FastAPI framework",
            "candidate_evidence": "FastAPI listed in technical skills and work history",
            "match_level": "full",
            "gap_note": None,
        },
        {
            "requirement": "AWS Lambda and S3 experience",
            "candidate_evidence": "AWS Lambda in technical skills; AWS cert",
            "match_level": "partial",
            "gap_note": "Has Lambda but S3 not explicitly mentioned",
        },
        {
            "requirement": "Kubernetes and container orchestration",
            "candidate_evidence": None,
            "match_level": "missing",
            "gap_note": "No Kubernetes or K8s mentioned in CV",
        },
    ],
    "skill_gap": {
        "matched": ["Python", "FastAPI", "PostgreSQL", "AWS Lambda"],
        "missing_critical": [],
        "missing_nice": ["Kubernetes"],
        "bonus": ["Docker", "Redis"],
    },
    "experience": {
        "required_years": 3.0,
        "candidate_years": 4.5,
        "meets_requirement": True,
        "domain_relevance": 1.0,
        "relevance_note": "Backend experience directly relevant",
    },
    "raw_similarity_score": 0.0,
    "match_summary": (
        "Strong match on Python and FastAPI. "
        "Gaps in Kubernetes. "
        "Recommend for technical interview."
    ),
    "low_confidence": False,
    "warnings": [],
}


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 1: Unit — test từng method nhỏ
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildQueryText:
    """
    _build_query_text() tóm tắt CandidateProfile thành 1 chuỗi text
    để embed và search ChromaDB.
    """

    def test_includes_technical_skills(
            self, agent: JDMatcherAgent, sample_candidate: CandidateProfile
    ) -> None:
        query = agent._build_query_text(sample_candidate)
        assert "Python" in query
        assert "FastAPI" in query

    def test_includes_recent_role(
            self, agent: JDMatcherAgent, sample_candidate: CandidateProfile
    ) -> None:
        query = agent._build_query_text(sample_candidate)
        assert "Backend Engineer" in query
        assert "FPT Software" in query

    def test_includes_certifications(
            self, agent: JDMatcherAgent, sample_candidate: CandidateProfile
    ) -> None:
        query = agent._build_query_text(sample_candidate)
        assert "AWS Solutions Architect" in query

    def test_empty_skills_still_works(self, agent: JDMatcherAgent) -> None:
        """Candidate chưa có skills — không crash."""
        candidate = CandidateProfile(full_name="Test User")
        query = agent._build_query_text(candidate)
        # Không crash, có thể là empty string hoặc có education
        assert isinstance(query, str)

    def test_limits_work_history_to_3(self, agent: JDMatcherAgent) -> None:
        """Chỉ lấy 3 job gần nhất để query không quá dài."""
        candidate = CandidateProfile(
            full_name="Test",
            work_history=[
                WorkEntry(company=f"Company{i}", role=f"Role{i}")
                for i in range(6)
            ],
        )
        query = agent._build_query_text(candidate)
        # Chỉ có 3 công ty đầu trong query
        assert "Company0" in query
        assert "Company1" in query
        assert "Company2" in query
        assert "Company5" not in query  # Công ty thứ 6 bị bỏ


class TestParseJsonResponse:
    """
    _parse_json_response() làm sạch output LLM.
    Giống test trong CVParserAgent.
    """

    def test_clean_json(self, agent: JDMatcherAgent) -> None:
        raw = json.dumps({"candidate_name": "Test"})
        assert agent._parse_json_response(raw)["candidate_name"] == "Test"

    def test_strips_markdown_fences(self, agent: JDMatcherAgent) -> None:
        raw = "```json\n" + json.dumps({"candidate_name": "Test"}) + "\n```"
        assert agent._parse_json_response(raw)["candidate_name"] == "Test"

    def test_strips_preamble(self, agent: JDMatcherAgent) -> None:
        raw = "Here is the analysis:\n" + json.dumps({"candidate_name": "Test"})
        assert agent._parse_json_response(raw)["candidate_name"] == "Test"

    def test_raises_on_no_json(self, agent: JDMatcherAgent) -> None:
        with pytest.raises(ValueError, match="No JSON object found"):
            agent._parse_json_response("I cannot determine the match.")

    def test_raises_on_invalid_json(self, agent: JDMatcherAgent) -> None:
        with pytest.raises(Exception):
            agent._parse_json_response("{invalid: json}")


class TestBuildPrompt:
    """_build_prompt() tạo ra prompt đúng format."""

    def test_contains_candidate_name(
            self,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        prompt = agent._build_prompt(sample_candidate, sample_chunks, "Senior Backend Engineer")
        assert "Nguyen Van A" in prompt

    def test_contains_jd_requirements(
            self,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        prompt = agent._build_prompt(sample_candidate, sample_chunks, "Senior Backend Engineer")
        assert "3+ years Python backend development" in prompt
        assert "FastAPI framework" in prompt

    def test_marks_required_vs_nice(
            self,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        prompt = agent._build_prompt(sample_candidate, sample_chunks, "Senior Backend Engineer")
        assert "[REQUIRED]" in prompt
        assert "[NICE]" in prompt

    def test_excludes_raw_text(
            self,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        """raw_text không được đưa vào prompt — tránh context quá dài."""
        sample_candidate_with_raw = sample_candidate.model_copy()
        prompt = agent._build_prompt(sample_candidate_with_raw, sample_chunks, "Senior Backend")
        assert "raw_text" not in prompt


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 2: Integration — test full pipeline với mock
# ──────────────────────────────────────────────────────────────────────────────

class TestMatchIntegration:

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.search_jd_requirements")
    def test_match_success(
            self,
            mock_search: MagicMock,
            mock_llm: MagicMock,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        """Full pipeline thành công: ChromaDB → LLM → MatchResult."""
        mock_search.return_value = sample_chunks
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        result = agent.match(sample_candidate, "backend-2025", "Senior Backend Engineer")

        assert isinstance(result, MatchResult)
        assert result.candidate_name == "Nguyen Van A"
        assert result.job_title == "Senior Backend Engineer"
        assert len(result.requirement_matches) == 4
        assert "Python" in result.skill_gap.matched
        assert "Kubernetes" in result.skill_gap.missing_nice
        assert result.experience.meets_requirement is True

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.search_jd_requirements")
    def test_match_enriches_similarity_score(
            self,
            mock_search: MagicMock,
            mock_llm: MagicMock,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        """raw_similarity_score được tính từ average của chunks."""
        mock_search.return_value = sample_chunks
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        result = agent.match(sample_candidate, "backend-2025", "Senior Backend Engineer")

        # Average score: (0.92 + 0.88 + 0.75 + 0.55) / 4 = 0.775
        expected_avg = round(
            sum(c.score for c in sample_chunks) / len(sample_chunks), 4
        )
        assert result.raw_similarity_score == expected_avg

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.search_jd_requirements")
    def test_match_saves_chunk_ids(
            self,
            mock_search: MagicMock,
            mock_llm: MagicMock,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        """jd_chunks_used lưu IDs để audit sau."""
        mock_search.return_value = sample_chunks
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        result = agent.match(sample_candidate, "backend-2025", "Senior Backend Engineer")

        assert "backend-2025_req_000" in result.jd_chunks_used
        assert "backend-2025_nice_000" in result.jd_chunks_used

    @patch("agents.jd_matcher.search_jd_requirements")
    def test_match_returns_empty_when_no_chunks(
            self,
            mock_search: MagicMock,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
    ) -> None:
        """Khi ChromaDB không có data → trả về empty result, không crash."""
        mock_search.return_value = []

        result = agent.match(sample_candidate, "nonexistent-job", "Some Job")

        assert result.low_confidence is True
        assert any("No JD chunks" in w for w in result.warnings)

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.search_jd_requirements")
    def test_match_retries_on_bad_json(
            self,
            mock_search: MagicMock,
            mock_llm: MagicMock,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        """Retry khi LLM trả về invalid JSON."""
        mock_search.return_value = sample_chunks
        mock_llm.side_effect = [
            "This is not JSON",  # Lần 1: fail
            json.dumps(VALID_LLM_JSON),  # Lần 2: thành công
        ]

        with patch("agents.jd_matcher.time.sleep"):
            result = agent.match(sample_candidate, "backend-2025", "Senior Backend Engineer")

        assert result.candidate_name == "Nguyen Van A"
        assert mock_llm.call_count == 2

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.search_jd_requirements")
    def test_match_raises_after_max_retries(
            self,
            mock_search: MagicMock,
            mock_llm: MagicMock,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        """RuntimeError sau 3 lần retry thất bại."""
        mock_search.return_value = sample_chunks
        mock_llm.return_value = "not json ever"

        with patch("agents.jd_matcher.time.sleep"):
            with pytest.raises(RuntimeError, match="failed to return valid JSON"):
                agent.match(sample_candidate, "backend-2025", "Senior Backend Engineer")

        assert mock_llm.call_count == 3

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.search_jd_requirements")
    def test_match_inherits_low_confidence_from_cv(
            self,
            mock_search: MagicMock,
            mock_llm: MagicMock,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        """Nếu CV parse LOW confidence → MatchResult cũng là low_confidence."""
        sample_candidate.confidence = ParseConfidence.LOW
        mock_search.return_value = sample_chunks
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        result = agent.match(sample_candidate, "backend-2025", "Senior Backend Engineer")

        assert result.low_confidence is True
        assert any("LOW confidence" in w for w in result.warnings)

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.search_jd_requirements")
    def test_batch_match_skips_failed(
            self,
            mock_search: MagicMock,
            mock_llm: MagicMock,
            agent: JDMatcherAgent,
            sample_candidate: CandidateProfile,
            sample_chunks: list[RetrievedChunk],
    ) -> None:
        """Batch match không crash khi 1 candidate fail."""
        candidate_ok = sample_candidate
        candidate_fail = CandidateProfile(full_name="Bad Candidate")

        mock_search.return_value = sample_chunks
        # Lần 1 OK, lần 2 LLM crash, lần 3 OK
        mock_llm.side_effect = [
            json.dumps(VALID_LLM_JSON),
            RuntimeError("LLM timeout"),
            json.dumps(VALID_LLM_JSON),
        ]

        with patch("agents.jd_matcher.time.sleep"):
            results = agent.match_batch(
                [candidate_ok, candidate_fail, candidate_ok],
                job_id="backend-2025",
                job_title="Senior Backend Engineer",
            )

        assert len(results) == 3
        assert results[0].candidate_name == "Nguyen Van A"  # OK
        assert results[1].low_confidence is True  # Fallback
        assert results[2].candidate_name == "Nguyen Van A"  # OK


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 3: Schema — test Pydantic validation của MatchResult
# ──────────────────────────────────────────────────────────────────────────────

class TestMatchResultSchema:

    def test_defaults_work(self) -> None:
        result = MatchResult(candidate_name="Test", job_title="Dev")
        assert result.requirement_matches == []
        assert result.skill_gap.matched == []
        assert result.low_confidence is False
        assert result.warnings == []

    def test_match_level_enum_validation(self) -> None:
        req = RequirementMatch(
            requirement="Python 3+ years",
            candidate_evidence="4 years Python",
            match_level="full",  # String → auto-convert to Enum
        )
        assert req.match_level == MatchLevel.FULL

    def test_invalid_match_level_raises(self) -> None:
        with pytest.raises(Exception):
            RequirementMatch(
                requirement="Python",
                candidate_evidence=None,
                match_level="excellent",  # Không phải full/partial/missing
            )

    def test_full_result_validates_correctly(self) -> None:
        result = MatchResult.model_validate(VALID_LLM_JSON)
        assert result.candidate_name == "Nguyen Van A"
        assert len(result.requirement_matches) == 4
        assert result.requirement_matches[0].match_level == MatchLevel.FULL
        assert result.requirement_matches[2].match_level == MatchLevel.PARTIAL
        assert result.requirement_matches[3].match_level == MatchLevel.MISSING
        assert result.experience.domain_relevance == 1.0
        assert result.skill_gap.bonus == ["Docker", "Redis"]

    def test_experience_assessment_defaults(self) -> None:
        exp = ExperienceAssessment()
        assert exp.meets_requirement is False
        assert exp.domain_relevance == 0.0
        assert exp.required_years is None

    def test_skill_gap_defaults(self) -> None:
        gap = SkillGapReport()
        assert gap.matched == []
        assert gap.missing_critical == []
        assert gap.missing_nice == []
        assert gap.bonus == []