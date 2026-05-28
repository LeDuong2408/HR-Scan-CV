"""
Tests for JDMatcherAgent v2.

Run: pytest tests/test_jd_matcher_v2.py -v

3 tầng:
  Unit JD Parse:    test _parse_json, parse_jd() với mock LLM
  Unit Evidence:    test _query_evidence với mock ChromaDB
  Integration:      test match() và match_batch() full pipeline
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.cv_parser import ParsedCV
from agents.jd_matcher import JDMatcherAgent
from schemas.jd_schema import ParsedJD
from schemas.match_schema import MatchLevel, MatchResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_JD_TEXT = """
Senior Backend Engineer

We are looking for a Senior Backend Engineer with 3+ years of Python experience.

Requirements:
- 3+ years Python backend development
- FastAPI or Django REST framework
- AWS Lambda and S3 experience
- PostgreSQL and Redis
- Docker and CI/CD pipelines

Nice to have:
- Kubernetes
- Terraform

Responsibilities:
- Design and build scalable REST APIs
- Maintain PostgreSQL databases
- Deploy on AWS infrastructure
"""

PARSED_JD_JSON = {
    "job_title":                  "Senior Backend Engineer",
    "required_skills":            [
        "Python 3+ years",
        "FastAPI",
        "AWS Lambda",
        "PostgreSQL",
        "Docker",
    ],
    "nice_to_have":               ["Kubernetes", "Terraform"],
    "required_experience_years":  3.0,
    "required_experience_domain": "backend development",
    "required_education_level":   "bachelor",
    "required_education_major":   "Computer Science or related",
    "key_responsibilities":       ["Design REST APIs", "Deploy on AWS"],
    "seniority_level":            "senior",
}

MATCH_RESULT_JSON = {
    "candidate_name": "Nguyen Van A",
    "job_title":      "Senior Backend Engineer",
    "requirement_matches": [
        {
            "requirement":          "Python 3+ years",
            "candidate_evidence":   "4 years Python at FPT Software",
            "match_level":          "full",
            "gap_note":             None,
        },
        {
            "requirement":        "FastAPI",
            "candidate_evidence": "FastAPI in Skills section",
            "match_level":        "full",
            "gap_note":           None,
        },
        {
            "requirement":        "AWS Lambda",
            "candidate_evidence": "AWS Lambda listed in skills",
            "match_level":        "partial",
            "gap_note":           "Listed in skills but no project evidence",
        },
        {
            "requirement":        "Kubernetes",
            "candidate_evidence": None,
            "match_level":        "missing",
            "gap_note":           "No Kubernetes evidence found",
        },
    ],
    "skill_gap": {
        "matched":          ["Python", "FastAPI", "PostgreSQL"],
        "missing_critical": ["Docker"],
        "missing_nice":     ["Kubernetes", "Terraform"],
        "bonus":            ["Redis"],
    },
    "experience": {
        "required_years":    3.0,
        "candidate_years":   4.5,
        "meets_requirement": True,
        "domain_relevance":  1.0,
        "relevance_note":    "Backend experience directly relevant",
    },
    "raw_similarity_score": 0.0,
    "match_summary":        "Strong match on Python and FastAPI. Gap in Docker and Kubernetes.",
    "low_confidence":       False,
    "warnings":             [],
}


@pytest.fixture
def agent() -> JDMatcherAgent:
    return JDMatcherAgent(api_key="fake-key")


@pytest.fixture
def sample_candidate() -> ParsedCV:
    return ParsedCV(
        cv_id          = "cv_abc123",
        file_name      = "nguyen_van_a.pdf",
        candidate_name = "Nguyen Van A",
        email          = "a@gmail.com",
        markdown       = "# Nguyen Van A\n# Skills\nPython FastAPI\n# Experience\nFPT 4 years",
        chunk_count    = 5,
        sections       = ["Skills", "Experience", "Education"],
        parse_method   = "pymupdf",
    )


@pytest.fixture
def sample_parsed_jd() -> ParsedJD:
    return ParsedJD.model_validate(PARSED_JD_JSON)


# ── Tầng 1: Unit — JD Parsing ─────────────────────────────────────────────────

class TestParseJson:
    def test_clean_json(self, agent):
        raw  = json.dumps({"job_title": "Dev"})
        data = agent._parse_json(raw)
        assert data["job_title"] == "Dev"

    def test_strips_markdown_fences(self, agent):
        raw  = "```json\n" + json.dumps({"job_title": "Dev"}) + "\n```"
        data = agent._parse_json(raw)
        assert data["job_title"] == "Dev"

    def test_strips_preamble(self, agent):
        raw  = "Here is the result:\n" + json.dumps({"x": 1})
        data = agent._parse_json(raw)
        assert data["x"] == 1

    def test_raises_on_no_json(self, agent):
        with pytest.raises(ValueError, match="No JSON"):
            agent._parse_json("Sorry, I cannot help.")


class TestParseJD:
    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    def test_parse_jd_success(self, mock_llm, agent):
        mock_llm.return_value = json.dumps(PARSED_JD_JSON)

        result = agent.parse_jd(SAMPLE_JD_TEXT, "Senior Backend Engineer", "backend-2025")

        assert isinstance(result, ParsedJD)
        assert result.job_title              == "Senior Backend Engineer"
        assert len(result.required_skills)   >= 3
        assert "Python 3+ years"              in result.required_skills
        assert result.required_experience_years == 3.0
        assert result.seniority_level        == "senior"

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    def test_parse_jd_cached(self, mock_llm, agent):
        """JD parse kết quả được cache — LLM chỉ gọi 1 lần."""
        mock_llm.return_value = json.dumps(PARSED_JD_JSON)

        r1 = agent.parse_jd(SAMPLE_JD_TEXT, "Dev", "job-1")
        r2 = agent.parse_jd(SAMPLE_JD_TEXT, "Dev", "job-1")

        assert r1 is r2          # Same object từ cache
        assert mock_llm.call_count == 1  # LLM chỉ gọi 1 lần

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    def test_parse_jd_different_jobs_not_cached(self, mock_llm, agent):
        """2 job khác nhau → 2 LLM calls riêng."""
        mock_llm.return_value = json.dumps(PARSED_JD_JSON)

        agent.parse_jd(SAMPLE_JD_TEXT, "Dev 1", "job-1")
        agent.parse_jd(SAMPLE_JD_TEXT, "Dev 2", "job-2")

        assert mock_llm.call_count == 2

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    def test_parse_jd_retries_on_bad_json(self, mock_llm, agent):
        mock_llm.side_effect = [
            "not json",
            json.dumps(PARSED_JD_JSON),
        ]
        with patch("agents.jd_matcher.time.sleep"):
            result = agent.parse_jd(SAMPLE_JD_TEXT, "Dev", "job-1")
        assert result.job_title == "Senior Backend Engineer"
        assert mock_llm.call_count == 2


# ── Tầng 2: Unit — Evidence Query ─────────────────────────────────────────────

class TestQueryEvidence:

    @patch("agents.jd_matcher.query_cv_chunks")
    def test_queries_all_required_skills(
        self, mock_query, agent, sample_candidate, sample_parsed_jd
    ):
        """Mỗi required_skill phải được query ChromaDB."""
        mock_query.return_value = [{"text": "Python experience", "section": "Skills", "score": 0.85, "chunk_index": 0}]

        evidence = agent._query_evidence(sample_candidate.cv_id, sample_parsed_jd)

        # Tất cả required skills phải có key trong evidence
        for skill in sample_parsed_jd.required_skills:
            assert skill in evidence

    @patch("agents.jd_matcher.query_cv_chunks")
    def test_queries_nice_to_have_skills(
        self, mock_query, agent, sample_candidate, sample_parsed_jd
    ):
        """Nice-to-have skills cũng được query."""
        mock_query.return_value = []

        evidence = agent._query_evidence(sample_candidate.cv_id, sample_parsed_jd)

        for skill in sample_parsed_jd.nice_to_have:
            assert skill in evidence

    @patch("agents.jd_matcher.query_cv_chunks")
    def test_missing_skill_has_empty_chunks(
        self, mock_query, agent, sample_candidate, sample_parsed_jd
    ):
        """Skill không có trong CV → chunks = [] (không raise, không filter)."""
        mock_query.return_value = []

        evidence = agent._query_evidence(sample_candidate.cv_id, sample_parsed_jd)

        # Tất cả skills phải có entry, dù empty
        total_skills = len(sample_parsed_jd.required_skills) + len(sample_parsed_jd.nice_to_have)
        assert len(evidence) == total_skills

    @patch("agents.jd_matcher.query_cv_chunks")
    def test_query_uses_correct_cv_id(
        self, mock_query, agent, sample_candidate, sample_parsed_jd
    ):
        """ChromaDB query phải filter đúng cv_id của candidate."""
        mock_query.return_value = []

        agent._query_evidence(sample_candidate.cv_id, sample_parsed_jd)

        for call in mock_query.call_args_list:
            assert call.kwargs.get("cv_id") == sample_candidate.cv_id or \
                   call.args[1] == sample_candidate.cv_id

    @patch("agents.jd_matcher.query_cv_chunks")
    def test_no_score_filter_applied(
        self, mock_query, agent, sample_candidate, sample_parsed_jd
    ):
        """min_score=0.0 — không filter bất kỳ chunk nào."""
        mock_query.return_value = []

        agent._query_evidence(sample_candidate.cv_id, sample_parsed_jd)

        for call in mock_query.call_args_list:
            min_score = call.kwargs.get("min_score", 0.0)
            assert min_score == 0.0


# ── Tầng 3: Integration ───────────────────────────────────────────────────────

class TestMatchIntegration:

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.query_cv_chunks")
    def test_match_success_full_pipeline(
        self, mock_query, mock_llm, agent, sample_candidate, sample_parsed_jd
    ):
        """Full pipeline: ChromaDB evidence → LLM → MatchResult."""
        mock_query.return_value = [
            {"text": "Python 4 years", "section": "Experience", "score": 0.90, "chunk_index": 0}
        ]
        # LLM chỉ được gọi 1 lần (match analyze)
        mock_llm.return_value = json.dumps(MATCH_RESULT_JSON)

        result = agent.match(sample_candidate, sample_parsed_jd)

        assert isinstance(result, MatchResult)
        assert result.candidate_name == "Nguyen Van A"
        assert result.job_title      == "Senior Backend Engineer"
        assert len(result.requirement_matches) > 0

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.query_cv_chunks")
    def test_match_computes_avg_similarity(
        self, mock_query, mock_llm, agent, sample_candidate, sample_parsed_jd
    ):
        """raw_similarity_score = average của all chunk scores."""
        mock_query.return_value = [
            {"text": "text", "section": "S", "score": 0.8, "chunk_index": 0},
            {"text": "text", "section": "S", "score": 0.6, "chunk_index": 1},
        ]
        mock_llm.return_value = json.dumps(MATCH_RESULT_JSON)

        result = agent.match(sample_candidate, sample_parsed_jd)

        # Score phải > 0 vì có chunks
        assert result.raw_similarity_score > 0

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.query_cv_chunks")
    def test_match_flags_low_confidence_sparse_cv(
        self, mock_query, mock_llm, agent, sample_parsed_jd
    ):
        """CV chỉ có 2 chunks → low_confidence = True."""
        sparse_candidate = ParsedCV(
            cv_id="cv_sparse", file_name="sparse.pdf",
            candidate_name="Sparse User",
            chunk_count=2,  # ít hơn 3
            sections=["Skills"],
            parse_method="pymupdf",
        )
        mock_query.return_value = []
        mock_llm.return_value   = json.dumps(MATCH_RESULT_JSON)

        result = agent.match(sparse_candidate, sample_parsed_jd)
        assert result.low_confidence is True

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.query_cv_chunks")
    def test_match_batch_parses_jd_once(
        self, mock_query, mock_llm, agent, sample_candidate
    ):
        """match_batch: JD parsing chỉ gọi 1 lần dù có 3 candidates."""
        mock_query.return_value = []
        # LLM calls: 1 (parse_jd) + 3 (match per candidate) = 4 total
        mock_llm.return_value = json.dumps(PARSED_JD_JSON)

        # Reset mock để track calls
        call_count = [0]
        def side_effect(prompt, system_prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(PARSED_JD_JSON)   # parse_jd call
            return json.dumps(MATCH_RESULT_JSON)     # match calls

        mock_llm.side_effect = side_effect

        with patch("agents.jd_matcher.time.sleep"):
            results = agent.match_batch(
                candidates = [sample_candidate] * 3,
                jd_text    = SAMPLE_JD_TEXT,
                job_title  = "Senior Backend Engineer",
                job_id     = "backend-2025",
            )

        assert len(results) == 3
        # 1 parse_jd + 3 match = 4 LLM calls
        assert call_count[0] == 4

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.query_cv_chunks")
    def test_match_batch_handles_failure(
        self, mock_query, mock_llm, agent, sample_candidate
    ):
        """1 candidate fail → fallback MatchResult, batch tiếp tục."""
        mock_query.return_value = []

        call_count = [0]
        def side_effect(prompt, system_prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(PARSED_JD_JSON)        # parse_jd OK
            elif call_count[0] == 2:
                return json.dumps(MATCH_RESULT_JSON)     # candidate 1 OK
            elif call_count[0] == 3:
                raise RuntimeError("LLM timeout")        # candidate 2 FAIL
            else:
                return json.dumps(MATCH_RESULT_JSON)     # candidate 3 OK

        mock_llm.side_effect = side_effect

        with patch("agents.jd_matcher.time.sleep"):
            results = agent.match_batch(
                candidates = [sample_candidate] * 3,
                jd_text    = SAMPLE_JD_TEXT,
                job_title  = "Dev",
                job_id     = "job-1",
            )

        assert len(results) == 3
        assert results[0].candidate_name == "Nguyen Van A"   # OK
        assert results[1].low_confidence is True              # Fallback
        assert results[2].candidate_name == "Nguyen Van A"   # OK

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.query_cv_chunks")
    def test_build_match_prompt_includes_all_skills(
        self, mock_query, mock_llm, agent, sample_candidate, sample_parsed_jd
    ):
        """Prompt gửi lên LLM phải chứa TẤT CẢ required skills."""
        mock_query.return_value = []
        mock_llm.return_value   = json.dumps(MATCH_RESULT_JSON)

        agent.match(sample_candidate, sample_parsed_jd)

        # Lấy prompt từ LLM call (call cuối = match analyze)
        last_call_prompt = mock_llm.call_args[0][0]
        for skill in sample_parsed_jd.required_skills:
            assert skill in last_call_prompt, \
                f"Skill '{skill}' missing from LLM prompt"

    @patch("agents.jd_matcher.JDMatcherAgent._call_llm")
    @patch("agents.jd_matcher.query_cv_chunks")
    def test_prompt_shows_empty_evidence_for_missing_skills(
        self, mock_query, mock_llm, agent, sample_candidate, sample_parsed_jd
    ):
        """Skill không có trong CV → prompt phải chứa 'No relevant content found'."""
        mock_query.return_value = []  # Candidate không có gì
        mock_llm.return_value   = json.dumps(MATCH_RESULT_JSON)

        agent.match(sample_candidate, sample_parsed_jd)

        last_call_prompt = mock_llm.call_args[0][0]
        assert "No relevant content found" in last_call_prompt


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestParsedJDSchema:
    def test_valid_parsed_jd(self):
        jd = ParsedJD.model_validate(PARSED_JD_JSON)
        assert jd.job_title == "Senior Backend Engineer"
        assert len(jd.required_skills) == 5
        assert jd.required_experience_years == 3.0

    def test_defaults(self):
        jd = ParsedJD(job_title="Dev")
        assert jd.required_skills == []
        assert jd.nice_to_have    == []
        assert jd.required_experience_years is None

    def test_match_result_still_valid(self):
        """MatchResult schema vẫn hoạt động với Agent 2 v2 output."""
        result = MatchResult.model_validate(MATCH_RESULT_JSON)
        assert result.requirement_matches[0].match_level == MatchLevel.FULL
        assert result.requirement_matches[3].match_level == MatchLevel.MISSING