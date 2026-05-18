"""
Tests for ScorerAgent.

Run: pytest tests/test_scorer.py -v

4 tầng:
  Unit Programmatic: test _score_technical, _score_experience (toán học, không cần mock)
  Unit LLM:          test _parse_llm_scores, _parse_json_response
  Integration:       test score() và score_and_rank() với mock LLM + mock ChromaDB
  Schema:            test ScoringRubric validation (weight sum = 100)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.scorer import ScorerAgent
from schemas.cv_schema import (
    CandidateProfile,
    EducationEntry,
    EducationLevel,
    LanguageEntry,
    ParseConfidence,
    WorkEntry,
)
from schemas.match_schema import (
    ExperienceAssessment,
    MatchResult,
    RequirementMatch,
    MatchLevel,
    SkillGapReport,
)
from schemas.score_schema import (
    CandidateScore,
    DEFAULT_RUBRIC,
    DimensionConfig,
    DimensionScore,
    RankedCandidate,
    ScoreTier,
    ScoringRubric,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def agent() -> ScorerAgent:
    return ScorerAgent(api_key="fake-key-for-testing")


@pytest.fixture
def sample_match() -> MatchResult:
    """MatchResult mẫu — output của JDMatcherAgent."""
    return MatchResult(
        candidate_name="Nguyen Van A",
        job_title="Senior Backend Engineer",
        requirement_matches=[
            RequirementMatch(
                requirement="3+ years Python",
                candidate_evidence="4 years Python at FPT",
                match_level=MatchLevel.FULL,
            ),
            RequirementMatch(
                requirement="AWS experience",
                candidate_evidence="AWS Lambda in skills",
                match_level=MatchLevel.PARTIAL,
            ),
            RequirementMatch(
                requirement="Kubernetes",
                candidate_evidence=None,
                match_level=MatchLevel.MISSING,
            ),
        ],
        skill_gap=SkillGapReport(
            matched=["Python", "FastAPI", "PostgreSQL", "AWS Lambda"],
            missing_critical=["Kubernetes"],
            missing_nice=["Terraform"],
            bonus=["Docker", "Redis"],
        ),
        experience=ExperienceAssessment(
            required_years=3.0,
            candidate_years=4.5,
            meets_requirement=True,
            domain_relevance=1.0,
            relevance_note="Backend experience directly relevant",
        ),
        match_summary="Strong match on Python. Gap in Kubernetes.",
        raw_similarity_score=0.82,
    )


@pytest.fixture
def sample_candidate() -> CandidateProfile:
    """CandidateProfile mẫu — output của CVParserAgent."""
    return CandidateProfile(
        full_name="Nguyen Van A",
        total_experience_years=4.5,
        technical_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS Lambda"],
        soft_skills=["Communication", "Teamwork"],
        certifications=["AWS Solutions Architect Associate"],
        work_history=[
            WorkEntry(
                company="FPT Software",
                role="Backend Engineer",
                duration_months=48,
                technologies=["Python", "FastAPI", "PostgreSQL"],
                achievements=["Reduced API latency by 30%", "Led team of 5"],
            )
        ],
        education=[
            EducationEntry(
                institution="HCMUT",
                degree="Bachelor of Computer Science",
                level=EducationLevel.BACHELOR,
                major="Computer Science",
                graduation_year=2020,
                gpa=3.4,
            )
        ],
        languages=[
            LanguageEntry(language="Vietnamese", proficiency="Native"),
            LanguageEntry(language="English", proficiency="B2"),
        ],
        confidence=ParseConfidence.HIGH,
    )


VALID_LLM_JSON = {
    "dimension_scores": [
        {
            "dimension": "education",
            "max_score": 15,
            "raw_score": 12.0,
            "percentage": 80.0,
            "rationale": "Bachelor CS from HCMUT, relevant major, GPA 3.4.",
            "scored_by": "llm",
        },
        {
            "dimension": "achievements",
            "max_score": 20,
            "raw_score": 15.0,
            "percentage": 75.0,
            "rationale": "Quantified achievement (30% latency), led team of 5.",
            "scored_by": "llm",
        },
        {
            "dimension": "soft_skills",
            "max_score": 10,
            "raw_score": 8.0,
            "percentage": 80.0,
            "rationale": "English B2, AWS cert, strong communication.",
            "scored_by": "llm",
        },
    ],
    "strengths": [
        "4.5 years Python backend directly relevant",
        "Quantified achievement: 30% latency reduction",
        "AWS certified",
    ],
    "concerns": [
        "Missing Kubernetes — required for this role",
        "S3 experience not explicitly mentioned",
    ],
    "recommendation": "Recommend for interview.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 1: Unit — Programmatic scoring (không cần mock, không cần LLM)
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreTechnical:
    """
    _score_technical() tính điểm dựa trên skill_gap.
    Kết quả phải deterministic.
    """

    def test_perfect_match_no_missing(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """4 matched, 0 missing critical → điểm cao."""
        sample_match.skill_gap.missing_critical = []
        score = agent._score_technical(sample_match, DEFAULT_RUBRIC)

        assert score.dimension == "technical_skills"
        assert score.max_score == 35
        assert score.raw_score > 25  # Điểm cao vì 4/4 matched
        assert score.scored_by == "programmatic"

    def test_missing_critical_reduces_score(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """Mỗi critical skill thiếu phạt 12%."""
        # Baseline: 1 critical missing
        score_1 = agent._score_technical(sample_match, DEFAULT_RUBRIC)

        # Thêm 2 critical missing nữa
        sample_match.skill_gap.missing_critical = ["Kubernetes", "Docker", "Terraform"]
        score_3 = agent._score_technical(sample_match, DEFAULT_RUBRIC)

        assert score_3.raw_score < score_1.raw_score  # Phạt nhiều hơn

    def test_no_skills_data_uses_conservative_score(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """Không có data → 40% thay vì 0 hay crash."""
        sample_match.skill_gap.matched = []
        sample_match.skill_gap.missing_critical = []
        score = agent._score_technical(sample_match, DEFAULT_RUBRIC)

        assert score.percentage == 40.0
        assert "No skill data" in score.rationale

    def test_score_never_exceeds_max(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """Score không vượt quá max_score dù candidate giỏi cỡ nào."""
        sample_match.skill_gap.matched = ["Python"] * 100  # Giả lập nhiều skills
        sample_match.skill_gap.missing_critical = []
        score = agent._score_technical(sample_match, DEFAULT_RUBRIC)

        assert score.raw_score <= score.max_score

    def test_rationale_mentions_matched_count(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        score = agent._score_technical(sample_match, DEFAULT_RUBRIC)
        # Rationale phải đề cập số skills matched
        assert "4" in score.rationale  # 4 matched skills

    def test_rationale_mentions_missing_critical(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        score = agent._score_technical(sample_match, DEFAULT_RUBRIC)
        assert "Kubernetes" in score.rationale

    def test_bonus_skills_mentioned_in_rationale(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        score = agent._score_technical(sample_match, DEFAULT_RUBRIC)
        assert "Docker" in score.rationale or "Redis" in score.rationale


class TestScoreExperience:
    """
    _score_experience() tính điểm dựa trên years + domain_relevance.
    """

    def test_exceeds_required_years_full_domain(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """4.5 năm với required 3, domain 100% → điểm cao."""
        score = agent._score_experience(sample_match, DEFAULT_RUBRIC)

        assert score.dimension == "experience"
        assert score.max_score == 20
        assert score.raw_score >= 16  # Rất tốt
        assert score.scored_by == "programmatic"

    def test_below_required_years_reduces_score(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """1 năm với required 3 → điểm thấp hơn."""
        sample_match.experience.candidate_years = 1.0
        score = agent._score_experience(sample_match, DEFAULT_RUBRIC)

        assert score.raw_score <= 12  # 1/3 years = 60% effective → score ≤ 12

    def test_low_domain_relevance_reduces_score(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """4.5 năm nhưng domain relevance 20% → điểm giảm."""
        sample_match.experience.domain_relevance = 0.2
        score = agent._score_experience(sample_match, DEFAULT_RUBRIC)

        # So với domain relevance 100%
        sample_match.experience.domain_relevance = 1.0
        score_full_domain = agent._score_experience(sample_match, DEFAULT_RUBRIC)

        assert score.raw_score < score_full_domain.raw_score

    def test_no_experience_data_conservative(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """Không có data kinh nghiệm → 40%."""
        sample_match.experience.required_years = None
        sample_match.experience.candidate_years = None
        score = agent._score_experience(sample_match, DEFAULT_RUBRIC)

        assert score.percentage == 40.0

    def test_score_does_not_exceed_max(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        """Dù 10 năm kinh nghiệm với required 1, score <= max."""
        sample_match.experience.candidate_years = 10.0
        sample_match.experience.required_years = 1.0
        score = agent._score_experience(sample_match, DEFAULT_RUBRIC)

        assert score.raw_score <= score.max_score

    def test_rationale_mentions_years(
            self, agent: ScorerAgent, sample_match: MatchResult
    ) -> None:
        score = agent._score_experience(sample_match, DEFAULT_RUBRIC)
        assert "4.5" in score.rationale
        assert "3" in score.rationale  # required years


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 2: Unit — LLM response parsing
# ──────────────────────────────────────────────────────────────────────────────

class TestParseLlmScores:
    """_parse_llm_scores() convert LLM JSON → DimensionScore objects."""

    def _llm_dims(self) -> dict:
        return {
            "education": DimensionConfig(weight=15, description="...", scored_by="llm"),
            "achievements": DimensionConfig(weight=20, description="...", scored_by="llm"),
            "soft_skills": DimensionConfig(weight=10, description="...", scored_by="llm"),
        }

    def test_parses_all_dimensions(self, agent: ScorerAgent) -> None:
        scores, strengths, concerns, recommendation = agent._parse_llm_scores(
            VALID_LLM_JSON, self._llm_dims(), DEFAULT_RUBRIC
        )
        assert "education" in scores
        assert "achievements" in scores
        assert "soft_skills" in scores

    def test_clamps_score_above_max(self, agent: ScorerAgent) -> None:
        """LLM cho điểm vượt max → tự clamp về max."""
        data = {
            "dimension_scores": [
                {
                    "dimension": "education", "max_score": 15,
                    "raw_score": 999,  # Vô lý
                    "percentage": 100, "rationale": "...", "scored_by": "llm",
                }
            ],
            "strengths": [], "concerns": [], "recommendation": "...",
        }
        scores, *_ = agent._parse_llm_scores(data, self._llm_dims(), DEFAULT_RUBRIC)
        assert scores["education"].raw_score <= 15

    def test_fallback_for_missing_dimension(self, agent: ScorerAgent) -> None:
        """LLM bỏ sót 1 dimension → fallback score 40%."""
        data = {
            "dimension_scores": [
                # Chỉ có education, thiếu achievements và soft_skills
                {
                    "dimension": "education", "max_score": 15,
                    "raw_score": 12, "percentage": 80,
                    "rationale": "...", "scored_by": "llm",
                }
            ],
            "strengths": [], "concerns": [], "recommendation": "...",
        }
        scores, *_ = agent._parse_llm_scores(data, self._llm_dims(), DEFAULT_RUBRIC)
        assert "achievements" in scores
        assert scores["achievements"].percentage == 40.0
        assert "conservatively" in scores["achievements"].rationale

    def test_ignores_unexpected_dimensions(self, agent: ScorerAgent) -> None:
        """LLM trả về dimension không yêu cầu → bỏ qua."""
        data = {
            "dimension_scores": [
                {
                    "dimension": "made_up_dimension", "max_score": 50,
                    "raw_score": 45, "percentage": 90,
                    "rationale": "...", "scored_by": "llm",
                }
            ],
            "strengths": [], "concerns": [], "recommendation": "...",
        }
        scores, *_ = agent._parse_llm_scores(data, self._llm_dims(), DEFAULT_RUBRIC)
        assert "made_up_dimension" not in scores

    def test_strengths_limited_to_3(self, agent: ScorerAgent) -> None:
        data = {**VALID_LLM_JSON, "strengths": ["A", "B", "C", "D", "E"]}
        _, strengths, *_ = agent._parse_llm_scores(data, self._llm_dims(), DEFAULT_RUBRIC)
        assert len(strengths) <= 3

    def test_concerns_limited_to_3(self, agent: ScorerAgent) -> None:
        data = {**VALID_LLM_JSON, "concerns": ["X", "Y", "Z", "W"]}
        _, _, concerns, _ = agent._parse_llm_scores(data, self._llm_dims(), DEFAULT_RUBRIC)
        assert len(concerns) <= 3


class TestParseJsonResponse:
    def test_clean_json(self, agent: ScorerAgent) -> None:
        raw = json.dumps({"dimension_scores": []})
        assert agent._parse_json_response(raw) == {"dimension_scores": []}

    def test_strips_markdown(self, agent: ScorerAgent) -> None:
        raw = "```json\n" + json.dumps({"x": 1}) + "\n```"
        assert agent._parse_json_response(raw) == {"x": 1}

    def test_raises_on_no_json(self, agent: ScorerAgent) -> None:
        with pytest.raises(ValueError, match="No JSON object found"):
            agent._parse_json_response("No JSON here.")


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 3: Integration — full score() và score_and_rank()
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreIntegration:

    @patch("agents.scorer.ScorerAgent._call_llm")
    @patch("agents.scorer.get_rubric")
    def test_score_success(
            self,
            mock_rubric: MagicMock,
            mock_llm: MagicMock,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        """Full pipeline: MatchResult + CandidateProfile → CandidateScore."""
        mock_rubric.return_value = None  # Dùng DEFAULT_RUBRIC
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        score = agent.score(sample_match, sample_candidate, "backend-2025")

        assert isinstance(score, CandidateScore)
        assert score.candidate_name == "Nguyen Van A"
        assert 0 <= score.total_score <= 100
        assert score.breakdown.technical_skills is not None
        assert score.breakdown.experience is not None
        assert score.breakdown.education is not None
        assert score.breakdown.achievements is not None
        assert score.breakdown.soft_skills is not None

    @patch("agents.scorer.ScorerAgent._call_llm")
    @patch("agents.scorer.get_rubric")
    def test_score_total_is_sum_of_dimensions(
            self,
            mock_rubric: MagicMock,
            mock_llm: MagicMock,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        """Total score = sum của tất cả dimension scores."""
        mock_rubric.return_value = None
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        score = agent.score(sample_match, sample_candidate, "backend-2025")

        expected_total = sum(
            ds.raw_score for ds in score.breakdown.as_list()
        )
        assert abs(score.total_score - round(expected_total, 2)) < 0.01

    @patch("agents.scorer.ScorerAgent._call_llm")
    @patch("agents.scorer.get_rubric")
    def test_score_tier_assigned_correctly(
            self,
            mock_rubric: MagicMock,
            mock_llm: MagicMock,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        """Tier tự động gán dựa trên total_score."""
        mock_rubric.return_value = None
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        score = agent.score(sample_match, sample_candidate, "backend-2025")

        if score.total_score >= 80:
            assert score.tier == ScoreTier.STRONG
        elif score.total_score >= 60:
            assert score.tier == ScoreTier.GOOD
        elif score.total_score >= 40:
            assert score.tier == ScoreTier.MODERATE
        else:
            assert score.tier == ScoreTier.WEAK

    @patch("agents.scorer.ScorerAgent._call_llm")
    @patch("agents.scorer.get_rubric")
    def test_score_inherits_low_confidence(
            self,
            mock_rubric: MagicMock,
            mock_llm: MagicMock,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        """low_confidence từ MatchResult truyền sang CandidateScore."""
        sample_match.low_confidence = True
        sample_match.warnings = ["CV parsed with LOW confidence"]
        mock_rubric.return_value = None
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        score = agent.score(sample_match, sample_candidate, "backend-2025")

        assert score.low_confidence is True
        assert any("LOW confidence" in w for w in score.warnings)

    @patch("agents.scorer.ScorerAgent._call_llm")
    @patch("agents.scorer.get_rubric")
    def test_score_retries_on_bad_json(
            self,
            mock_rubric: MagicMock,
            mock_llm: MagicMock,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        mock_rubric.return_value = None
        mock_llm.side_effect = [
            "not json",  # Lần 1: fail
            json.dumps(VALID_LLM_JSON),  # Lần 2: thành công
        ]

        with patch("agents.scorer.time.sleep"):
            score = agent.score(sample_match, sample_candidate, "backend-2025")

        assert score.candidate_name == "Nguyen Van A"
        assert mock_llm.call_count == 2


class TestScoreAndRank:

    @patch("agents.scorer.ScorerAgent._call_llm")
    @patch("agents.scorer.get_rubric")
    def test_rank_order_by_score_descending(
            self,
            mock_rubric: MagicMock,
            mock_llm: MagicMock,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        """Rank 1 phải là candidate có điểm cao nhất."""
        mock_rubric.return_value = None
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        with patch("agents.scorer.time.sleep"):
            ranked = agent.score_and_rank(
                matches=[sample_match, sample_match, sample_match],
                candidates=[sample_candidate, sample_candidate, sample_candidate],
                job_id="backend-2025",
            )

        assert len(ranked) == 3
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3
        # Scores giảm dần
        assert ranked[0].score.total_score >= ranked[1].score.total_score
        assert ranked[1].score.total_score >= ranked[2].score.total_score

    @patch("agents.scorer.ScorerAgent._call_llm")
    @patch("agents.scorer.get_rubric")
    def test_percentile_top_candidate_is_100(
            self,
            mock_rubric: MagicMock,
            mock_llm: MagicMock,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        """Candidate rank 1 luôn có percentile 100."""
        mock_rubric.return_value = None
        mock_llm.return_value = json.dumps(VALID_LLM_JSON)

        with patch("agents.scorer.time.sleep"):
            ranked = agent.score_and_rank(
                matches=[sample_match, sample_match],
                candidates=[sample_candidate, sample_candidate],
                job_id="backend-2025",
            )

        assert ranked[0].percentile == 100.0

    def test_raises_when_lengths_mismatch(
            self,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        """Lỗi rõ ràng khi matches và candidates không cùng độ dài."""
        with pytest.raises(ValueError, match="must have the same length"):
            agent.score_and_rank(
                matches=[sample_match, sample_match],
                candidates=[sample_candidate],  # Khác length
                job_id="backend-2025",
            )

    @patch("agents.scorer.ScorerAgent.score")
    @patch("agents.scorer.get_rubric")
    def test_batch_continues_after_one_fails(
            self,
            mock_rubric: MagicMock,
            mock_score: MagicMock,
            agent: ScorerAgent,
            sample_match: MatchResult,
            sample_candidate: CandidateProfile,
    ) -> None:
        """1 candidate fail → trả về fallback score, batch không dừng."""
        mock_rubric.return_value = None
        good_score = CandidateScore(
            candidate_name="Nguyen Van A",
            job_title="Senior Backend Engineer",
            total_score=75.0,
            rubric_used="backend-2025",
        )
        mock_score.side_effect = [
            good_score,
            RuntimeError("Unexpected error"),
            good_score,
        ]

        with patch("agents.scorer.time.sleep"):
            ranked = agent.score_and_rank(
                matches=[sample_match, sample_match, sample_match],
                candidates=[sample_candidate, sample_candidate, sample_candidate],
                job_id="backend-2025",
            )

        assert len(ranked) == 3
        # Candidate fail có total_score = 0 → rank cuối
        assert ranked[-1].score.total_score == 0.0
        assert ranked[-1].score.low_confidence is True


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 4: Schema validation
# ──────────────────────────────────────────────────────────────────────────────

class TestScoringRubricSchema:

    def test_valid_rubric_passes(self) -> None:
        """Rubric hợp lệ (weights = 100) không raise."""
        rubric = ScoringRubric(
            job_id="test-job",
            job_title="Test Job",
            dimensions={
                "technical_skills": DimensionConfig(weight=60, description="...", scored_by="programmatic"),
                "experience": DimensionConfig(weight=40, description="...", scored_by="programmatic"),
            },
        )
        assert rubric.dimensions["technical_skills"].weight == 60

    def test_invalid_weight_sum_raises(self) -> None:
        """Weights tổng != 100 phải raise ValidationError."""
        with pytest.raises(Exception, match="sum to 100"):
            ScoringRubric(
                job_id="test-job",
                job_title="Test Job",
                dimensions={
                    "technical_skills": DimensionConfig(weight=50, description="...", scored_by="programmatic"),
                    "experience": DimensionConfig(weight=30, description="...", scored_by="programmatic"),
                    # Tổng = 80, không phải 100
                },
            )

    def test_default_rubric_is_valid(self) -> None:
        """DEFAULT_RUBRIC phải hợp lệ."""
        total = sum(d.weight for d in DEFAULT_RUBRIC.dimensions.values())
        assert total == 100

    def test_score_tier_boundaries(self) -> None:
        """ScoreTier.from_score() phân loại đúng."""
        assert ScoreTier.from_score(80) == ScoreTier.STRONG
        assert ScoreTier.from_score(79) == ScoreTier.GOOD
        assert ScoreTier.from_score(60) == ScoreTier.GOOD
        assert ScoreTier.from_score(59) == ScoreTier.MODERATE
        assert ScoreTier.from_score(40) == ScoreTier.MODERATE
        assert ScoreTier.from_score(39) == ScoreTier.WEAK
        assert ScoreTier.from_score(0) == ScoreTier.WEAK

    def test_dimension_score_clamps_raw(self) -> None:
        """raw_score > max_score tự clamp về max."""
        ds = DimensionScore(
            dimension="test",
            max_score=20,
            raw_score=999,
            percentage=100.0,
            rationale="...",
            scored_by="llm",
        )
        assert ds.raw_score <= 20

    def test_candidate_score_auto_tier(self) -> None:
        """Tier tự động gán sau khi validate."""
        score = CandidateScore(
            candidate_name="Test",
            job_title="Dev",
            total_score=85.0,
            rubric_used="test",
        )
        assert score.tier == ScoreTier.STRONG

    def test_ranked_candidate_percentile_clamped(self) -> None:
        """Percentile không vượt 100."""
        score = CandidateScore(
            candidate_name="X", job_title="Y",
            total_score=100.0, rubric_used="z"
        )
        ranked = RankedCandidate(rank=1, percentile=150.0, score=score)
        assert ranked.percentile == 100.0