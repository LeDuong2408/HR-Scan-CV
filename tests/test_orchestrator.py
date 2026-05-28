"""
Tests for LangGraph Orchestrator.

Run: pytest tests/test_orchestrator.py -v

4 tầng:
  Unit Nodes:      test từng node function độc lập (mock agents)
  Unit Edges:      test conditional edge functions
  Integration:     test toàn bộ graph.invoke() với mock agents
  State:           test GraphState schema và reducers
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from graph.nodes import (
    match_node,
    parse_node,
    report_node,
    score_node,
    should_continue_after_match,
    should_continue_after_parse,
    should_continue_after_score,
)
from graph.state import GraphState, PipelineStatus
from graph.workflow import build_graph, run_pipeline
from schemas.cv_schema import CandidateProfile, ParseConfidence
from schemas.match_schema import (
    ExperienceAssessment,
    MatchResult,
    SkillGapReport,
)
from schemas.report_schema import ReportMeta, ReportOutput
from schemas.score_schema import (
    CandidateScore,
    RankedCandidate,
    ScoreTier,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> GraphState:
    """State cơ bản với đủ input để chạy pipeline."""
    return GraphState(
        file_paths = ["cv1.pdf", "cv2.pdf"],
        job_id     = "backend-2025",
        job_title  = "Senior Backend Engineer",
        api_key    = "fake-key",
        status     = PipelineStatus.PARSING,
    )


@pytest.fixture
def sample_candidate() -> CandidateProfile:
    return CandidateProfile(
        full_name              = "Nguyen Van A",
        total_experience_years = 4.5,
        technical_skills       = ["Python", "FastAPI"],
        confidence             = ParseConfidence.HIGH,
    )


@pytest.fixture
def sample_match(sample_candidate) -> MatchResult:
    return MatchResult(
        candidate_name = sample_candidate.full_name,
        job_title      = "Senior Backend Engineer",
        skill_gap      = SkillGapReport(
            matched=["Python", "FastAPI"],
            missing_critical=[],
        ),
        experience = ExperienceAssessment(
            required_years=3.0,
            candidate_years=4.5,
            meets_requirement=True,
            domain_relevance=1.0,
        ),
        match_summary = "Strong match.",
    )


@pytest.fixture
def sample_ranked(sample_candidate, sample_match) -> list[RankedCandidate]:
    score = CandidateScore(
        candidate_name = "Nguyen Van A",
        job_title      = "Senior Backend Engineer",
        total_score    = 82.0,
        tier           = ScoreTier.STRONG,
        strengths      = ["Python", "AWS cert"],
        concerns       = ["No Kubernetes"],
        recommendation = "Recommend for interview.",
        rubric_used    = "backend-2025",
    )
    return [RankedCandidate(rank=1, percentile=100.0, score=score)]


@pytest.fixture
def sample_report() -> ReportOutput:
    return ReportOutput(
        meta      = ReportMeta(
            report_id        = "ABC12345",
            job_id           = "backend-2025",
            job_title        = "Senior Backend Engineer",
            total_candidates = 1,
            shortlist_count  = 1,
        ),
        pdf_bytes    = b"%PDF-1.4 fake",
        summary_text = "1 candidate evaluated. Top: Nguyen Van A.",
        shortlist    = ["Nguyen Van A"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 1: Unit — test từng node function
# ──────────────────────────────────────────────────────────────────────────────

class TestParseNode:

    @patch("graph.nodes.CVParserAgent")
    def test_parse_success_updates_state(
        self,
        MockAgent:       MagicMock,
        base_state:      GraphState,
        sample_candidate: CandidateProfile,
    ) -> None:
        """Parse thành công → parsed_candidates, status=MATCHING."""
        mock_instance = MockAgent.return_value
        mock_instance.parse.return_value = sample_candidate

        result = parse_node(base_state)

        assert result["status"] == PipelineStatus.MATCHING
        assert len(result["parsed_candidates"]) == 2  # 2 files
        assert result["progress"] == 0.25

    @patch("graph.nodes.CVParserAgent")
    def test_parse_partial_failure_continues(
        self,
        MockAgent:       MagicMock,
        base_state:      GraphState,
        sample_candidate: CandidateProfile,
    ) -> None:
        """1 file fail, 1 file OK → vẫn tiếp tục với file OK."""
        mock_instance = MockAgent.return_value
        mock_instance.parse.side_effect = [
            sample_candidate,          # cv1.pdf OK
            Exception("Corrupt file"), # cv2.pdf fail
        ]

        result = parse_node(base_state)

        # Vẫn tiếp tục vì có ít nhất 1 file OK
        assert result["status"] == PipelineStatus.MATCHING
        assert len(result["parsed_candidates"]) == 1
        assert len(result["parse_errors"]) == 1

    @patch("graph.nodes.CVParserAgent")
    def test_parse_all_fail_returns_failed_status(
        self,
        MockAgent:  MagicMock,
        base_state: GraphState,
    ) -> None:
        """Tất cả files fail → status=FAILED, pipeline dừng sớm."""
        mock_instance = MockAgent.return_value
        mock_instance.parse.side_effect = Exception("All broken")

        result = parse_node(base_state)

        assert result["status"] == PipelineStatus.FAILED
        assert len(result["errors"]) > 0

    @patch("graph.nodes.CVParserAgent")
    def test_parse_logs_progress_message(
        self,
        MockAgent:        MagicMock,
        base_state:       GraphState,
        sample_candidate: CandidateProfile,
    ) -> None:
        """current_step được set để Chainlit có thể stream."""
        mock_instance = MockAgent.return_value
        mock_instance.parse.return_value = sample_candidate

        result = parse_node(base_state)

        assert "current_step" in result
        assert len(result["current_step"]) > 0


class TestMatchNode:

    @patch("graph.nodes.JDMatcherAgent")
    def test_match_success_updates_state(
        self,
        MockAgent:    MagicMock,
        base_state:   GraphState,
        sample_candidate: CandidateProfile,
        sample_match: MatchResult,
    ) -> None:
        """Match thành công → match_results, status=SCORING."""
        base_state.parsed_candidates = [sample_candidate]
        mock_instance = MockAgent.return_value
        mock_instance.match_batch.return_value = [sample_match]

        result = match_node(base_state)

        assert result["status"] == PipelineStatus.SCORING
        assert len(result["match_results"]) == 1
        assert result["progress"] == 0.55

    @patch("graph.nodes.JDMatcherAgent")
    def test_match_empty_candidates_fails(
        self,
        MockAgent:  MagicMock,
        base_state: GraphState,
    ) -> None:
        """parsed_candidates rỗng → status=FAILED ngay."""
        base_state.parsed_candidates = []

        result = match_node(base_state)

        assert result["status"] == PipelineStatus.FAILED
        MockAgent.return_value.match_batch.assert_not_called()


class TestScoreNode:

    @patch("graph.nodes.ScorerAgent")
    def test_score_success_updates_state(
        self,
        MockAgent:       MagicMock,
        base_state:      GraphState,
        sample_match:    MatchResult,
        sample_candidate: CandidateProfile,
        sample_ranked:   list[RankedCandidate],
    ) -> None:
        """Score thành công → ranked_candidates, status=REPORTING."""
        base_state.match_results      = [sample_match]
        base_state.parsed_candidates  = [sample_candidate]
        mock_instance = MockAgent.return_value
        mock_instance.score_and_rank.return_value = sample_ranked

        result = score_node(base_state)

        assert result["status"] == PipelineStatus.REPORTING
        assert len(result["ranked_candidates"]) == 1
        assert result["progress"] == 0.80

    @patch("graph.nodes.ScorerAgent")
    def test_score_empty_matches_fails(
        self,
        MockAgent:  MagicMock,
        base_state: GraphState,
    ) -> None:
        base_state.match_results = []
        result = score_node(base_state)
        assert result["status"] == PipelineStatus.FAILED


class TestReportNode:

    @patch("graph.nodes.ReportWriterAgent")
    def test_report_success_updates_state(
        self,
        MockAgent:      MagicMock,
        base_state:     GraphState,
        sample_ranked:  list[RankedCandidate],
        sample_report:  ReportOutput,
    ) -> None:
        """Report thành công → report object, status=COMPLETED."""
        base_state.ranked_candidates = sample_ranked
        mock_instance = MockAgent.return_value
        mock_instance.write.return_value = sample_report

        result = report_node(base_state)

        assert result["status"] == PipelineStatus.COMPLETED
        assert result["report"]  == sample_report
        assert result["progress"] == 1.0

    @patch("graph.nodes.ReportWriterAgent")
    def test_report_empty_ranked_fails(
        self,
        MockAgent:  MagicMock,
        base_state: GraphState,
    ) -> None:
        base_state.ranked_candidates = []
        result = report_node(base_state)
        assert result["status"] == PipelineStatus.FAILED


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 2: Unit — conditional edges
# ──────────────────────────────────────────────────────────────────────────────

class TestConditionalEdges:

    def test_parse_success_routes_to_match(self, base_state: GraphState) -> None:
        base_state.status = PipelineStatus.MATCHING
        assert should_continue_after_parse(base_state) == "match"

    def test_parse_failed_routes_to_end(self, base_state: GraphState) -> None:
        base_state.status = PipelineStatus.FAILED
        assert should_continue_after_parse(base_state) == "failed"

    def test_match_success_routes_to_score(self, base_state: GraphState) -> None:
        base_state.status       = PipelineStatus.SCORING
        base_state.match_results = []
        assert should_continue_after_match(base_state) == "score"

    def test_match_failed_routes_to_end(self, base_state: GraphState) -> None:
        base_state.status = PipelineStatus.FAILED
        assert should_continue_after_match(base_state) == "failed"

    def test_score_success_routes_to_report(self, base_state: GraphState) -> None:
        base_state.status = PipelineStatus.REPORTING
        assert should_continue_after_score(base_state) == "report"

    def test_score_failed_routes_to_end(self, base_state: GraphState) -> None:
        base_state.status = PipelineStatus.FAILED
        assert should_continue_after_score(base_state) == "failed"


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 3: Integration — full graph.invoke()
# ──────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:

    @patch("graph.nodes.ReportWriterAgent")
    @patch("graph.nodes.ScorerAgent")
    @patch("graph.nodes.JDMatcherAgent")
    @patch("graph.nodes.CVParserAgent")
    def test_happy_path_end_to_end(
        self,
        MockParser:  MagicMock,
        MockMatcher: MagicMock,
        MockScorer:  MagicMock,
        MockWriter:  MagicMock,
        sample_candidate: CandidateProfile,
        sample_match:     MatchResult,
        sample_ranked:    list[RankedCandidate],
        sample_report:    ReportOutput,
    ) -> None:
        """
        Happy path: tất cả nodes thành công → state.status = COMPLETED.
        Đây là test quan trọng nhất — verify toàn bộ graph kết nối đúng.
        """
        MockParser.return_value.parse.return_value            = sample_candidate
        MockMatcher.return_value.match_batch.return_value     = [sample_match]
        MockScorer.return_value.score_and_rank.return_value   = sample_ranked
        MockWriter.return_value.write.return_value            = sample_report

        graph = build_graph()
        final = run_pipeline(
            graph      = graph,
            file_paths = ["cv1.pdf"],
            job_id     = "backend-2025",
            jd_text    = "Test",
            job_title  = "Senior Backend Engineer",
            api_key    = "fake-key",
        )

        assert final.status            == PipelineStatus.COMPLETED
        assert final.report            == sample_report
        assert final.progress          == 1.0
        assert len(final.parsed_candidates) == 1
        assert len(final.match_results)     == 1
        assert len(final.ranked_candidates) == 1

    @patch("graph.nodes.CVParserAgent")
    def test_pipeline_stops_at_failed_parse(
        self,
        MockParser: MagicMock,
    ) -> None:
        """Nếu parse fail hoàn toàn → graph kết thúc ở END, không chạy tiếp."""
        MockParser.return_value.parse.side_effect = Exception("Cannot read file")

        graph = build_graph()
        final = run_pipeline(
            graph      = graph,
            file_paths = ["bad.pdf"],
            job_id     = "backend-2025",
            jd_text    = "Test",
            job_title  = "Dev",
            api_key    = "fake-key",
        )

        assert final.status == PipelineStatus.FAILED
        assert final.report is None  # Không có report

    @patch("graph.nodes.ReportWriterAgent")
    @patch("graph.nodes.ScorerAgent")
    @patch("graph.nodes.JDMatcherAgent")
    @patch("graph.nodes.CVParserAgent")
    def test_pipeline_propagates_parse_errors_to_final_state(
        self,
        MockParser:  MagicMock,
        MockMatcher: MagicMock,
        MockScorer:  MagicMock,
        MockWriter:  MagicMock,
        sample_candidate: CandidateProfile,
        sample_match:     MatchResult,
        sample_ranked:    list[RankedCandidate],
        sample_report:    ReportOutput,
    ) -> None:
        """Parse errors được tích lũy vào state.parse_errors đến cuối pipeline."""
        MockParser.return_value.parse.side_effect = [
            sample_candidate,          # cv1.pdf OK
            Exception("Corrupt"),      # cv2.pdf fail
        ]
        MockMatcher.return_value.match_batch.return_value   = [sample_match]
        MockScorer.return_value.score_and_rank.return_value = sample_ranked
        MockWriter.return_value.write.return_value          = sample_report

        graph = build_graph()
        final = run_pipeline(
            graph      = graph,
            file_paths = ["cv1.pdf", "cv2.pdf"],
            job_id     = "backend-2025",
            jd_text    = "Test",
            job_title  = "Dev",
            api_key    = "fake-key",
        )

        # Pipeline hoàn thành nhưng có parse_errors được ghi lại
        assert final.status == PipelineStatus.COMPLETED
        assert len(final.parse_errors) == 1
        assert "cv2.pdf" in final.parse_errors[0]

    def test_run_pipeline_raises_on_empty_files(self) -> None:
        """Không có files → ValueError ngay, không chạy graph."""
        graph = build_graph()
        with pytest.raises(ValueError, match="file_paths cannot be empty"):
            run_pipeline(
                graph=graph, file_paths=[],
                job_id="j", job_title="t", api_key="k",
                jd_text= "Test",
            )

    @patch("graph.nodes.ReportWriterAgent")
    @patch("graph.nodes.ScorerAgent")
    @patch("graph.nodes.JDMatcherAgent")
    @patch("graph.nodes.CVParserAgent")
    def test_stream_pipeline_yields_node_events(
        self,
        MockParser:  MagicMock,
        MockMatcher: MagicMock,
        MockScorer:  MagicMock,
        MockWriter:  MagicMock,
        sample_candidate: CandidateProfile,
        sample_match:     MatchResult,
        sample_ranked:    list[RankedCandidate],
        sample_report:    ReportOutput,
    ) -> None:
        """stream_pipeline() yield đúng số events."""
        from graph.workflow import stream_pipeline

        MockParser.return_value.parse.return_value            = sample_candidate
        MockMatcher.return_value.match_batch.return_value     = [sample_match]
        MockScorer.return_value.score_and_rank.return_value   = sample_ranked
        MockWriter.return_value.write.return_value            = sample_report

        graph  = build_graph()
        events = list(stream_pipeline(
            graph=graph, file_paths=["cv1.pdf"],
            job_id="backend-2025", job_title="Dev", api_key="fake",
        ))

        node_names = [e["node"] for e in events]
        # Tất cả 4 nodes phải xuất hiện trong events
        assert "parse"  in node_names
        assert "match"  in node_names
        assert "score"  in node_names
        assert "report" in node_names


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 4: State schema tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGraphState:

    def test_default_state_is_pending(self) -> None:
        state = GraphState()
        assert state.status   == PipelineStatus.PENDING
        assert state.progress == 0.0

    def test_lists_default_empty(self) -> None:
        state = GraphState()
        assert state.file_paths         == []
        assert state.parsed_candidates  == []
        assert state.match_results      == []
        assert state.ranked_candidates  == []
        assert state.errors             == []
        assert state.parse_errors       == []

    def test_report_default_none(self) -> None:
        state = GraphState()
        assert state.report is None

    def test_state_with_full_input(self) -> None:
        state = GraphState(
            file_paths = ["a.pdf", "b.pdf"],
            job_id     = "backend-2025",
            job_title  = "Dev",
            api_key    = "key-123",
        )
        assert len(state.file_paths) == 2
        assert state.job_id          == "backend-2025"