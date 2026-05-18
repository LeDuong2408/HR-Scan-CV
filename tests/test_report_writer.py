"""
Tests for ReportWriterAgent.

Run: pytest tests/test_report_writer.py -v

3 tầng:
  Unit:        test _build_summary_text (template logic)
  Integration: test write() với mock S3 và mock pdf_generator
  PDF smoke:   test generate_pdf() tạo ra bytes hợp lệ (không mock)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.report_writer import ReportWriterAgent
from schemas.report_schema import ReportOutput
from schemas.score_schema import (
    CandidateScore,
    DimensionScore,
    RankedCandidate,
    ScoreBreakdown,
    ScoreTier,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_score(
    name:  str,
    total: float,
    tier:  str = "",
    low_confidence: bool = False,
) -> CandidateScore:
    """Helper tạo CandidateScore với breakdown đầy đủ."""
    tech = DimensionScore(
        dimension="technical_skills", max_score=35,
        raw_score=total * 0.35, percentage=total,
        rationale="Test rationale for technical skills dimension.",
        scored_by="programmatic",
    )
    exp = DimensionScore(
        dimension="experience", max_score=20,
        raw_score=total * 0.20, percentage=total,
        rationale="Test rationale for experience dimension.",
        scored_by="programmatic",
    )
    edu = DimensionScore(
        dimension="education", max_score=15,
        raw_score=total * 0.15, percentage=total,
        rationale="Test rationale for education dimension.",
        scored_by="llm",
    )
    ach = DimensionScore(
        dimension="achievements", max_score=20,
        raw_score=total * 0.20, percentage=total,
        rationale="Test rationale for achievements dimension.",
        scored_by="llm",
    )
    soft = DimensionScore(
        dimension="soft_skills", max_score=10,
        raw_score=total * 0.10, percentage=total,
        rationale="Test rationale for soft skills dimension.",
        scored_by="llm",
    )
    return CandidateScore(
        candidate_name = name,
        job_title      = "Senior Backend Engineer",
        total_score    = total,
        tier           = tier or ScoreTier.from_score(total),
        breakdown      = ScoreBreakdown(
            technical_skills = tech,
            experience       = exp,
            education        = edu,
            achievements     = ach,
            soft_skills      = soft,
        ),
        strengths      = ["Strong Python skills", "AWS certified"],
        concerns       = ["Missing Kubernetes"],
        recommendation = "Recommend for interview.",
        low_confidence = low_confidence,
        rubric_used    = "backend-2025",
    )


def _make_ranked(candidates: list[tuple[str, float]]) -> list[RankedCandidate]:
    """
    Helper tạo list[RankedCandidate] từ list of (name, score).
    Tự động sort và assign rank + percentile.
    """
    total   = len(candidates)
    ranked  = []
    for rank_idx, (name, score) in enumerate(candidates, 1):
        percentile = round((total - rank_idx + 1) / total * 100, 1)
        ranked.append(RankedCandidate(
            rank       = rank_idx,
            percentile = percentile,
            score      = _make_score(name, score),
        ))
    return ranked


@pytest.fixture
def agent() -> ReportWriterAgent:
    return ReportWriterAgent(use_s3=False, local_output_dir="/tmp/test_reports")


@pytest.fixture
def sample_ranked() -> list[RankedCandidate]:
    """5 ứng viên với điểm khác nhau, đã sort."""
    return _make_ranked([
        ("Nguyen Van A", 85.0),   # Strong
        ("Tran Thi B",   72.0),   # Good
        ("Le Van C",     65.0),   # Good
        ("Pham Thi D",   48.0),   # Moderate
        ("Hoang Van E",  31.0),   # Weak
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 1: Unit — _build_summary_text
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildSummaryText:

    def test_contains_job_title(
        self, agent: ReportWriterAgent, sample_ranked: list[RankedCandidate]
    ) -> None:
        text = agent._build_summary_text(sample_ranked, "Senior Backend Engineer", "ABC123")
        assert "Senior Backend Engineer" in text

    def test_contains_report_id(
        self, agent: ReportWriterAgent, sample_ranked: list[RankedCandidate]
    ) -> None:
        text = agent._build_summary_text(sample_ranked, "Dev", "XYZ999")
        assert "XYZ999" in text

    def test_contains_total_count(
        self, agent: ReportWriterAgent, sample_ranked: list[RankedCandidate]
    ) -> None:
        text = agent._build_summary_text(sample_ranked, "Dev", "ID1")
        assert "5" in text  # 5 candidates

    def test_tier_counts_correct(
        self, agent: ReportWriterAgent, sample_ranked: list[RankedCandidate]
    ) -> None:
        text = agent._build_summary_text(sample_ranked, "Dev", "ID1")
        # 1 strong, 2 good, 1 moderate, 1 weak
        assert "Strong match:   1" in text
        assert "Good match:     2" in text
        assert "Moderate:       1" in text
        assert "Weak match:     1" in text

    def test_top_names_in_shortlist(
        self, agent: ReportWriterAgent, sample_ranked: list[RankedCandidate]
    ) -> None:
        text = agent._build_summary_text(sample_ranked, "Dev", "ID1")
        assert "Nguyen Van A" in text
        assert "Tran Thi B"   in text

    def test_low_confidence_warning_shown(
        self, agent: ReportWriterAgent
    ) -> None:
        """Candidate với low_confidence → warning xuất hiện."""
        ranked = _make_ranked([("Low Conf User", 70.0)])
        ranked[0].score.low_confidence = True

        text = agent._build_summary_text(ranked, "Dev", "ID1")
        assert "low-confidence" in text.lower()

    def test_no_warning_when_all_confident(
        self, agent: ReportWriterAgent, sample_ranked: list[RankedCandidate]
    ) -> None:
        """Không có low_confidence → không có warning."""
        text = agent._build_summary_text(sample_ranked, "Dev", "ID1")
        assert "low-confidence" not in text.lower()

    def test_empty_shortlist_when_all_weak(
        self, agent: ReportWriterAgent
    ) -> None:
        """Tất cả điểm thấp → shortlist rỗng."""
        ranked = _make_ranked([
            ("Weak A", 20.0),
            ("Weak B", 15.0),
        ])
        text = agent._build_summary_text(ranked, "Dev", "ID1")
        # shortlist chỉ lấy strong/good → trống
        assert "shortlist:" in text.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 2: Integration — write() pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestWriteIntegration:

    @patch("agents.report_writer.generate_pdf")
    @patch("agents.report_writer.save_local")
    def test_write_returns_report_output(
        self,
        mock_save:   MagicMock,
        mock_pdf:    MagicMock,
        agent:       ReportWriterAgent,
        sample_ranked: list[RankedCandidate],
    ) -> None:
        """write() trả về ReportOutput với đủ fields."""
        mock_pdf.return_value  = b"%PDF-1.4 fake pdf content"
        mock_save.return_value = "/tmp/test_reports/backend-2025_ABC.pdf"

        output = agent.write(sample_ranked, "Senior Backend Engineer", "backend-2025")

        assert isinstance(output, ReportOutput)
        assert output.pdf_bytes == b"%PDF-1.4 fake pdf content"
        assert output.meta.job_id    == "backend-2025"
        assert output.meta.job_title == "Senior Backend Engineer"
        assert output.meta.total_candidates == 5

    @patch("agents.report_writer.generate_pdf")
    @patch("agents.report_writer.save_local")
    def test_shortlist_contains_strong_and_good(
        self,
        mock_save: MagicMock,
        mock_pdf:  MagicMock,
        agent:     ReportWriterAgent,
        sample_ranked: list[RankedCandidate],
    ) -> None:
        """Shortlist chỉ gồm strong + good candidates."""
        mock_pdf.return_value  = b"fake pdf"
        mock_save.return_value = "/tmp/report.pdf"

        output = agent.write(sample_ranked, "Senior Backend Engineer", "backend-2025")

        # strong: Nguyen Van A, good: Tran Thi B, Le Van C
        assert "Nguyen Van A" in output.shortlist
        assert "Tran Thi B"   in output.shortlist
        assert "Le Van C"     in output.shortlist
        # moderate/weak không vào shortlist
        assert "Pham Thi D"  not in output.shortlist
        assert "Hoang Van E" not in output.shortlist

    @patch("agents.report_writer.generate_pdf")
    @patch("agents.report_writer.save_local")
    def test_report_id_is_unique(
        self,
        mock_save: MagicMock,
        mock_pdf:  MagicMock,
        agent:     ReportWriterAgent,
        sample_ranked: list[RankedCandidate],
    ) -> None:
        """Mỗi lần gọi write() → report_id khác nhau."""
        mock_pdf.return_value  = b"fake pdf"
        mock_save.return_value = "/tmp/report.pdf"

        out1 = agent.write(sample_ranked, "Dev", "job-1")
        out2 = agent.write(sample_ranked, "Dev", "job-1")

        assert out1.meta.report_id != out2.meta.report_id

    @patch("agents.report_writer.generate_pdf")
    @patch("agents.report_writer.upload_report")
    def test_s3_url_set_when_upload_succeeds(
        self,
        mock_upload: MagicMock,
        mock_pdf:    MagicMock,
        sample_ranked: list[RankedCandidate],
    ) -> None:
        """S3 URL được set khi upload thành công."""
        mock_pdf.return_value    = b"fake pdf"
        mock_upload.return_value = "https://s3.amazonaws.com/bucket/report.pdf"

        agent_s3 = ReportWriterAgent(use_s3=True)
        output   = agent_s3.write(sample_ranked, "Dev", "job-1")

        assert output.s3_url == "https://s3.amazonaws.com/bucket/report.pdf"
        mock_upload.assert_called_once()

    @patch("agents.report_writer.generate_pdf")
    @patch("agents.report_writer.upload_report")
    @patch("agents.report_writer.save_local")
    def test_fallback_to_local_when_s3_fails(
        self,
        mock_save:   MagicMock,
        mock_upload: MagicMock,
        mock_pdf:    MagicMock,
        sample_ranked: list[RankedCandidate],
    ) -> None:
        """S3 upload fail → fallback lưu local, không crash."""
        mock_pdf.return_value    = b"fake pdf"
        mock_upload.return_value = None  # S3 fail → trả về None
        mock_save.return_value   = "/tmp/report.pdf"

        agent_s3 = ReportWriterAgent(use_s3=True)
        output   = agent_s3.write(sample_ranked, "Dev", "job-1")

        # Không crash, s3_url là None, local save được gọi
        assert output.s3_url is None
        mock_save.assert_called_once()

    def test_raises_on_empty_candidates(
        self, agent: ReportWriterAgent
    ) -> None:
        """Không có candidates → ValueError rõ ràng."""
        with pytest.raises(ValueError, match="empty candidate list"):
            agent.write([], "Dev", "job-1")

    @patch("agents.report_writer.generate_pdf")
    @patch("agents.report_writer.save_local")
    def test_generate_pdf_called_with_correct_args(
        self,
        mock_save: MagicMock,
        mock_pdf:  MagicMock,
        agent:     ReportWriterAgent,
        sample_ranked: list[RankedCandidate],
    ) -> None:
        """generate_pdf được gọi đúng arguments."""
        mock_pdf.return_value  = b"fake pdf"
        mock_save.return_value = "/tmp/report.pdf"

        agent.write(sample_ranked, "Senior Backend Engineer", "backend-2025")

        mock_pdf.assert_called_once_with(
            ranked          = sample_ranked,
            job_title       = "Senior Backend Engineer",
            job_id          = "backend-2025",
            shortlist_count = agent.shortlist_count,
        )

    @patch("agents.report_writer.generate_pdf")
    @patch("agents.report_writer.save_local")
    def test_summary_text_in_output(
        self,
        mock_save:     MagicMock,
        mock_pdf:      MagicMock,
        agent:         ReportWriterAgent,
        sample_ranked: list[RankedCandidate],
    ) -> None:
        mock_pdf.return_value  = b"fake pdf"
        mock_save.return_value = "/tmp/report.pdf"

        output = agent.write(sample_ranked, "Senior Backend Engineer", "backend-2025")

        assert len(output.summary_text) > 0
        assert "Senior Backend Engineer" in output.summary_text


# ──────────────────────────────────────────────────────────────────────────────
# Tầng 3: PDF Smoke Test — generate_pdf() thực sự tạo ra PDF hợp lệ
# ──────────────────────────────────────────────────────────────────────────────

class TestPdfSmoke:
    """
    Không mock generate_pdf() — test thực sự tạo PDF.
    Kiểm tra output có phải là bytes hợp lệ không.
    """

    def test_generate_pdf_returns_bytes(
        self, sample_ranked: list[RankedCandidate]
    ) -> None:
        from tools.pdf_generator import generate_pdf

        pdf = generate_pdf(sample_ranked, "Senior Backend Engineer", "backend-2025")

        assert isinstance(pdf, bytes)
        assert len(pdf) > 1000  # PDF thực sự có nội dung

    def test_generate_pdf_starts_with_pdf_header(
        self, sample_ranked: list[RankedCandidate]
    ) -> None:
        """PDF hợp lệ phải bắt đầu với magic bytes '%PDF'."""
        from tools.pdf_generator import generate_pdf

        pdf = generate_pdf(sample_ranked, "Dev", "job-1")
        assert pdf[:4] == b"%PDF"

    def test_generate_pdf_single_candidate(self) -> None:
        """Edge case: chỉ có 1 candidate."""
        from tools.pdf_generator import generate_pdf

        ranked = _make_ranked([("Only One", 70.0)])
        pdf    = generate_pdf(ranked, "Dev", "job-1")
        assert pdf[:4] == b"%PDF"

    def test_generate_pdf_large_batch(self) -> None:
        """PDF vẫn tạo được với 20 candidates."""
        from tools.pdf_generator import generate_pdf

        ranked = _make_ranked([
            (f"Candidate {i:02d}", float(90 - i * 3))
            for i in range(20)
        ])
        pdf = generate_pdf(ranked, "Senior Backend Engineer", "backend-2025")
        assert pdf[:4] == b"%PDF"
        assert len(pdf) > 5000  # Report lớn hơn