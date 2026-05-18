"""
Agent 4: Report Writer Agent

Nhận list[RankedCandidate] từ Scorer Agent (Agent 3)
→ Tạo PDF report chuyên nghiệp
→ Upload S3 (hoặc lưu local nếu dev)
→ Tạo text summary cho Chainlit hiển thị inline
→ Trả về ReportOutput cho FastAPI endpoint

Agent này KHÔNG gọi LLM.
Toàn bộ logic là deterministic:
  - PDF layout: reportlab
  - S3 upload: boto3
  - Summary text: template-based từ data có sẵn

Tại sao không dùng LLM để viết summary?
  - Tất cả thông tin cần thiết đã có trong RankedCandidate
  - Template nhất quán hơn, không drift giữa các lần chạy
  - Tiết kiệm API quota cho các bước quan trọng hơn
  - Nhanh hơn: tạo report 50 CVs trong < 5 giây
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from schemas.report_schema import ReportMeta, ReportOutput
from schemas.score_schema import RankedCandidate, ScoreTier
from tools.pdf_generator import generate_pdf
from tools.s3_uploader import save_local, upload_report

logger = logging.getLogger(__name__)


class ReportWriterAgent:
    """
    Agent 4: Generate PDF report from ranked candidates.

    Usage:
        agent = ReportWriterAgent(use_s3=True)  # False khi dev local
        output = agent.write(
            ranked    = ranked_candidates,   # list[RankedCandidate] từ ScorerAgent
            job_title = "Senior Backend Engineer",
            job_id    = "backend-2025",
        )
        print(output.summary_text)   # Hiển thị trên Chainlit
        print(output.s3_url)         # Download link
        # hoặc dùng output.pdf_bytes trực tiếp
    """

    def __init__(
        self,
        use_s3:          bool = False,
        local_output_dir: str = "reports",
        shortlist_count:  int = 5,
    ) -> None:
        self.use_s3           = use_s3
        self.local_output_dir = local_output_dir
        self.shortlist_count  = shortlist_count

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def write(
        self,
        ranked:    list[RankedCandidate],
        job_title: str,
        job_id:    str,
    ) -> ReportOutput:
        """
        Full pipeline: list[RankedCandidate] → ReportOutput.

        Args:
            ranked:    Sorted list từ ScorerAgent.score_and_rank()
            job_title: Tên vị trí
            job_id:    ID của JD

        Returns:
            ReportOutput với pdf_bytes, s3_url, summary_text
        """
        if not ranked:
            raise ValueError("Cannot generate report for empty candidate list")

        report_id = str(uuid.uuid4())[:8].upper()
        logger.info(
            "Generating report %s for '%s' (%d candidates)",
            report_id, job_title, len(ranked),
        )

        # Bước 1: Tạo PDF
        pdf_bytes = generate_pdf(
            ranked          = ranked,
            job_title       = job_title,
            job_id          = job_id,
            shortlist_count = self.shortlist_count,
        )
        logger.info("PDF generated: %.1f KB", len(pdf_bytes) / 1024)

        # Bước 2: Upload hoặc lưu local
        s3_url = None
        if self.use_s3:
            s3_url = upload_report(pdf_bytes, job_id, report_id)

        if not s3_url:
            # Fallback: lưu local (dev mode hoặc S3 failed)
            local_path = save_local(pdf_bytes, job_id, report_id, self.local_output_dir)
            logger.info("Report saved locally: %s", local_path)

        # Bước 3: Tạo summary text cho Chainlit
        summary = self._build_summary_text(ranked, job_title, report_id)

        # Bước 4: Build output
        shortlist = [
            r.score.candidate_name
            for r in ranked[:self.shortlist_count]
            if r.score.tier in {ScoreTier.STRONG, ScoreTier.GOOD}
        ]

        meta = ReportMeta(
            report_id         = report_id,
            job_id            = job_id,
            job_title         = job_title,
            total_candidates  = len(ranked),
            shortlist_count   = len(shortlist),
        )

        output = ReportOutput(
            meta         = meta,
            pdf_bytes    = pdf_bytes,
            s3_url       = s3_url,
            summary_text = summary,
            shortlist    = shortlist,
        )

        logger.info(
            "Report complete: ID=%s | %d candidates | shortlist=%d | size=%.1fKB",
            report_id, len(ranked), len(shortlist), len(pdf_bytes) / 1024,
        )
        return output

    # ──────────────────────────────────────────────────────────────────────────
    # Private: Summary text builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_summary_text(
        self,
        ranked:    list[RankedCandidate],
        job_title: str,
        report_id: str,
    ) -> str:
        """
        Tạo text summary hiển thị trên Chainlit sau khi report xong.
        Template-based, không cần LLM — deterministic và nhanh.
        """
        total    = len(ranked)
        strong   = [r for r in ranked if r.score.tier == ScoreTier.STRONG]
        good     = [r for r in ranked if r.score.tier == ScoreTier.GOOD]
        moderate = [r for r in ranked if r.score.tier == ScoreTier.MODERATE]
        weak     = [r for r in ranked if r.score.tier == ScoreTier.WEAK]

        top = ranked[:self.shortlist_count]
        top_names = ", ".join(r.score.candidate_name for r in top)

        low_conf = [r for r in ranked if r.score.low_confidence]
        warn_line = (
            f"\n⚠ {len(low_conf)} candidate(s) had low-confidence CV parsing "
            f"and should be manually reviewed."
            if low_conf else ""
        )

        return (
            f"📊 **Screening Report — {job_title}**\n"
            f"Report ID: `{report_id}`\n\n"
            f"**{total} candidates** evaluated:\n"
            f"  🟢 Strong match:   {len(strong)}\n"
            f"  🔵 Good match:     {len(good)}\n"
            f"  🟡 Moderate:       {len(moderate)}\n"
            f"  🔴 Weak match:     {len(weak)}\n\n"
            f"**Top {len(top)} shortlist:**\n{top_names}\n"
            f"{warn_line}"
        )