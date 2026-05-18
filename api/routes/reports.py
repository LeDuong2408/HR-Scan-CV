"""
Route: /api/v1/reports

Lấy kết quả sau khi pipeline hoàn thành:
  GET /run/{run_id}           — Full report (summary + shortlist + metadata)
  GET /run/{run_id}/download  — Download PDF binary
  GET /run/{run_id}/shortlist — Chỉ lấy shortlist (tên + score + tier)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Response models ────────────────────────────────────────────────────────

class ShortlistCandidate(BaseModel):
    rank:        int
    name:        str
    total_score: float
    tier:        str
    percentile:  float
    recommendation: str | None


class ReportSummaryResponse(BaseModel):
    report_id:        str
    job_id:           str
    job_title:        str
    total_candidates: int
    shortlist_count:  int
    summary_text:     str
    shortlist:        list[ShortlistCandidate]
    has_pdf:          bool
    s3_url:           str | None


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/run/{run_id}", response_model=ReportSummaryResponse)
async def get_report(run_id: str):
    """
    Lấy full report sau khi pipeline completed.
    Gọi sau khi /scan/status trả về status='completed'.
    """
    from api.main import scan_jobs

    job = scan_jobs.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if job["status"] == "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline failed: {job.get('error', 'Unknown error')}",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline still running. Status: {job['status']} ({job['progress']:.0%})",
        )

    report = job.get("result")
    if not report:
        raise HTTPException(status_code=404, detail="Report not found in completed job")

    # Build shortlist từ ranked_candidates
    # Lấy từ scan_jobs state nếu có, fallback về report.shortlist
    shortlist_items: list[ShortlistCandidate] = []

    # Cố gắng lấy ranked_candidates để có score details
    ranked = job.get("ranked_candidates", [])
    if ranked:
        for r in ranked[:10]:  # Top 10
            shortlist_items.append(ShortlistCandidate(
                rank           = r.rank,
                name           = r.score.candidate_name,
                total_score    = round(r.score.total_score, 1),
                tier           = r.score.tier,
                percentile     = r.percentile,
                recommendation = r.score.recommendation,
            ))
    else:
        # Fallback: chỉ có tên từ report.shortlist
        for i, name in enumerate(report.shortlist, 1):
            shortlist_items.append(ShortlistCandidate(
                rank=i, name=name,
                total_score=0.0, tier="unknown",
                percentile=0.0, recommendation=None,
            ))

    return ReportSummaryResponse(
        report_id        = report.meta.report_id,
        job_id           = report.meta.job_id,
        job_title        = report.meta.job_title,
        total_candidates = report.meta.total_candidates,
        shortlist_count  = report.meta.shortlist_count,
        summary_text     = report.summary_text,
        shortlist        = shortlist_items,
        has_pdf          = len(report.pdf_bytes) > 0,
        s3_url           = report.s3_url,
    )


@router.get("/run/{run_id}/download")
async def download_report_pdf(run_id: str):
    """
    Download PDF report binary.
    Browser sẽ hiện dialog save file.
    """
    from api.main import scan_jobs

    job = scan_jobs.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Report not ready yet")

    report = job.get("result")
    if not report or not report.pdf_bytes:
        raise HTTPException(status_code=404, detail="PDF not available")

    filename = f"cv_report_{report.meta.report_id}_{report.meta.job_id}.pdf"

    return Response(
        content      = report.pdf_bytes,
        media_type   = "application/pdf",
        headers      = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length":      str(len(report.pdf_bytes)),
        },
    )


@router.get("/run/{run_id}/shortlist")
async def get_shortlist(run_id: str):
    """
    Lấy shortlist nhanh — chỉ cần tên và score, không cần full report.
    Chainlit dùng để hiển thị quick summary ngay sau khi pipeline xong.
    """
    from api.main import scan_jobs

    job = scan_jobs.get(run_id)
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Report not ready")

    report  = job.get("result")
    ranked  = job.get("ranked_candidates", [])

    shortlist = []
    for r in ranked[:5]:   # Top 5 only
        shortlist.append({
            "rank":       r.rank,
            "name":       r.score.candidate_name,
            "score":      round(r.score.total_score, 1),
            "tier":       r.score.tier,
            "strengths":  r.score.strengths[:2],    # Top 2 strengths
            "concerns":   r.score.concerns[:1],     # Top 1 concern
        })

    return {
        "run_id":   run_id,
        "job_title": report.meta.job_title if report else "",
        "shortlist": shortlist,
    }