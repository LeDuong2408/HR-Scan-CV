"""
Pydantic models for Report Writer Agent output.

Produced by: Report Writer Agent (Agent 4)
Consumed by: FastAPI endpoint → Chainlit frontend
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ReportMeta(BaseModel):
    """Metadata của report — dùng để track và audit."""
    report_id:    str
    job_id:       str
    job_title:    str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    total_candidates: int = 0
    shortlist_count:  int = 0


class ReportOutput(BaseModel):
    """
    Output hoàn chỉnh của Report Writer Agent.

    Attributes:
        meta:          Thông tin về report
        pdf_bytes:     PDF binary — trả về trực tiếp cho frontend download
        s3_url:        Presigned URL nếu đã upload S3 (None nếu chạy local)
        summary_text:  Text summary ngắn để Chainlit hiển thị inline
        shortlist:     Top candidates được recommend phỏng vấn
    """
    meta:         ReportMeta
    pdf_bytes:    bytes = b""
    s3_url:       Optional[str] = None
    summary_text: str = ""
    shortlist:    list[str] = Field(default_factory=list)  # Tên top candidates