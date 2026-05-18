"""
LangGraph State Schema

State là "huyết mạch" của toàn bộ graph — mọi node đều đọc và ghi vào đây.
Mỗi field được annotated với reducer để LangGraph biết cách merge
khi nhiều node cập nhật song song.

Flow của state:
  ┌─────────────────────────────────────────────────────────┐
  │ INPUT                                                    │
  │   file_paths, job_id, job_title                         │
  └────────────────────┬────────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────────┐
  │ AFTER PARSE NODE                                         │
  │   + parsed_candidates: list[CandidateProfile]           │
  │   + parse_errors: list[str]                             │
  └────────────────────┬────────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────────┐
  │ AFTER MATCH NODE                                         │
  │   + match_results: list[MatchResult]                    │
  └────────────────────┬────────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────────┐
  │ AFTER SCORE NODE                                         │
  │   + ranked_candidates: list[RankedCandidate]            │
  └────────────────────┬────────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────────┐
  │ AFTER REPORT NODE                                        │
  │   + report: ReportOutput                                │
  │   + status: "completed"                                 │
  └─────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Annotated, Any, Optional
import operator

from pydantic import BaseModel, Field

from schemas.cv_schema import CandidateProfile
from schemas.match_schema import MatchResult
from schemas.report_schema import ReportOutput
from schemas.score_schema import RankedCandidate


class PipelineStatus:
    PENDING   = "pending"
    PARSING   = "parsing"
    MATCHING  = "matching"
    SCORING   = "scoring"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED    = "failed"


class GraphState(BaseModel):
    """
    State duy nhất chạy xuyên suốt toàn bộ LangGraph pipeline.

    Annotated[list, operator.add] = reducer:
      Khi nhiều node cùng cập nhật 1 field,
      LangGraph tự động merge bằng cách nối list lại.
      → An toàn cho parallel processing.
    """

    # ── INPUT (set trước khi chạy graph) ─────────────────────────────────────
    file_paths:  list[str] = Field(default_factory=list)
    job_id:      str       = ""
    job_title:   str       = ""
    api_key:     str       = ""  # Gemini API key

    # ── PIPELINE PROGRESS ────────────────────────────────────────────────────
    status:      str  = PipelineStatus.PENDING
    current_step: str = ""        # Log message cho Chainlit real-time
    progress:    float = 0.0      # 0.0 → 1.0

    # ── NODE OUTPUTS ─────────────────────────────────────────────────────────
    # Annotated với operator.add để list được append thay vì replace
    parsed_candidates: Annotated[list[CandidateProfile], operator.add] = Field(
        default_factory=list
    )
    parse_errors: Annotated[list[str], operator.add] = Field(
        default_factory=list
    )
    match_results: Annotated[list[MatchResult], operator.add] = Field(
        default_factory=list
    )
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    report: Optional[ReportOutput] = None

    # ── ERROR TRACKING ───────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    retry_count: int = 0

    class Config:
        arbitrary_types_allowed = True