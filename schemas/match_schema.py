"""
Pydantic models for JD Matcher Agent output.

Consumed by: Scorer Agent (Agent 3)
Produced by: JD Matcher Agent (Agent 2)

Design note:
  MatchResult chứa đủ thông tin để Scorer Agent chấm điểm
  mà KHÔNG cần đọc lại CV hay JD gốc — self-contained.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MatchLevel(str, Enum):
    """Mức độ khớp của 1 requirement cụ thể."""
    FULL    = "full"    # CV có đủ, đúng domain, đúng level
    PARTIAL = "partial" # CV có nhưng thiếu depth hoặc khác domain
    MISSING = "missing" # CV hoàn toàn không có


class RequirementMatch(BaseModel):
    """
    Kết quả match của 1 JD requirement cụ thể với CV.

    Ví dụ:
      requirement: "3+ years Python backend development"
      candidate_evidence: "4 years Python at FPT Software (FastAPI, PostgreSQL)"
      match_level: FULL
      gap_note: None
    """
    requirement: str                       # Yêu cầu gốc từ JD
    candidate_evidence: Optional[str]      # Bằng chứng từ CV (None nếu missing)
    match_level: MatchLevel
    gap_note: Optional[str] = None         # Ghi chú nếu partial/missing


class SkillGapReport(BaseModel):
    """Phân loại toàn bộ skills."""
    matched:          list[str] = Field(default_factory=list)  # Có trong cả JD & CV
    missing_critical: list[str] = Field(default_factory=list)  # JD yêu cầu, CV thiếu
    missing_nice:     list[str] = Field(default_factory=list)  # Nice-to-have, CV thiếu
    bonus:            list[str] = Field(default_factory=list)  # CV có, JD không yêu cầu


class ExperienceAssessment(BaseModel):
    """Đánh giá phần kinh nghiệm làm việc."""
    required_years:     Optional[float] = None  # Số năm JD yêu cầu
    candidate_years:    Optional[float] = None  # Số năm thực tế của ứng viên
    meets_requirement:  bool = False
    domain_relevance:   float = 0.0             # 0.0 – 1.0: kinh nghiệm có đúng domain không
    relevance_note:     Optional[str] = None


class MatchResult(BaseModel):
    """
    Output hoàn chỉnh của JD Matcher Agent.
    Đây là input của Scorer Agent.
    """
    # Identity
    candidate_name:  str
    job_title:       str

    # Chi tiết match từng requirement
    requirement_matches: list[RequirementMatch] = Field(default_factory=list)

    # Tổng hợp skills
    skill_gap: SkillGapReport = Field(default_factory=SkillGapReport)

    # Đánh giá kinh nghiệm
    experience: ExperienceAssessment = Field(
        default_factory=ExperienceAssessment
    )

    # Điểm semantic similarity thô từ ChromaDB (trước khi LLM phân tích)
    # Dùng để debug / audit, không phải điểm cuối
    raw_similarity_score: float = 0.0

    # Summary cho HR đọc nhanh
    match_summary: Optional[str] = None

    # Meta
    jd_chunks_used:  list[str] = Field(default_factory=list)
    low_confidence:  bool = False
    warnings:        list[str] = Field(default_factory=list)