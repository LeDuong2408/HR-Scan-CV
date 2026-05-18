"""
Pydantic models for Scorer Agent output.

Consumed by: Report Writer Agent (Agent 4)
Produced by: Scorer Agent (Agent 3)

Design note:
  - DimensionScore: điểm từng tiêu chí với lý do rõ ràng
  - ScoreBreakdown: tổng hợp 5 dimensions
  - CandidateScore: output cuối cùng, đủ để Report Writer tạo PDF
  - RankedCandidate: sau khi sort toàn bộ batch
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# Rubric — đọc từ ChromaDB, define cách chấm điểm
# ──────────────────────────────────────────────────────────────────────────────

class DimensionConfig(BaseModel):
    """Config của 1 tiêu chí chấm điểm trong rubric."""
    weight:      int            # Tổng điểm tối đa của dimension này
    description: str            # Mô tả tiêu chí (đưa vào LLM prompt)
    scored_by:   str = "llm"    # "llm" | "programmatic" | "hybrid"


class ScoringRubric(BaseModel):
    """
    Rubric chấm điểm — lưu trong ChromaDB, load mỗi lần Scorer chạy.
    Tổng weight phải = 100.

    Mặc định:
      technical_skills:  35 — khớp kỹ năng kỹ thuật với JD
      experience:        20 — số năm và domain relevance
      education:         15 — bằng cấp và chuyên ngành
      achievements:      20 — chất lượng thành tựu trong công việc
      soft_skills:       10 — ngôn ngữ, soft skills, certifications
    """
    job_id:     str
    job_title:  str
    dimensions: dict[str, DimensionConfig]

    @field_validator("dimensions")
    @classmethod
    def validate_total_weight(
        cls, dims: dict[str, DimensionConfig]
    ) -> dict[str, DimensionConfig]:
        total = sum(d.weight for d in dims.values())
        if total != 100:
            raise ValueError(
                f"Rubric weights must sum to 100, got {total}. "
                f"Dimensions: { {k: d.weight for k, d in dims.items()} }"
            )
        return dims


# ──────────────────────────────────────────────────────────────────────────────
# Score output
# ──────────────────────────────────────────────────────────────────────────────

class DimensionScore(BaseModel):
    """Điểm của 1 tiêu chí cụ thể."""
    dimension:   str            # Tên dimension (ví dụ: "technical_skills")
    max_score:   int            # Tối đa (= weight trong rubric)
    raw_score:   float          # Điểm thực tế (0 → max_score)
    percentage:  float          # raw_score / max_score × 100
    rationale:   str            # Giải thích tại sao cho điểm này (LLM viết)
    scored_by:   str            # "llm" | "programmatic"

    @field_validator("raw_score")
    @classmethod
    def score_not_exceed_max(cls, v: float, info) -> float:
        # Pydantic v2: info.data chứa các fields đã validate trước đó
        max_s = info.data.get("max_score", 100)
        if v > max_s:
            return float(max_s)
        if v < 0:
            return 0.0
        return round(v, 2)

    @field_validator("percentage")
    @classmethod
    def clamp_percentage(cls, v: float) -> float:
        return round(max(0.0, min(100.0, v)), 2)


class ScoreBreakdown(BaseModel):
    """Điểm chi tiết theo từng dimension."""
    technical_skills: Optional[DimensionScore] = None
    experience:       Optional[DimensionScore] = None
    education:        Optional[DimensionScore] = None
    achievements:     Optional[DimensionScore] = None
    soft_skills:      Optional[DimensionScore] = None

    def as_list(self) -> list[DimensionScore]:
        """Trả về tất cả dimensions đã được score (bỏ None)."""
        return [
            d for d in [
                self.technical_skills,
                self.experience,
                self.education,
                self.achievements,
                self.soft_skills,
            ]
            if d is not None
        ]


class ScoreTier(str):
    """
    Tier dựa trên tổng điểm:
      STRONG:   >= 80  → Mạnh, ưu tiên phỏng vấn
      GOOD:     60-79  → Tốt, cân nhắc phỏng vấn
      MODERATE: 40-59  → Trung bình, cần xem xét kỹ
      WEAK:     < 40   → Yếu, không phù hợp
    """
    STRONG   = "strong"
    GOOD     = "good"
    MODERATE = "moderate"
    WEAK     = "weak"

    @classmethod
    def from_score(cls, total: float) -> str:
        if total >= 80:  return cls.STRONG
        if total >= 60:  return cls.GOOD
        if total >= 40:  return cls.MODERATE
        return cls.WEAK


class CandidateScore(BaseModel):
    """
    Output hoàn chỉnh của Scorer Agent cho 1 ứng viên.
    Đây là input của Report Writer Agent.
    """
    # Identity — lấy từ MatchResult
    candidate_name:  str
    job_title:       str

    # Tổng điểm
    total_score:     float = Field(ge=0.0, le=100.0)
    tier:            str   = ""             # strong / good / moderate / weak

    # Chi tiết từng dimension
    breakdown:       ScoreBreakdown = Field(default_factory=ScoreBreakdown)

    # Highlights cho Report Writer
    strengths:       list[str] = Field(default_factory=list)   # Top 3 điểm mạnh
    concerns:        list[str] = Field(default_factory=list)   # Top 3 điểm cần lưu ý
    recommendation:  Optional[str] = None                      # Khuyến nghị của Scorer

    # Meta
    low_confidence:  bool       = False
    warnings:        list[str]  = Field(default_factory=list)
    rubric_used:     str        = ""        # job_id của rubric đã dùng

    def model_post_init(self, __context) -> None:
        """Tự động tính tier sau khi validate."""
        if not self.tier:
            self.tier = ScoreTier.from_score(self.total_score)


class RankedCandidate(BaseModel):
    """
    Ứng viên sau khi được xếp hạng trong batch.
    Report Writer dùng list[RankedCandidate] để tạo PDF.
    """
    rank:            int           # 1 = tốt nhất
    percentile:      float         # Top X% của batch (100 = tốt nhất)
    score:           CandidateScore

    @field_validator("percentile")
    @classmethod
    def clamp_percentile(cls, v: float) -> float:
        return round(max(0.0, min(100.0, v)), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Default rubric — dùng khi không có rubric trong ChromaDB
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_RUBRIC = ScoringRubric(
    job_id="default",
    job_title="General Software Engineer",
    dimensions={
        "technical_skills": DimensionConfig(
            weight=35,
            description=(
                "Evaluate how well the candidate's technical skills match "
                "the job requirements. Consider both breadth and depth. "
                "Full match = 35, partial = 15–25, missing critical = < 15."
            ),
            scored_by="hybrid",
        ),
        "experience": DimensionConfig(
            weight=20,
            description=(
                "Evaluate years of experience AND domain relevance. "
                "Meets/exceeds required years in relevant domain = 20. "
                "Meets years but different domain = 10–15. "
                "Below required years = 5–10."
            ),
            scored_by="programmatic",
        ),
        "education": DimensionConfig(
            weight=15,
            description=(
                "Evaluate highest education level and major relevance. "
                "Relevant CS/Engineering degree = 12–15. "
                "Unrelated degree = 6–9. No degree = 0–5."
            ),
            scored_by="llm",
        ),
        "achievements": DimensionConfig(
            weight=20,
            description=(
                "Evaluate quality and impact of achievements in work history. "
                "Look for: quantified results, business impact, technical depth. "
                "Strong quantified achievements = 16–20. "
                "Vague achievements = 8–14. No achievements listed = 0–7."
            ),
            scored_by="llm",
        ),
        "soft_skills": DimensionConfig(
            weight=10,
            description=(
                "Evaluate languages, certifications, and soft skills. "
                "Strong English + relevant certs = 8–10. "
                "Average English + no certs = 4–6."
            ),
            scored_by="llm",
        ),
    },
)