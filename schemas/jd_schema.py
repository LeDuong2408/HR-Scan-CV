"""
Schema: ParsedJD

Output của LLM khi parse JD text → structured fields.
Đây là intermediate schema, chỉ dùng nội bộ trong Agent 2.

Flow:
  JD raw text
      ↓ LLM (Step 1)
  ParsedJD          ← schema này
      ↓ ChromaDB query per skill
  CV evidence chunks
      ↓ LLM (Step 2)
  MatchResult       → Agent 3
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ParsedJD(BaseModel):
    """
    JD text được LLM phân tích thành các trường cụ thể.

    Mỗi item trong required_skills / nice_to_have sẽ được dùng
    làm query text để search ChromaDB cv_chunks.
    Giữ ngắn gọn (1 skill per item) để embedding chính xác nhất.
    """
    job_title:   str

    # Required — thiếu → loại
    required_skills: list[str] = Field(
        default_factory=list,
        description="Mỗi item là 1 skill/requirement ngắn gọn để query ChromaDB. "
                    "Ví dụ: ['Python backend 3+ years', 'FastAPI', 'AWS Lambda', 'PostgreSQL']",
    )

    # Nice to have — thiếu → trừ điểm nhẹ
    nice_to_have: list[str] = Field(default_factory=list)

    # Experience
    required_experience_years:  Optional[float] = None
    required_experience_domain: Optional[str]   = None  # e.g. "backend development"

    # Education
    required_education_level: Optional[str] = None   # "bachelor" | "master" | "phd" | "any"
    required_education_major: Optional[str] = None   # "Computer Science or related"

    # Context cho LLM step 2
    key_responsibilities: list[str] = Field(default_factory=list)
    seniority_level:      Optional[str] = None   # "junior" | "mid" | "senior" | "lead"