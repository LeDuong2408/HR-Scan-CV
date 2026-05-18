"""
Route: /api/v1/jobs

Quản lý Job Descriptions:
  POST /               — Tạo JD mới, ingest vào ChromaDB
  GET  /               — List tất cả JDs đã ingest
  DELETE /{job_id}     — Xóa JD (khi update)
  POST /{job_id}/rubric — Upload scoring rubric cho 1 JD
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag.ingestor import (
    clear_job,
    ingest_job_description,
    ingest_rubric,
    list_jobs,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    job_id:       str = Field(..., description="Unique ID, e.g. 'backend-eng-2025-01'")
    job_title:    str = Field(..., description="e.g. 'Senior Backend Engineer'")
    requirements: list[str] = Field(
        ...,
        description="List of required skills/experience, each as 1 sentence",
        min_length=1,
    )
    nice_to_have: list[str] = Field(default_factory=list)
    metadata:     dict      = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id":    "backend-eng-2025-01",
                "job_title": "Senior Backend Engineer",
                "requirements": [
                    "3+ years Python backend development experience",
                    "Experience with REST API design using FastAPI or Django",
                    "AWS Lambda and S3 hands-on experience",
                    "Strong understanding of PostgreSQL and Redis",
                    "Experience with Docker and CI/CD pipelines",
                ],
                "nice_to_have": [
                    "Kubernetes and container orchestration",
                    "Terraform or infrastructure as code",
                ],
                "metadata": {"department": "Engineering", "level": "Senior"},
            }
        }
    }


class CreateJobResponse(BaseModel):
    job_id:       str
    job_title:    str
    chunks_count: int
    message:      str


class RubricDimensionRequest(BaseModel):
    weight:      int
    description: str
    scored_by:   str = "llm"


class CreateRubricRequest(BaseModel):
    job_id:     str
    job_title:  str
    dimensions: dict[str, RubricDimensionRequest]

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id":    "backend-eng-2025-01",
                "job_title": "Senior Backend Engineer",
                "dimensions": {
                    "technical_skills": {"weight": 35, "description": "Match technical skills vs JD", "scored_by": "programmatic"},
                    "experience":       {"weight": 20, "description": "Years and domain relevance",   "scored_by": "programmatic"},
                    "education":        {"weight": 15, "description": "Degree level and relevance",   "scored_by": "llm"},
                    "achievements":     {"weight": 20, "description": "Quality of achievements",      "scored_by": "llm"},
                    "soft_skills":      {"weight": 10, "description": "Languages and soft skills",    "scored_by": "llm"},
                },
            }
        }
    }


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/", response_model=CreateJobResponse, status_code=201)
async def create_job(req: CreateJobRequest):
    """
    Ingest Job Description vào ChromaDB.
    Mỗi requirement được embed và lưu như 1 chunk riêng.
    """
    try:
        count = ingest_job_description(
            job_id       = req.job_id,
            job_title    = req.job_title,
            requirements = req.requirements,
            nice_to_have = req.nice_to_have,
            metadata     = req.metadata,
        )
        logger.info("Created job: %s (%d chunks)", req.job_id, count)
        return CreateJobResponse(
            job_id       = req.job_id,
            job_title    = req.job_title,
            chunks_count = count,
            message      = f"Job '{req.job_title}' ingested with {count} requirement chunks.",
        )
    except Exception as e:
        logger.error("Failed to create job %s: %s", req.job_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_all_jobs():
    """List tất cả JDs đã được ingest vào ChromaDB."""
    try:
        jobs = list_jobs()
        return {"jobs": jobs, "total": len(jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """Xóa toàn bộ chunks của 1 JD (dùng khi update requirements)."""
    try:
        clear_job(job_id)
        logger.info("Deleted job: %s", job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/rubric", status_code=201)
async def create_rubric(job_id: str, req: CreateRubricRequest):
    """
    Upload scoring rubric cho 1 JD.
    Rubric được lưu vào ChromaDB và load mỗi lần Scorer Agent chạy.
    Tổng weight của tất cả dimensions phải = 100.
    """
    total_weight = sum(d.weight for d in req.dimensions.values())
    if total_weight != 100:
        raise HTTPException(
            status_code=422,
            detail=f"Dimension weights must sum to 100, got {total_weight}",
        )

    try:
        rubric_dict = {
            "job_id":     req.job_id,
            "job_title":  req.job_title,
            "dimensions": {
                k: {"weight": v.weight, "description": v.description, "scored_by": v.scored_by}
                for k, v in req.dimensions.items()
            },
        }
        ingest_rubric(job_id=job_id, rubric=rubric_dict)
        return {"message": f"Rubric for job '{job_id}' saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))