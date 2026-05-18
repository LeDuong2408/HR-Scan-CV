"""
RAG Layer 2: Document Ingestor

Nhiệm vụ: Load Job Description + Scoring Rubric vào ChromaDB
để JD Matcher Agent có thể retrieve sau này.

Khi nào chạy ingestor?
  - Lần đầu setup hệ thống
  - Khi HR tạo JD mới cho 1 vị trí mới
  - Khi cập nhật rubric chấm điểm

Chunking strategy:
  JD được chia thành chunks nhỏ theo từng requirement
  (không phải chia theo số ký tự) vì mỗi requirement
  là 1 đơn vị ngữ nghĩa độc lập.

  Ví dụ JD: "3+ years Python. AWS required. Strong communication."
  → Chunk 1: "3+ years Python backend development"
  → Chunk 2: "AWS Lambda, S3, ECS experience required"
  → Chunk 3: "Strong communication and teamwork skills"
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from rag.embedder import embed_batch

logger = logging.getLogger(__name__)

# ChromaDB persist directory — lưu local, không mất khi restart
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"

# Collection names
JD_COLLECTION      = "job_descriptions"
RUBRIC_COLLECTION  = "scoring_rubrics"


def _get_client() -> chromadb.PersistentClient:
    """Trả về ChromaDB client với persistent storage."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def ingest_job_description(
    job_id:        str,
    job_title:     str,
    requirements:  list[str],
    nice_to_have:  list[str]  = None,
    metadata:      dict       = None,
) -> int:
    """
    Ingest 1 Job Description vào ChromaDB.

    Mỗi requirement được lưu như 1 chunk riêng biệt.
    Metadata giúp filter theo job_id khi retrieve.

    Args:
        job_id:       ID duy nhất cho JD này (ví dụ: "backend-eng-2025-01")
        job_title:    Tên vị trí ("Senior Backend Engineer")
        requirements: List các yêu cầu bắt buộc
        nice_to_have: List các yêu cầu tốt nếu có (optional)
        metadata:     Thông tin thêm (department, level, v.v.)

    Returns:
        Số chunks đã được ingest
    """
    client     = _get_client()
    collection = client.get_or_create_collection(
        name=JD_COLLECTION,
        metadata={"hnsw:space": "cosine"},  # Dùng cosine similarity
    )

    chunks:    list[str]        = []
    chunk_ids: list[str]        = []
    metadatas: list[dict]       = []

    base_meta = {
        "job_id":    job_id,
        "job_title": job_title,
        "type":      "requirement",
        **(metadata or {}),
    }

    # Ingest requirements bắt buộc
    for i, req in enumerate(requirements):
        chunk_id = f"{job_id}_req_{i:03d}"
        chunks.append(req)
        chunk_ids.append(chunk_id)
        metadatas.append({**base_meta, "priority": "required", "index": i})

    # Ingest nice-to-have (optional)
    for i, req in enumerate(nice_to_have or []):
        chunk_id = f"{job_id}_nice_{i:03d}"
        chunks.append(req)
        chunk_ids.append(chunk_id)
        metadatas.append({**base_meta, "priority": "nice_to_have", "index": i})

    # Embed tất cả chunks cùng lúc (batch)
    vectors = embed_batch(chunks)

    # Upsert vào ChromaDB (update nếu đã tồn tại)
    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=vectors,
        metadatas=metadatas,
    )

    logger.info(
        "Ingested JD '%s' (%s): %d required + %d nice-to-have chunks",
        job_title, job_id, len(requirements), len(nice_to_have or []),
    )
    return len(chunks)


def ingest_rubric(
    job_id:     str,
    rubric:     dict[str, Any],
) -> None:
    """
    Lưu scoring rubric vào ChromaDB dưới dạng JSON document.

    Rubric không được chunk — lưu nguyên 1 document duy nhất
    vì Scorer Agent cần đọc toàn bộ rubric cùng lúc.

    Args:
        job_id: Phải match với job_id đã dùng trong ingest_job_description
        rubric: Dict mô tả cách chấm điểm (xem ví dụ bên dưới)

    Ví dụ rubric:
    {
        "dimensions": {
            "technical_skills": {"weight": 35, "description": "..."},
            "experience":       {"weight": 20, "description": "..."},
            "education":        {"weight": 15, "description": "..."},
            "achievements":     {"weight": 20, "description": "..."},
            "soft_skills":      {"weight": 10, "description": "..."},
        }
    }
    """
    client     = _get_client()
    collection = client.get_or_create_collection(
        name=RUBRIC_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    rubric_text = json.dumps(rubric, ensure_ascii=False)
    # Embed rubric text để có thể semantic search nếu cần
    vector = embed_batch([rubric_text])[0]

    collection.upsert(
        ids=[f"rubric_{job_id}"],
        documents=[rubric_text],
        embeddings=[vector],
        metadatas=[{"job_id": job_id}],
    )
    logger.info("Ingested rubric for job_id: %s", job_id)


def list_jobs() -> list[dict]:
    """Liệt kê tất cả job_id đã được ingest."""
    client     = _get_client()
    collection = client.get_or_create_collection(JD_COLLECTION)
    results    = collection.get()

    seen, jobs = set(), []
    for meta in results.get("metadatas") or []:
        jid = meta.get("job_id")
        if jid and jid not in seen:
            seen.add(jid)
            jobs.append({"job_id": jid, "job_title": meta.get("job_title", "")})
    return jobs


def clear_job(job_id: str) -> None:
    """Xóa toàn bộ chunks của 1 job_id (dùng khi update JD)."""
    client     = _get_client()
    collection = client.get_or_create_collection(JD_COLLECTION)
    results    = collection.get(where={"job_id": job_id})
    ids        = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        logger.info("Cleared %d chunks for job_id: %s", len(ids), job_id)