"""
RAG Layer 3: Retriever

Nhiệm vụ: Tìm kiếm các JD requirements liên quan nhất
với CV của ứng viên bằng semantic similarity.

Tại sao semantic search thay vì keyword search?
  Keyword: "FastAPI" không match "REST API framework"
  Semantic: Hiểu cả 2 đều nói về cùng 1 khái niệm → match ✅

Flow:
  CV skills text → embed → vector
  → ChromaDB cosine similarity search
  → Top-K JD requirement chunks liên quan nhất
  → Trả về cả text + score + metadata
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings

from rag.embedder import embed_text
from rag.ingestor import (
    CHROMA_DIR,
    JD_COLLECTION,
    RUBRIC_COLLECTION,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    1 JD requirement chunk được retrieve về.

    Attributes:
        chunk_id:   ID trong ChromaDB (ví dụ: "backend-eng-2025-01_req_003")
        text:       Nội dung requirement ("3+ years Python backend")
        score:      Cosine similarity score (0.0 – 1.0, cao hơn = liên quan hơn)
        priority:   "required" hoặc "nice_to_have"
        job_id:     ID của JD chứa chunk này
        job_title:  Tên vị trí
    """
    chunk_id:  str
    text:      str
    score:     float
    priority:  str
    job_id:    str
    job_title: str


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def search_jd_requirements(
    query_text: str,
    job_id:     str,
    top_k:      int = 10,
    min_score:  float = 0.3,
) -> list[RetrievedChunk]:
    """
    Tìm top-K JD requirements liên quan nhất với query_text.

    Args:
        query_text: Text từ CV (thường là skills + work history tóm tắt)
        job_id:     Chỉ search trong JD của vị trí này
        top_k:      Số lượng results tối đa
        min_score:  Lọc bỏ results có similarity quá thấp (noise)

    Returns:
        List RetrievedChunk, sắp xếp theo score giảm dần
    """
    if not query_text.strip():
        raise ValueError("query_text cannot be empty")

    client     = _get_client()
    collection = client.get_or_create_collection(JD_COLLECTION)

    # Kiểm tra collection có data không
    if collection.count() == 0:
        logger.warning("JD collection is empty. Did you run ingest_job_description()?")
        return []

    # Embed query text
    query_vector = embed_text(query_text)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        where={"job_id": {"$eq": job_id}},          # Filter theo job_id
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []

    if not results["ids"] or not results["ids"][0]:
        return chunks

    for chunk_id, doc, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB trả về distance (0 = identical, 2 = opposite)
        # Chuyển sang similarity score: 1 - distance/2
        score = round(1.0 - distance / 2.0, 4)

        if score < min_score:
            continue  # Lọc bỏ chunk không liên quan

        chunks.append(RetrievedChunk(
            chunk_id=chunk_id,
            text=doc,
            score=score,
            priority=meta.get("priority", "required"),
            job_id=meta.get("job_id", ""),
            job_title=meta.get("job_title", ""),
        ))

    # Sắp xếp theo score giảm dần
    chunks.sort(key=lambda c: c.score, reverse=True)

    logger.debug(
        "Retrieved %d/%d chunks for job_id=%s (min_score=%.2f)",
        len(chunks), top_k, job_id, min_score,
    )
    return chunks


def get_rubric(job_id: str) -> Optional[dict]:
    """
    Lấy scoring rubric cho 1 job_id.

    Returns:
        Dict rubric, hoặc None nếu chưa ingest
    """
    client     = _get_client()
    collection = client.get_or_create_collection(RUBRIC_COLLECTION)

    results = collection.get(
        ids=[f"rubric_{job_id}"],
        include=["documents"],
    )

    if not results["documents"]:
        logger.warning("No rubric found for job_id: %s", job_id)
        return None

    try:
        return json.loads(results["documents"][0])
    except json.JSONDecodeError as e:
        logger.error("Failed to parse rubric JSON for job_id %s: %s", job_id, e)
        return None


def search_similar(
    query_text: str,
    top_k:      int = 5,
) -> list[RetrievedChunk]:
    """
    Search across ALL jobs (không filter job_id).
    Dùng cho exploratory search hoặc testing.
    """
    client     = _get_client()
    collection = client.get_or_create_collection(JD_COLLECTION)

    if collection.count() == 0:
        return []

    query_vector = embed_text(query_text)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for chunk_id, doc, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1.0 - distance / 2.0, 4)
        chunks.append(RetrievedChunk(
            chunk_id=chunk_id,
            text=doc,
            score=score,
            priority=meta.get("priority", "required"),
            job_id=meta.get("job_id", ""),
            job_title=meta.get("job_title", ""),
        ))

    return sorted(chunks, key=lambda c: c.score, reverse=True)