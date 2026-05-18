"""
FastAPI Backend — HR CV Scanner

Entry point của toàn bộ backend. Chạy lệnh:
  uvicorn api.main:app --reload --port 8000

Kiến trúc:
  /api/v1/jobs     — Quản lý Job Description (ingest vào ChromaDB)
  /api/v1/scan     — Upload CVs và chạy pipeline
  /api/v1/reports  — Lấy report đã tạo

Design:
  - build_graph() được gọi 1 lần khi startup → reuse cho mọi request
  - Mỗi scan request chạy trong background task → không block HTTP response
  - SSE endpoint stream real-time progress về Chainlit frontend
  - Job results lưu in-memory (dict) — đủ cho demo, production dùng DynamoDB
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.jobs import router as jobs_router
from api.routes.reports import router as reports_router
from api.routes.scan import router as scan_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── In-memory store ────────────────────────────────────────────────────────
# Shared state giữa các routes:
#   scan_jobs[job_run_id] = {
#       "status":   "pending" | "running" | "completed" | "failed",
#       "progress": 0.0 → 1.0,
#       "step":     "current step message",
#       "result":   ReportOutput | None,
#       "error":    str | None,
#   }
scan_jobs: dict[str, Any] = {}

# Compiled LangGraph — build once, reuse everywhere
pipeline_graph: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build LangGraph graph on startup."""
    global pipeline_graph
    logger.info("Building LangGraph pipeline...")
    from graph.workflow import build_graph
    pipeline_graph = build_graph()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "HR CV Scanner API",
    description = "Multi-agent CV screening system",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allow Chainlit frontend (port 8001) to call this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8001", "http://127.0.0.1:8001"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────
app.include_router(jobs_router,    prefix="/api/v1/jobs",    tags=["Jobs"])
app.include_router(scan_router,    prefix="/api/v1/scan",    tags=["Scan"])
app.include_router(reports_router, prefix="/api/v1/reports", tags=["Reports"])


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "pipeline_ready": pipeline_graph is not None,
        "active_scans":  sum(
            1 for j in scan_jobs.values()
            if j["status"] == "running"
        ),
    }