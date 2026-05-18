"""
Route: /api/v1/scan

Upload CVs và chạy pipeline:
  POST /start            — Upload files, start background pipeline
  GET  /status/{run_id}  — Poll status của 1 run
  GET  /stream/{run_id}  — SSE stream real-time progress (Chainlit dùng cái này)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILE_SIZE_MB   = 10


# ── Response models ────────────────────────────────────────────────────────

class ScanStartResponse(BaseModel):
    run_id:    str
    job_id:    str
    job_title: str
    file_count: int
    message:   str
    stream_url: str   # SSE URL để Chainlit connect


class ScanStatusResponse(BaseModel):
    run_id:   str
    status:   str
    progress: float
    step:     str
    error:    str | None = None
    report_id: str | None = None


# ── Background pipeline runner ─────────────────────────────────────────────

async def _run_pipeline_background(
    run_id:     str,
    file_paths: list[str],
    job_id:     str,
    job_title:  str,
    api_key:    str,
    tmpdir:     str,
) -> None:
    """
    Chạy LangGraph pipeline trong background task.
    Cập nhật scan_jobs[run_id] theo từng bước để SSE có thể stream.
    """
    from api.main import scan_jobs
    from graph.workflow import stream_pipeline

    # Lấy graph đã build từ app state
    import api.main as app_module
    graph = app_module.pipeline_graph

    try:
        scan_jobs[run_id]["status"]   = "running"
        scan_jobs[run_id]["progress"] = 0.05
        scan_jobs[run_id]["step"]     = f"Pipeline started for {len(file_paths)} CVs..."

        # Stream từng node event
        for event in stream_pipeline(
            graph      = graph,
            file_paths = file_paths,
            job_id     = job_id,
            job_title  = job_title,
            api_key    = api_key,
        ):
            state_update = event.get("state", {})

            # Cập nhật shared state từ mỗi node output
            if "progress"     in state_update:
                scan_jobs[run_id]["progress"] = state_update["progress"]
            if "current_step" in state_update:
                scan_jobs[run_id]["step"]     = state_update["current_step"]
            if "status"       in state_update:
                scan_jobs[run_id]["status"]   = state_update["status"]
            if "report"       in state_update and state_update["report"]:
                scan_jobs[run_id]["result"]   = state_update["report"]

            # Nhỏ async sleep để không block event loop
            await asyncio.sleep(0)

        # Kiểm tra kết quả cuối
        if scan_jobs[run_id]["status"] != "completed":
            scan_jobs[run_id]["status"] = "completed"
            scan_jobs[run_id]["progress"] = 1.0

        logger.info("Pipeline completed for run_id: %s", run_id)

    except Exception as e:
        logger.error("Pipeline failed for run_id %s: %s", run_id, e)
        scan_jobs[run_id]["status"] = "failed"
        scan_jobs[run_id]["error"]  = str(e)

    finally:
        # Xóa temp files
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.debug("Cleaned up tmpdir: %s", tmpdir)


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/start", response_model=ScanStartResponse, status_code=202)
async def start_scan(
    background_tasks: BackgroundTasks,
    files:     list[UploadFile] = File(..., description="CV files (PDF/DOCX)"),
    job_id:    str              = Form(...),
    job_title: str              = Form(...),
    api_key:   str              = Form(..., description="Gemini API key"),
):
    """
    Upload CVs và start pipeline trong background.

    Returns ngay lập tức với run_id.
    Client poll /status/{run_id} hoặc subscribe /stream/{run_id} để theo dõi.

    Form fields:
      - files:     List CV files (PDF/DOCX)
      - job_id:    ID của JD đã ingest vào ChromaDB
      - job_title: Tên vị trí
      - api_key:   Gemini API key
    """
    from api.main import scan_jobs

    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    validated: list[UploadFile] = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"File '{f.filename}' is not supported. Allowed: PDF, DOCX, DOC",
            )
        validated.append(f)

    # Lưu files vào tmpdir
    tmpdir     = tempfile.mkdtemp(prefix="hr_scan_")
    file_paths: list[str] = []

    for upload in validated:
        dest = os.path.join(tmpdir, upload.filename or f"cv_{uuid.uuid4()}.pdf")
        content = await upload.read()

        # Check file size
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename}' exceeds {MAX_FILE_SIZE_MB}MB limit",
            )

        with open(dest, "wb") as fh:
            fh.write(content)
        file_paths.append(dest)

    # Tạo run_id và khởi tạo job state
    run_id = str(uuid.uuid4())[:8].upper()
    scan_jobs[run_id] = {
        "status":    "pending",
        "progress":  0.0,
        "step":      "Queued...",
        "result":    None,
        "error":     None,
        "job_id":    job_id,
        "job_title": job_title,
        "file_count": len(file_paths),
    }

    # Start background pipeline
    background_tasks.add_task(
        _run_pipeline_background,
        run_id     = run_id,
        file_paths = file_paths,
        job_id     = job_id,
        job_title  = job_title,
        api_key    = api_key,
        tmpdir     = tmpdir,
    )

    logger.info(
        "Scan started: run_id=%s | %d files | job=%s",
        run_id, len(file_paths), job_id,
    )

    return ScanStartResponse(
        run_id     = run_id,
        job_id     = job_id,
        job_title  = job_title,
        file_count = len(file_paths),
        message    = f"Pipeline started for {len(file_paths)} CVs.",
        stream_url = f"/api/v1/scan/stream/{run_id}",
    )


@router.get("/status/{run_id}", response_model=ScanStatusResponse)
async def get_scan_status(run_id: str):
    """Poll status của 1 scan run. Dùng khi không cần SSE."""
    from api.main import scan_jobs

    job = scan_jobs.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    report_id = None
    if job["result"]:
        report_id = job["result"].meta.report_id

    return ScanStatusResponse(
        run_id    = run_id,
        status    = job["status"],
        progress  = job["progress"],
        step      = job["step"],
        error     = job["error"],
        report_id = report_id,
    )


@router.get("/stream/{run_id}")
async def stream_scan_progress(run_id: str):
    """
    SSE endpoint — stream real-time progress về Chainlit.

    Chainlit subscribe vào đây và hiển thị từng bước.
    Tự động kết thúc khi status = 'completed' hoặc 'failed'.

    Format SSE event:
      data: {"status": "running", "progress": 0.55, "step": "Scoring candidates..."}
    """
    from api.main import scan_jobs

    if run_id not in scan_jobs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        last_step = ""
        while True:
            job = scan_jobs.get(run_id)
            if not job:
                break

            current_step = job["step"]

            # Chỉ emit khi có thay đổi (tránh spam)
            if current_step != last_step:
                payload = {
                    "run_id":   run_id,
                    "status":   job["status"],
                    "progress": round(job["progress"], 3),
                    "step":     current_step,
                    "error":    job["error"],
                }
                # SSE format: "data: {json}\n\n"
                yield f"data: {json.dumps(payload)}\n\n"
                last_step = current_step

            # Pipeline kết thúc → gửi final event và dừng
            if job["status"] in {"completed", "failed"}:
                final_payload = {
                    "run_id":    run_id,
                    "status":    job["status"],
                    "progress":  1.0,
                    "step":      job["step"],
                    "error":     job["error"],
                    "report_id": (
                        job["result"].meta.report_id
                        if job.get("result") else None
                    ),
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                break

            await asyncio.sleep(0.5)  # Poll interval

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )