from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter(prefix="", tags=["upload"])


@router.post("/upload-cvs")
async def upload_cvs(jd_id: str = Form(...), files: list[UploadFile] = File(...)) -> dict:
    accepted = [f.filename for f in files]
    return {
        "job_id": str(uuid4()),
        "jd_id": jd_id,
        "received_files": accepted,
        "status": "queued",
    }

