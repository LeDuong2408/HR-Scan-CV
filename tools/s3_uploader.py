"""
Tool: AWS S3 Uploader

Upload PDF report lên S3 và trả về presigned URL.

Free tier: 5GB storage, 20,000 GET, 2,000 PUT requests/tháng.
Demo với < 100 CVs/ngày sẽ không bao giờ vượt free tier.

Fallback: Nếu không có AWS credentials → lưu local, không crash.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

S3_BUCKET        = os.getenv("S3_BUCKET_NAME",   "hr-cv-scanner-reports")
S3_REGION        = os.getenv("AWS_REGION",        "ap-southeast-1")
PRESIGNED_EXPIRY = int(os.getenv("S3_URL_EXPIRY", "604800"))  # 7 ngày


def upload_report(
    pdf_bytes: bytes,
    job_id:    str,
    report_id: str,
) -> str | None:
    """
    Upload PDF lên S3, trả về presigned URL.

    Args:
        pdf_bytes: PDF binary từ pdf_generator
        job_id:    Dùng để tổ chức S3 key (folder structure)
        report_id: Unique ID của report

    Returns:
        Presigned URL (string) nếu thành công, None nếu không có AWS config
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        logger.warning("boto3 not installed — skipping S3 upload")
        return None

    # Nếu không có credentials → fallback gracefully
    try:
        s3 = boto3.client("s3", region_name=S3_REGION)
        s3_key = _build_s3_key(job_id, report_id)

        s3.put_object(
            Bucket      = S3_BUCKET,
            Key         = s3_key,
            Body        = pdf_bytes,
            ContentType = "application/pdf",
            Metadata    = {
                "job_id":    job_id,
                "report_id": report_id,
                "generated": datetime.utcnow().isoformat(),
            },
        )

        url = s3.generate_presigned_url(
            "get_object",
            Params     = {"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn  = PRESIGNED_EXPIRY,
        )
        logger.info("Uploaded report to S3: s3://%s/%s", S3_BUCKET, s3_key)
        return url

    except Exception as e:
        logger.warning("S3 upload failed: %s — report available in-memory only", e)
        return None


def save_local(
    pdf_bytes:  bytes,
    job_id:     str,
    report_id:  str,
    output_dir: str = "reports",
) -> str:
    """
    Fallback: Lưu PDF xuống local disk.
    Dùng khi dev local hoặc không có S3.

    Returns:
        Absolute path của file đã lưu
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = f"{job_id}_{report_id}.pdf"
    filepath = out / filename
    filepath.write_bytes(pdf_bytes)

    logger.info("Report saved locally: %s", filepath.resolve())
    return str(filepath.resolve())


def _build_s3_key(job_id: str, report_id: str) -> str:
    """
    S3 key có cấu trúc rõ ràng để dễ quản lý.
    Format: reports/{year}/{month}/{job_id}/{report_id}.pdf
    """
    now = datetime.utcnow()
    return f"reports/{now.year}/{now.month:02d}/{job_id}/{report_id}.pdf"