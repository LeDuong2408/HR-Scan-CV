"""
LangSmith Tracing — HR CV Scanner

Setup LangSmith để trace toàn bộ multi-agent pipeline.

Cách hoạt động:
  - Mỗi LangGraph node run = 1 "run" trong LangSmith
  - Mỗi LLM call bên trong agent = 1 "child run"
  - Toàn bộ pipeline (parse → match → score → report) = 1 "trace"

Để xem traces: https://smith.langchain.com

Lấy API key: https://smith.langchain.com/settings
"""
from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


def setup_langsmith(
    project_name: str = "hr-cv-scanner",
    enabled:      bool | None = None,
) -> bool:
    """
    Kích hoạt LangSmith tracing bằng environment variables.

    Gọi hàm này 1 lần khi app khởi động (trong api/main.py lifespan).

    Args:
        project_name: Tên project trên LangSmith dashboard
        enabled:      None = tự detect từ env var LANGSMITH_API_KEY

    Returns:
        True nếu tracing được bật, False nếu không có API key
    """
    api_key = os.getenv("LANGSMITH_API_KEY", "")

    if enabled is False or not api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing: DISABLED (no API key)")
        return False

    # Set env vars mà LangGraph/LangChain tự đọc
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"]    = api_key
    os.environ["LANGCHAIN_PROJECT"]    = project_name

    logger.info(
        "LangSmith tracing: ENABLED | project='%s' | dashboard: https://smith.langchain.com",
        project_name,
    )
    return True


def trace_agent(
    name:        str,
    run_type:    str = "chain",
    tags:        list[str] | None = None,
    metadata:    dict | None = None,
):
    """
    Decorator để wrap 1 function thành 1 LangSmith span (child run).

    Dùng cho các method quan trọng trong agents mà muốn
    thấy riêng trong dashboard (không chỉ LLM call).

    Usage:
        @trace_agent(name="cv_parser.extract_text", tags=["parsing"])
        def _extract_text(self, path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.getenv("LANGCHAIN_TRACING_V2") != "true":
                return func(*args, **kwargs)

            try:
                from langsmith import traceable
                traced = traceable(
                    name     = name,
                    run_type = run_type,
                    tags     = tags or [],
                    metadata = metadata or {},
                )(func)
                return traced(*args, **kwargs)
            except Exception:
                # Nếu LangSmith có lỗi → vẫn chạy bình thường, không crash
                return func(*args, **kwargs)

        return wrapper
    return decorator


def get_run_url(run_id: str) -> str | None:
    """
    Lấy URL của 1 run trên LangSmith dashboard.
    Dùng để hiển thị link trong Streamlit sau khi pipeline xong.
    """
    project = os.getenv("LANGCHAIN_PROJECT", "hr-cv-scanner")
    if not os.getenv("LANGCHAIN_API_KEY"):
        return None
    return f"https://smith.langchain.com/o/default/projects/{project}/r/{run_id}"


def log_pipeline_start(
    run_id:     str,
    job_id:     str,
    job_title:  str,
    file_count: int,
) -> dict:
    """
    Log metadata khi pipeline bắt đầu.
    Return dict để lưu vào scan_jobs[run_id] cho Streamlit hiển thị.
    """
    metadata = {
        "run_id":     run_id,
        "job_id":     job_id,
        "job_title":  job_title,
        "file_count": file_count,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "hr-cv-scanner")
        logger.info(
            "LangSmith trace started | project=%s | run_id=%s | "
            "View at: https://smith.langchain.com",
            project, run_id,
        )

    return metadata