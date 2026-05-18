from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("cv-scanner-api")


def configure_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[override]
        started = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - started) * 1000
        logger.info("%s %s -> %s (%.2fms)", request.method, request.url.path, response.status_code, duration)
        return response

