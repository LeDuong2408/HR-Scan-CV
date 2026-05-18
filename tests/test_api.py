"""
Tests for FastAPI Backend routes.

Run: pytest tests/test_api.py -v

Dùng httpx.AsyncClient với app trực tiếp — không cần server đang chạy.
3 tầng:
  Jobs route:    CRUD Job Descriptions
  Scan route:    Upload + start + status + SSE
  Reports route: Get report + download PDF
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from api.main import app, scan_jobs
from schemas.report_schema import ReportMeta, ReportOutput
from schemas.score_schema import (
    CandidateScore,
    RankedCandidate,
    ScoreTier,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_scan_jobs():
    """Reset scan_jobs trước mỗi test."""
    scan_jobs.clear()
    yield
    scan_jobs.clear()


@pytest.fixture
def client() -> TestClient:
    """Synchronous test client."""
    return TestClient(app)


def _make_completed_job(run_id: str = "TEST01") -> dict:
    """Helper: tạo 1 completed job trong scan_jobs."""
    report = ReportOutput(
        meta      = ReportMeta(
            report_id        = "RPT001",
            job_id           = "backend-2025",
            job_title        = "Senior Backend Engineer",
            total_candidates = 2,
            shortlist_count  = 1,
        ),
        pdf_bytes    = b"%PDF-1.4 fake pdf content here",
        summary_text = "2 candidates. Top: Nguyen Van A.",
        shortlist    = ["Nguyen Van A"],
    )
    score = CandidateScore(
        candidate_name = "Nguyen Van A",
        job_title      = "Senior Backend Engineer",
        total_score    = 85.0,
        tier           = ScoreTier.STRONG,
        strengths      = ["Python expert", "AWS certified"],
        concerns       = ["No Kubernetes"],
        recommendation = "Recommend for interview.",
        rubric_used    = "backend-2025",
    )
    ranked = [RankedCandidate(rank=1, percentile=100.0, score=score)]

    scan_jobs[run_id] = {
        "status":             "completed",
        "progress":           1.0,
        "step":               "Done!",
        "result":             report,
        "error":              None,
        "job_id":             "backend-2025",
        "job_title":          "Senior Backend Engineer",
        "file_count":         2,
        "ranked_candidates":  ranked,
    }
    return scan_jobs[run_id]


# ──────────────────────────────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ──────────────────────────────────────────────────────────────────────────────
# Jobs route
# ──────────────────────────────────────────────────────────────────────────────

class TestJobsRoute:

    @patch("api.routes.jobs.ingest_job_description")
    def test_create_job_success(
        self, mock_ingest: MagicMock, client: TestClient
    ) -> None:
        mock_ingest.return_value = 7  # 7 chunks ingested

        r = client.post("/api/v1/jobs/", json={
            "job_id":    "backend-2025",
            "job_title": "Senior Backend Engineer",
            "requirements": [
                "3+ years Python backend development",
                "FastAPI experience required",
                "AWS Lambda and S3",
            ],
            "nice_to_have": ["Kubernetes"],
        })

        assert r.status_code == 201
        data = r.json()
        assert data["job_id"]       == "backend-2025"
        assert data["chunks_count"] == 7
        mock_ingest.assert_called_once()

    def test_create_job_empty_requirements_fails(
        self, client: TestClient
    ) -> None:
        r = client.post("/api/v1/jobs/", json={
            "job_id":       "test",
            "job_title":    "Test",
            "requirements": [],   # Empty → validation error
        })
        assert r.status_code == 422

    @patch("api.routes.jobs.list_jobs")
    def test_list_jobs(
        self, mock_list: MagicMock, client: TestClient
    ) -> None:
        mock_list.return_value = [
            {"job_id": "backend-2025", "job_title": "Senior Backend"},
            {"job_id": "frontend-2025", "job_title": "Frontend Dev"},
        ]
        r = client.get("/api/v1/jobs/")
        assert r.status_code == 200
        assert r.json()["total"] == 2

    @patch("api.routes.jobs.clear_job")
    def test_delete_job(
        self, mock_clear: MagicMock, client: TestClient
    ) -> None:
        r = client.delete("/api/v1/jobs/backend-2025")
        assert r.status_code == 204
        mock_clear.assert_called_once_with("backend-2025")

    @patch("api.routes.jobs.ingest_rubric")
    def test_create_rubric_valid_weights(
        self, mock_ingest: MagicMock, client: TestClient
    ) -> None:
        r = client.post("/api/v1/jobs/backend-2025/rubric", json={
            "job_id":    "backend-2025",
            "job_title": "Senior Backend Engineer",
            "dimensions": {
                "technical_skills": {"weight": 35, "description": "...", "scored_by": "programmatic"},
                "experience":       {"weight": 20, "description": "...", "scored_by": "programmatic"},
                "education":        {"weight": 15, "description": "...", "scored_by": "llm"},
                "achievements":     {"weight": 20, "description": "...", "scored_by": "llm"},
                "soft_skills":      {"weight": 10, "description": "...", "scored_by": "llm"},
            },
        })
        assert r.status_code == 201
        mock_ingest.assert_called_once()

    def test_create_rubric_invalid_weights_fails(
        self, client: TestClient
    ) -> None:
        r = client.post("/api/v1/jobs/backend-2025/rubric", json={
            "job_id":    "backend-2025",
            "job_title": "Dev",
            "dimensions": {
                "technical_skills": {"weight": 50, "description": "...", "scored_by": "llm"},
                # Total = 50, not 100
            },
        })
        assert r.status_code == 422
        assert "100" in r.json()["detail"]


# ──────────────────────────────────────────────────────────────────────────────
# Scan route
# ──────────────────────────────────────────────────────────────────────────────

class TestScanRoute:

    def _make_pdf_file(self, name: str = "cv.pdf") -> tuple:
        return ("files", (name, BytesIO(b"%PDF fake content"), "application/pdf"))

    @patch("api.routes.scan._run_pipeline_background")
    def test_start_scan_success(
        self, mock_bg: MagicMock, client: TestClient
    ) -> None:
        """Upload valid PDF → 202 Accepted với run_id."""
        r = client.post(
            "/api/v1/scan/start",
            files = [self._make_pdf_file("cv1.pdf")],
            data  = {
                "job_id":    "backend-2025",
                "job_title": "Senior Backend Engineer",
                "api_key":   "fake-gemini-key",
            },
        )
        assert r.status_code == 202
        data = r.json()
        assert "run_id"     in data
        assert "stream_url" in data
        assert data["file_count"] == 1

    @patch("api.routes.scan._run_pipeline_background")
    def test_start_scan_multiple_files(
        self, mock_bg: MagicMock, client: TestClient
    ) -> None:
        r = client.post(
            "/api/v1/scan/start",
            files = [
                self._make_pdf_file("cv1.pdf"),
                self._make_pdf_file("cv2.pdf"),
                self._make_pdf_file("cv3.pdf"),
            ],
            data = {
                "job_id":    "backend-2025",
                "job_title": "Dev",
                "api_key":   "key",
            },
        )
        assert r.status_code == 202
        assert r.json()["file_count"] == 3

    def test_start_scan_no_files_fails(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/scan/start",
            data = {"job_id": "backend-2025", "job_title": "Dev", "api_key": "key"},
        )
        # FastAPI validation: files is required
        assert r.status_code in {400, 422}

    def test_start_scan_unsupported_file_type_fails(
        self, client: TestClient
    ) -> None:
        r = client.post(
            "/api/v1/scan/start",
            files = [("files", ("resume.txt", BytesIO(b"text content"), "text/plain"))],
            data  = {"job_id": "j", "job_title": "t", "api_key": "k"},
        )
        assert r.status_code == 415

    def test_get_status_running(self, client: TestClient) -> None:
        """Status endpoint với running job."""
        scan_jobs["TESTRUN"] = {
            "status": "running", "progress": 0.45,
            "step": "Matching CVs...", "result": None, "error": None,
        }
        r = client.get("/api/v1/scan/status/TESTRUN")
        assert r.status_code == 200
        data = r.json()
        assert data["status"]   == "running"
        assert data["progress"] == 0.45

    def test_get_status_not_found(self, client: TestClient) -> None:
        r = client.get("/api/v1/scan/status/NOTEXIST")
        assert r.status_code == 404

    def test_get_status_completed_has_report_id(
        self, client: TestClient
    ) -> None:
        _make_completed_job("DONE01")
        r = client.get("/api/v1/scan/status/DONE01")
        assert r.status_code == 200
        assert r.json()["report_id"] == "RPT001"

    def test_stream_not_found(self, client: TestClient) -> None:
        r = client.get("/api/v1/scan/stream/NOTEXIST")
        assert r.status_code == 404

    def test_stream_returns_sse_content_type(
        self, client: TestClient
    ) -> None:
        """SSE endpoint trả về text/event-stream."""
        _make_completed_job("STREAM01")
        r = client.get("/api/v1/scan/stream/STREAM01", headers={"Accept": "text/event-stream"})
        assert "text/event-stream" in r.headers.get("content-type", "")


# ──────────────────────────────────────────────────────────────────────────────
# Reports route
# ──────────────────────────────────────────────────────────────────────────────

class TestReportsRoute:

    def test_get_report_success(self, client: TestClient) -> None:
        _make_completed_job("RPT01")
        r = client.get("/api/v1/reports/run/RPT01")
        assert r.status_code == 200
        data = r.json()
        assert data["report_id"]        == "RPT001"
        assert data["job_id"]           == "backend-2025"
        assert data["total_candidates"] == 2
        assert data["has_pdf"]          is True
        assert len(data["shortlist"])   == 1
        assert data["shortlist"][0]["name"] == "Nguyen Van A"

    def test_get_report_not_found(self, client: TestClient) -> None:
        r = client.get("/api/v1/reports/run/NOEXIST")
        assert r.status_code == 404

    def test_get_report_not_ready_returns_202(
        self, client: TestClient
    ) -> None:
        scan_jobs["NOTDONE"] = {
            "status": "running", "progress": 0.5,
            "step": "...", "result": None, "error": None,
        }
        r = client.get("/api/v1/reports/run/NOTDONE")
        assert r.status_code == 202

    def test_get_report_failed_returns_400(
        self, client: TestClient
    ) -> None:
        scan_jobs["FAILED01"] = {
            "status": "failed", "progress": 0.0,
            "step": "Failed", "result": None,
            "error": "LLM timeout",
        }
        r = client.get("/api/v1/reports/run/FAILED01")
        assert r.status_code == 400

    def test_download_pdf_returns_bytes(self, client: TestClient) -> None:
        _make_completed_job("DL01")
        r = client.get("/api/v1/reports/run/DL01/download")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.content.startswith(b"%PDF")

    def test_download_pdf_has_correct_filename(
        self, client: TestClient
    ) -> None:
        _make_completed_job("DL02")
        r = client.get("/api/v1/reports/run/DL02/download")
        cd = r.headers.get("content-disposition", "")
        assert "cv_report_" in cd
        assert ".pdf"       in cd

    def test_download_pdf_not_ready_fails(
        self, client: TestClient
    ) -> None:
        scan_jobs["NOTDL"] = {
            "status": "running", "progress": 0.5,
            "step": "...", "result": None, "error": None,
        }
        r = client.get("/api/v1/reports/run/NOTDL/download")
        assert r.status_code == 400

    def test_get_shortlist_returns_top5(self, client: TestClient) -> None:
        _make_completed_job("SL01")
        r = client.get("/api/v1/reports/run/SL01/shortlist")
        assert r.status_code == 200
        data = r.json()
        assert "shortlist" in data
        assert data["shortlist"][0]["name"]  == "Nguyen Van A"
        assert data["shortlist"][0]["score"] == 85.0
        assert data["shortlist"][0]["tier"]  == ScoreTier.STRONG