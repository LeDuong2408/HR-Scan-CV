"""
LangGraph Nodes

Mỗi node là 1 function nhận GraphState → trả về dict (partial state update).
LangGraph tự merge dict đó vào state hiện tại.

Nguyên tắc thiết kế mỗi node:
  1. Log progress trước khi làm — Chainlit stream được
  2. Gọi agent tương ứng
  3. Catch exception → ghi vào state.errors, không re-raise
  4. Cập nhật status + progress
  5. Return dict — không return toàn bộ state

Tại sao không gọi agent.parse_batch() / match_batch()?
  Gọi batch methods trong node đơn giản hơn và đủ dùng cho demo.
  Khi cần scale, các node này có thể được fan-out thành
  Send() API của LangGraph để xử lý song song từng file.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.cv_parser import CVParserAgent
from agents.jd_matcher import JDMatcherAgent
from agents.report_writer import ReportWriterAgent
from agents.scorer import ScorerAgent
from graph.state import GraphState, PipelineStatus
from schemas.cv_schema import CandidateProfile

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Node 1: Parse CVs
# ──────────────────────────────────────────────────────────────────────────────

def parse_node(state: GraphState) -> dict[str, Any]:
    """
    Node 1: Parse tất cả CV files → list[CandidateProfile].

    Input state:  file_paths, api_key
    Output state: parsed_candidates, parse_errors, status, progress
    """
    logger.info("=== NODE: parse (%d files) ===", len(state.file_paths))

    agent  = CVParserAgent(api_key=state.api_key)
    parsed: list[CandidateProfile] = []
    errors: list[str] = []

    total = len(state.file_paths)

    for i, path in enumerate(state.file_paths, 1):
        try:
            profile = agent.parse(path)
            parsed.append(profile)
            logger.info("[%d/%d] Parsed: %s", i, total, profile.full_name)
        except Exception as e:
            msg = f"Failed to parse {path}: {e}"
            logger.error(msg)
            errors.append(msg)

    # Nếu không parse được file nào → fail sớm
    if not parsed:
        return {
            "status":       PipelineStatus.FAILED,
            "current_step": "All CV files failed to parse.",
            "parse_errors": errors,
            "errors":       ["No CVs could be parsed — pipeline aborted."],
            "progress":     0.25,
        }

    return {
        "status":            PipelineStatus.MATCHING,
        "current_step":      f"Parsed {len(parsed)}/{total} CVs. Starting JD matching...",
        "parsed_candidates": parsed,
        "parse_errors":      errors,
        "progress":          0.25,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 2: Match CVs against JD
# ──────────────────────────────────────────────────────────────────────────────

def match_node(state: GraphState) -> dict[str, Any]:
    """
    Node 2: Match từng CandidateProfile với JD → list[MatchResult].

    Input state:  parsed_candidates, job_id, job_title, api_key
    Output state: match_results, status, progress
    """
    logger.info("=== NODE: match (%d candidates) ===", len(state.parsed_candidates))

    if not state.parsed_candidates:
        return {
            "status": PipelineStatus.FAILED,
            "errors": ["match_node received empty parsed_candidates"],
        }

    agent   = JDMatcherAgent(api_key=state.api_key)
    results = agent.match_batch(
        candidates = state.parsed_candidates,
        job_id     = state.job_id,
        job_title  = state.job_title,
    )

    return {
        "status":       PipelineStatus.SCORING,
        "current_step": f"Matched {len(results)} candidates. Starting scoring...",
        "match_results": results,
        "progress":     0.55,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 3: Score and rank candidates
# ──────────────────────────────────────────────────────────────────────────────

def score_node(state: GraphState) -> dict[str, Any]:
    """
    Node 3: Score từng MatchResult → RankedCandidate list.

    Input state:  match_results, parsed_candidates, job_id, api_key
    Output state: ranked_candidates, status, progress
    """
    logger.info("=== NODE: score (%d candidates) ===", len(state.match_results))

    if not state.match_results:
        return {
            "status": PipelineStatus.FAILED,
            "errors": ["score_node received empty match_results"],
        }

    agent  = ScorerAgent(api_key=state.api_key)
    ranked = agent.score_and_rank(
        matches    = state.match_results,
        candidates = state.parsed_candidates,
        job_id     = state.job_id,
    )

    return {
        "status":            PipelineStatus.REPORTING,
        "current_step":      f"Scored {len(ranked)} candidates. Generating report...",
        "ranked_candidates": ranked,
        "progress":          0.80,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 4: Generate report
# ──────────────────────────────────────────────────────────────────────────────

def report_node(state: GraphState) -> dict[str, Any]:
    """
    Node 4: Generate PDF report từ ranked candidates.

    Input state:  ranked_candidates, job_id, job_title
    Output state: report, status, progress
    """
    logger.info("=== NODE: report (%d ranked) ===", len(state.ranked_candidates))

    if not state.ranked_candidates:
        return {
            "status": PipelineStatus.FAILED,
            "errors": ["report_node received empty ranked_candidates"],
        }

    # use_s3=True nếu có S3_BUCKET_NAME env var, ngược lại lưu local
    import os
    use_s3 = bool(os.getenv("S3_BUCKET_NAME"))

    agent  = ReportWriterAgent(use_s3=use_s3)
    report = agent.write(
        ranked    = state.ranked_candidates,
        job_title = state.job_title,
        job_id    = state.job_id,
    )

    return {
        "status":       PipelineStatus.COMPLETED,
        "current_step": f"Done! Report {report.meta.report_id} — {len(report.shortlist)} candidates shortlisted.",
        "report":       report,
        "progress":     1.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edges — quyết định node nào chạy tiếp theo
# ──────────────────────────────────────────────────────────────────────────────

def should_continue_after_parse(state: GraphState) -> str:
    """
    Sau parse node: tiếp tục match hay kết thúc với lỗi?

    Returns:
        "match"  → chạy match_node tiếp
        "failed" → kết thúc pipeline với lỗi
    """
    if state.status == PipelineStatus.FAILED:
        logger.error("Pipeline failed at parse stage: %s", state.errors)
        return "failed"
    return "match"


def should_continue_after_match(state: GraphState) -> str:
    """Sau match node → score hay failed?"""
    if state.status == PipelineStatus.FAILED:
        return "failed"
    # Cảnh báo nếu có empty matches nhưng không abort
    empty = sum(1 for m in state.match_results if m.low_confidence)
    if empty > 0:
        logger.warning("%d/%d matches have low confidence", empty, len(state.match_results))
    return "score"


def should_continue_after_score(state: GraphState) -> str:
    """Sau score node → report hay failed?"""
    if state.status == PipelineStatus.FAILED:
        return "failed"
    return "report"