"""
LangGraph Nodes (v2)

Cập nhật để tương thích với:
  - Agent 1 v2: CVParserAgent → ParsedCV (không còn CandidateProfile, không cần api_key)
  - Agent 2 v2: JDMatcherAgent.match_batch() nhận jd_text + job_title + job_id
  - Agent 3:    ScorerAgent.score_and_rank() nhận list[MatchResult] + list[ParsedCV]

Nguyên tắc mỗi node:
  1. Validate input state trước — fail sớm nếu thiếu data
  2. Gọi agent tương ứng
  3. Catch exception → ghi errors, không re-raise
  4. Cập nhật status + progress + current_step
  5. Return dict partial update — LangGraph tự merge vào state
"""
from __future__ import annotations

import logging
import os
from typing import Any

from agents.cv_parser import CVParserAgent, ParsedCV
from agents.jd_matcher import JDMatcherAgent
from agents.report_writer import ReportWriterAgent
from agents.scorer import ScorerAgent
from graph.state import GraphState, PipelineStatus

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Node 1: Parse CVs
# ──────────────────────────────────────────────────────────────────────────────

def parse_node(state: GraphState) -> dict[str, Any]:
    """
    Node 1: Convert CV files → Markdown → Chunks → ChromaDB.
    Trả về list[ParsedCV] với cv_id để Node 2 query ChromaDB.

    Không gọi LLM — nhanh hơn v1, không tốn API quota.

    Input state:  file_paths
    Output state: parsed_candidates, parse_errors, status, progress
    """
    # LangSmith node name
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        try:
            from langsmith import get_current_run_tree
            rt = get_current_run_tree()
            if rt:
                rt.name = "node:parse"
        except Exception:
            pass

    logger.info("=== NODE: parse (%d files) ===", len(state.file_paths))

    # reprocess=True: nếu cùng file upload lại → xóa chunks cũ, process lại
    agent  = CVParserAgent(reprocess=True)
    parsed: list[ParsedCV] = []
    errors: list[str]      = []
    total  = len(state.file_paths)

    for i, path in enumerate(state.file_paths, 1):
        try:
            profile = agent.parse(path)
            parsed.append(profile)
            logger.info(
                "[%d/%d] Parsed: %s | cv_id=%s | %d chunks | sections=%s",
                i, total,
                profile.candidate_name,
                profile.cv_id,
                profile.chunk_count,
                profile.sections,
            )
        except Exception as e:
            msg = f"Failed to parse {path}: {e}"
            logger.error(msg)
            errors.append(msg)

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
    Node 2: Parse JD → query ChromaDB evidence per skill → analyze match.

    Pipeline nội bộ của agent:
      1. LLM parse jd_text → ParsedJD (required_skills, experience...)  [cache]
      2. Với mỗi required_skill: query cv_chunks của candidate đó
      3. LLM analyze evidence → MatchResult

    Input state:  parsed_candidates, jd_text, job_id, job_title, api_key
    Output state: match_results, status, progress
    """
    logger.info("=== NODE: match (%d candidates) ===", len(state.parsed_candidates))

    if not state.parsed_candidates:
        return {
            "status": PipelineStatus.FAILED,
            "errors": ["match_node: parsed_candidates is empty"],
        }

    if not state.jd_text or len(state.jd_text.strip()) < 10:
        return {
            "status": PipelineStatus.FAILED,
            "errors": [
                "match_node: jd_text is missing or too short. "
                "Provide full Job Description text when starting the scan."
            ],
        }

    agent   = JDMatcherAgent(api_key=state.api_key)
    results = agent.match_batch(
        candidates = state.parsed_candidates,
        jd_text    = state.jd_text,
        job_title  = state.job_title,
        job_id     = state.job_id,
    )

    low_conf = sum(1 for r in results if r.low_confidence)
    if low_conf:
        logger.warning("%d/%d matches flagged low_confidence", low_conf, len(results))

    return {
        "status":        PipelineStatus.SCORING,
        "current_step":  f"Matched {len(results)} candidates. Starting scoring...",
        "match_results": results,
        "progress":      0.55,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 3: Score and rank
# ──────────────────────────────────────────────────────────────────────────────

def score_node(state: GraphState) -> dict[str, Any]:
    """
    Node 3: Chấm điểm hybrid (programmatic + LLM) → xếp hạng toàn batch.

    Input state:  match_results, parsed_candidates, job_id, api_key
    Output state: ranked_candidates, status, progress

    Note: ScorerAgent nhận list[ParsedCV] (v2) thay vì list[CandidateProfile] (v1).
    Scorer dùng parsed_candidates để lấy education/achievements cho LLM scoring.
    """
    logger.info("=== NODE: score (%d candidates) ===", len(state.match_results))

    if not state.match_results:
        return {
            "status": PipelineStatus.FAILED,
            "errors": ["score_node: match_results is empty"],
        }

    if len(state.match_results) != len(state.parsed_candidates):
        logger.warning(
            "Mismatch: %d match_results vs %d parsed_candidates — using min length",
            len(state.match_results), len(state.parsed_candidates),
        )

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
    Node 4: Tạo PDF report từ ranked candidates.
    Không gọi LLM — deterministic, nhanh.

    Input state:  ranked_candidates, job_id, job_title
    Output state: report, status, progress
    """
    logger.info("=== NODE: report (%d ranked) ===", len(state.ranked_candidates))

    if not state.ranked_candidates:
        return {
            "status": PipelineStatus.FAILED,
            "errors": ["report_node: ranked_candidates is empty"],
        }

    use_s3 = bool(os.getenv("S3_BUCKET_NAME"))
    agent  = ReportWriterAgent(use_s3=use_s3)

    report = agent.write(
        ranked    = state.ranked_candidates,
        job_title = state.job_title,
        job_id    = state.job_id,
    )

    return {
        "status":       PipelineStatus.COMPLETED,
        "current_step": (
            f"Done! Report {report.meta.report_id} — "
            f"{len(report.shortlist)} candidates shortlisted."
        ),
        "report":   report,
        "progress": 1.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edges
# ──────────────────────────────────────────────────────────────────────────────

def should_continue_after_parse(state: GraphState) -> str:
    if state.status == PipelineStatus.FAILED:
        logger.error("Pipeline failed at parse: %s", state.errors)
        return "failed"
    return "match"


def should_continue_after_match(state: GraphState) -> str:
    if state.status == PipelineStatus.FAILED:
        return "failed"
    return "score"


def should_continue_after_score(state: GraphState) -> str:
    if state.status == PipelineStatus.FAILED:
        return "failed"
    return "report"