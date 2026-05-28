"""
LangGraph Workflow — State Machine Orchestrator

Kết nối 4 nodes thành 1 graph có:
  - Conditional edges (quyết định luồng chạy)
  - Error handling tại mỗi edge
  - Checkpoint (có thể pause/resume nếu cần)

Graph structure:
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  parse  │
                    └────┬────┘
                         │
              ┌──────────┴──────────┐
         "match"              "failed"
              │                     │
         ┌────▼────┐           ┌────▼────┐
         │  match  │           │   END   │
         └────┬────┘           └─────────┘
              │
    ┌─────────┴─────────┐
"score"             "failed"
    │                   │
┌───▼───┐          ┌────▼────┐
│ score │          │   END   │
└───┬───┘          └─────────┘
    │
┌───┴────────────┐
"report"    "failed"
    │              │
┌───▼────┐   ┌────▼────┐
│ report │   │   END   │
└───┬────┘   └─────────┘
    │
┌───▼───┐
│  END  │
└───────┘

Usage:
    from graph.workflow import build_graph, run_pipeline

    # Build once, reuse nhiều lần
    graph = build_graph()

    # Run pipeline
    final_state = run_pipeline(
        graph      = graph,
        file_paths = ["cv1.pdf", "cv2.pdf"],
        job_id     = "backend-2025",
        job_title  = "Senior Backend Engineer",
        api_key    = "GEMINI_KEY",
    )
    print(final_state.report.summary_text)
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    match_node,
    parse_node,
    report_node,
    score_node,
    should_continue_after_match,
    should_continue_after_parse,
    should_continue_after_score,
)
from graph.state import GraphState, PipelineStatus

logger = logging.getLogger(__name__)


def build_graph() -> Any:
    """
    Build và compile LangGraph state machine.

    Gọi 1 lần khi khởi động app, reuse cho mọi request.
    Compiled graph là thread-safe.

    Returns:
        CompiledGraph — dùng để gọi .invoke() hoặc .stream()
    """
    builder = StateGraph(GraphState)

    # ── Đăng ký các nodes ────────────────────────────────────────────────────
    builder.add_node("parse",  parse_node)
    builder.add_node("match",  match_node)
    builder.add_node("score",  score_node)
    builder.add_node("report", report_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    builder.add_edge(START, "parse")

    # ── Conditional edges ────────────────────────────────────────────────────
    # Sau parse: tiếp tục hay abort?
    builder.add_conditional_edges(
        "parse",
        should_continue_after_parse,
        {
            "match":  "match",  # Thành công → match node
            "failed": END,      # Thất bại → kết thúc
        },
    )

    # Sau match: tiếp tục hay abort?
    builder.add_conditional_edges(
        "match",
        should_continue_after_match,
        {
            "score":  "score",
            "failed": END,
        },
    )

    # Sau score: tiếp tục hay abort?
    builder.add_conditional_edges(
        "score",
        should_continue_after_score,
        {
            "report": "report",
            "failed": END,
        },
    )

    # Report luôn dẫn đến END
    builder.add_edge("report", END)

    return builder.compile()


def run_pipeline(
    graph:      Any,
    file_paths: list[str],
    job_id:     str,
    jd_text:    str,
    job_title:  str,
    api_key:    str,
) -> GraphState:
    """
    Chạy toàn bộ pipeline và trả về final state.

    Args:
        graph:      Compiled graph từ build_graph()
        file_paths: List đường dẫn đến CV files
        job_id:     ID của JD đã ingest vào ChromaDB
        job_title:  Tên vị trí
        jd_text:    Thông tin JD,
        api_key:    Gemini API key

    Returns:
        GraphState cuối cùng — đọc state.report để lấy kết quả
    """
    if not file_paths:
        raise ValueError("file_paths cannot be empty")

    initial_state = GraphState(
        file_paths = file_paths,
        job_id     = job_id,
        jd_text    = jd_text,
        job_title  = job_title,
        api_key    = api_key,
        status     = PipelineStatus.PARSING,
        current_step = f"Starting pipeline for {len(file_paths)} CVs...",
    )

    logger.info(
        "Pipeline started: %d CVs | job='%s' (%s)",
        len(file_paths), job_title, job_id,
    )

    final = graph.invoke(initial_state)

    # LangGraph trả về dict — convert lại về GraphState
    if isinstance(final, dict):
        final = GraphState(**final)

    logger.info(
        "Pipeline finished: status=%s | report=%s",
        final.status,
        final.report.meta.report_id if final.report else "None",
    )
    return final


def stream_pipeline(
    graph:      Any,
    file_paths: list[str],
    job_id:     str,
    jd_text:     str,
    job_title:  str,
    api_key:    str,
):
    """
    Stream pipeline events — dùng cho Chainlit real-time logging.

    Yields từng (node_name, state_update) dict khi mỗi node hoàn thành.
    Chainlit đọc state_update["current_step"] để hiển thị progress.

    Usage (trong Chainlit):
        async for event in stream_pipeline(graph, ...):
            node = event["node"]
            step = event["state"].get("current_step", "")
            await cl.Message(content=f"[{node}] {step}").send()
    """
    initial_state = GraphState(
        file_paths   = file_paths,
        job_id       = job_id,
        jd_text      = jd_text,
        job_title    = job_title,
        api_key      = api_key,
        status       = PipelineStatus.PARSING,
        current_step = f"Starting pipeline for {len(file_paths)} CVs...",
    )

    for event in graph.stream(initial_state):
        # event = {"node_name": {state_update_dict}}
        for node_name, state_update in event.items():
            yield {
                "node":  node_name,
                "state": state_update,
            }