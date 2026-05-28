"""
Agent 3: Scorer / Ranker Agent

Nhận MatchResult từ JD Matcher Agent (Agent 2)
→ Load rubric từ ChromaDB
→ Chấm điểm theo 5 dimensions (hybrid: programmatic + LLM)
→ Rank toàn bộ batch
→ Trả về list[RankedCandidate] cho Report Writer Agent (Agent 4)

Thiết kế quan trọng — Hybrid Scoring:
  ┌─────────────────────┬────────────────┬───────────────────────────────┐
  │ Dimension           │ Scored by      │ Lý do                         │
  ├─────────────────────┼────────────────┼───────────────────────────────┤
  │ technical_skills    │ Programmatic   │ Đếm matched skills → tỉ lệ %  │
  │ experience          │ Programmatic   │ Toán học: năm × domain_factor  │
  │ education           │ LLM            │ Cần hiểu ngữ nghĩa bằng cấp   │
  │ achievements        │ LLM            │ Cần đánh giá chất lượng impact │
  │ soft_skills         │ LLM            │ Cần đọc ngữ cảnh               │
  └─────────────────────┴────────────────┴───────────────────────────────┘

Tại sao không để LLM chấm hết?
  1. LLM "drift": cùng 1 CV, chạy 3 lần → 3 điểm khác nhau
  2. Chậm và tốn API quota (3 LLM calls thay 5)
  3. Toán học không cần AI — đếm skills khớp là đếm, không cần judgement
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from prompts.scorer_prompt import SCORER_SYSTEM_PROMPT, SCORER_USER_TEMPLATE
from rag.retriever import get_rubric

from agents.cv_parser import ParsedCV
from schemas.match_schema import MatchResult
try:
    from langsmith import traceable as _traceable
    _TRACE = True
except ImportError:
    _TRACE = False

from schemas.score_schema import (
    CandidateScore,
    DimensionScore,
    RankedCandidate,
    ScoreBreakdown,
    ScoringRubric,
    DEFAULT_RUBRIC,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2

# Dimensions tính bằng toán học — không cần LLM
PROGRAMMATIC_DIMS = {"technical_skills", "experience"}

# Dimensions cần LLM judgement
LLM_DIMS = {"education", "achievements", "soft_skills"}


class ScorerAgent:
    """
    Agent 3: Score candidates and rank the full batch.

    Usage:
        agent = ScorerAgent(api_key="GEMINI_KEY")

        # Score 1 ứng viên
        score = agent.score(match_result, candidate_profile, job_id="backend-2025")

        # Score và rank toàn bộ batch
        ranked = agent.score_and_rank(
            matches=[...],       # list[MatchResult]
            candidates=[...],    # list[ParsedCV]
            job_id="backend-2025",
        )
        # ranked[0] là ứng viên tốt nhất
    """

    def __init__(
        self,
        api_key:       str,
        model:         str = "gemini-3.5-flash",
        system_prompt: str = SCORER_SYSTEM_PROMPT,
    ) -> None:
        self.client        = genai.Client(api_key=api_key)
        self.model_name    = model
        self.system_prompt = system_prompt

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def score(
        self,
        match:     MatchResult,
        candidate: ParsedCV,
        job_id:    str,
    ) -> CandidateScore:
        if _TRACE:
            from langsmith import traceable
            return traceable(
                name="ScorerAgent.score",
                run_type="chain",
                tags=["scorer", "agent-3"],
                metadata={"candidate": candidate.candidate_name, "job_id": job_id},
            )(self._score_internal)(match, candidate, job_id)
        return self._score_internal(match, candidate, job_id)

    def _score_internal(
        self,
        match:     MatchResult,
        candidate: ParsedCV,
        job_id:    str,
    ) -> CandidateScore:
        """
        Score 1 ứng viên dựa trên MatchResult và CandidateProfile.

        Args:
            match:     Output của JDMatcherAgent
            candidate: Output của CVParserAgent
            job_id:    Dùng để load rubric từ ChromaDB

        Returns:
            CandidateScore với điểm từng dimension và tổng điểm
        """
        logger.info("Scoring candidate: %s", candidate.candidate_name)

        # Bước 1: Load rubric
        rubric = self._load_rubric(job_id)

        # Bước 2: Chấm programmatic dimensions (không cần LLM)
        prog_scores = self._score_programmatic(match, candidate, rubric)

        # Bước 3: Chấm LLM dimensions (1 API call cho tất cả)
        llm_scores, strengths, concerns, recommendation = (
            self._score_llm_dimensions(match, candidate, rubric)
        )

        # Bước 4: Gộp và tính tổng
        all_scores  = {**prog_scores, **llm_scores}
        total_score = sum(ds.raw_score for ds in all_scores.values())
        total_score = round(min(100.0, max(0.0, total_score)), 2)

        breakdown = ScoreBreakdown(
            technical_skills = all_scores.get("technical_skills"),
            experience       = all_scores.get("experience"),
            education        = all_scores.get("education"),
            achievements     = all_scores.get("achievements"),
            soft_skills      = all_scores.get("soft_skills"),
        )

        result = CandidateScore(
            candidate_name = candidate.candidate_name,
            job_title      = match.job_title,
            total_score    = total_score,
            breakdown      = breakdown,
            strengths      = strengths,
            concerns       = concerns,
            recommendation = recommendation,
            low_confidence = match.low_confidence,
            warnings       = list(match.warnings),
            rubric_used    = job_id,
        )

        logger.info(
            "Scored %s: %.1f/100 (%s)",
            candidate.candidate_name, total_score, result.tier,
        )
        return result

    def score_and_rank(
        self,
        matches:    list[MatchResult],
        candidates: list[ParsedCV],
        job_id:     str,
    ) -> list[RankedCandidate]:
        """
        Score toàn bộ batch và trả về danh sách xếp hạng.

        Args:
            matches:    list MatchResult từ JDMatcherAgent (cùng thứ tự với candidates)
            candidates: list CandidateProfile từ CVParserAgent
            job_id:     Dùng để load rubric

        Returns:
            list[RankedCandidate] sắp xếp từ cao xuống thấp (rank 1 = tốt nhất)
        """
        if len(matches) != len(candidates):
            raise ValueError(
                f"matches ({len(matches)}) and candidates ({len(candidates)}) "
                "must have the same length"
            )

        scores: list[CandidateScore] = []

        for i, (match, candidate) in enumerate(zip(matches, candidates), 1):
            logger.info(
                "Batch scoring %d/%d: %s", i, len(candidates), candidate.candidate_name
            )
            try:
                score = self.score(match, candidate, job_id)
            except Exception as e:
                logger.error("Score failed for %s: %s — using 0", candidate.candidate_name, e)
                score = self._fallback_score(candidate.candidate_name, match.job_title, str(e))

            scores.append(score)

            # Rate limit: 4s giữa các LLM calls
            if i < len(candidates):
                time.sleep(4)

        return self._rank(scores)

    # ──────────────────────────────────────────────────────────────────────────
    # Private: Rubric loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_rubric(self, job_id: str) -> ScoringRubric:
        """
        Load rubric từ ChromaDB. Nếu không có → dùng DEFAULT_RUBRIC.
        Default rubric đảm bảo system luôn chạy được dù chưa setup.
        """
        raw = get_rubric(job_id)
        if not raw:
            logger.warning(
                "No rubric found for job_id='%s' — using DEFAULT_RUBRIC", job_id
            )
            return DEFAULT_RUBRIC

        try:
            return ScoringRubric.model_validate(raw)
        except Exception as e:
            logger.error(
                "Invalid rubric for job_id='%s': %s — using DEFAULT_RUBRIC", job_id, e
            )
            return DEFAULT_RUBRIC

    # ──────────────────────────────────────────────────────────────────────────
    # Private: Programmatic scoring (no LLM)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_programmatic(
        self,
        match:     MatchResult,
        candidate: ParsedCV,
        rubric:    ScoringRubric,
    ) -> dict[str, DimensionScore]:
        """
        Tính điểm bằng toán học cho technical_skills và experience.
        Kết quả hoàn toàn deterministic — cùng input → cùng output.
        """
        scores: dict[str, DimensionScore] = {}

        if "technical_skills" in rubric.dimensions:
            scores["technical_skills"] = self._score_technical(match, rubric)

        if "experience" in rubric.dimensions:
            scores["experience"] = self._score_experience(match, rubric)

        return scores

    def _score_technical(
        self, match: MatchResult, rubric: ScoringRubric
    ) -> DimensionScore:
        """
        Tính điểm technical skills dựa trên skill_gap từ MatchResult.

        Formula:
          base     = matched_count / total_required_count
          penalty  = len(missing_critical) × 0.15   (mỗi critical miss = -15%)
          score    = (base - penalty) × max_weight
        """
        gap      = match.skill_gap
        max_w    = rubric.dimensions["technical_skills"].weight

        total_required = len(gap.matched) + len(gap.missing_critical)
        if total_required == 0:
            # Không có data → cho điểm giữa
            return DimensionScore(
                dimension  = "technical_skills",
                max_score  = max_w,
                raw_score  = max_w * 0.4,
                percentage = 40.0,
                rationale  = "No skill data available — scored conservatively.",
                scored_by  = "programmatic",
            )

        # Tỉ lệ skills matched
        match_ratio = len(gap.matched) / total_required

        # Phạt mỗi critical skill bị thiếu
        critical_penalty = len(gap.missing_critical) * 0.12
        effective_ratio  = max(0.0, match_ratio - critical_penalty)

        raw_score = round(effective_ratio * max_w, 2)

        # Rationale
        rationale_parts = [
            f"Matched {len(gap.matched)}/{total_required} required skills.",
        ]
        if gap.missing_critical:
            rationale_parts.append(
                f"Missing critical: {', '.join(gap.missing_critical[:3])}."
            )
        if gap.bonus:
            rationale_parts.append(
                f"Bonus skills: {', '.join(gap.bonus[:3])}."
            )

        return DimensionScore(
            dimension  = "technical_skills",
            max_score  = max_w,
            raw_score  = raw_score,
            percentage = round(effective_ratio * 100, 2),
            rationale  = " ".join(rationale_parts),
            scored_by  = "programmatic",
        )

    def _score_experience(
        self, match: MatchResult, rubric: ScoringRubric
    ) -> DimensionScore:
        """
        Tính điểm experience.

        Formula:
          year_score   = min(candidate_years / required_years, 1.2) × 0.6
          domain_score = domain_relevance × 0.4
          total        = (year_score + domain_score) × max_weight
        """
        exp   = match.experience
        max_w = rubric.dimensions["experience"].weight

        required  = exp.required_years  or 0.0
        actual    = exp.candidate_years or 0.0
        relevance = exp.domain_relevance

        if required == 0 and actual == 0:
            return DimensionScore(
                dimension  = "experience",
                max_score  = max_w,
                raw_score  = max_w * 0.4,
                percentage = 40.0,
                rationale  = "Experience data unavailable — scored conservatively.",
                scored_by  = "programmatic",
            )

        # Year component (max 1.2 để tránh quá nhiều điểm bonus)
        year_ratio  = min(actual / required, 1.2) if required > 0 else 1.0
        year_score  = year_ratio * 0.6

        # Domain relevance component
        domain_score = relevance * 0.4

        effective_ratio = min(year_score + domain_score, 1.0)
        raw_score       = round(effective_ratio * max_w, 2)

        # Rationale
        if required > 0:
            year_note = (
                f"Has {actual:.1f} yrs (required {required:.0f})."
                if actual >= required
                else f"Has {actual:.1f} yrs, below required {required:.0f}."
            )
        else:
            year_note = f"Has {actual:.1f} years experience."

        domain_note = (
            f"Domain relevance: {relevance:.0%}. "
            + (exp.relevance_note or "")
        ).strip()

        return DimensionScore(
            dimension  = "experience",
            max_score  = max_w,
            raw_score  = raw_score,
            percentage = round(effective_ratio * 100, 2),
            rationale  = f"{year_note} {domain_note}",
            scored_by  = "programmatic",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private: LLM scoring (education, achievements, soft_skills)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_llm_dimensions(
        self,
        match:     MatchResult,
        candidate: ParsedCV,
        rubric:    ScoringRubric,
    ) -> tuple[dict[str, DimensionScore], list[str], list[str], str]:
        """
        1 LLM call để chấm tất cả LLM dimensions cùng lúc.
        Trả về: (scores_dict, strengths, concerns, recommendation)
        """
        # Lọc ra LLM dimensions có trong rubric
        llm_dims = {
            k: v for k, v in rubric.dimensions.items()
            if v.scored_by in {"llm", "hybrid"} and k in LLM_DIMS
        }

        if not llm_dims:
            return {}, [], [], "Not enough data to recommend."

        prompt = self._build_llm_prompt(match, candidate, rubric, llm_dims)
        raw    = self._call_llm_with_retry(prompt)

        return self._parse_llm_scores(raw, llm_dims, rubric)

    def _build_llm_prompt(
        self,
        match:     MatchResult,
        candidate: ParsedCV,
        rubric:    ScoringRubric,
        llm_dims:  dict,
    ) -> str:
        """Format prompt cho LLM scoring call."""
        # MatchResult summary (không cần toàn bộ JSON)
        match_summary = {
            "candidate_name":     match.candidate_name,
            "job_title":          match.job_title,
            "match_summary":      match.match_summary,
            "missing_critical":   match.skill_gap.missing_critical,
            "matched_skills":     match.skill_gap.matched,
            "experience": {
                "years":          match.experience.candidate_years,
                "domain_relevance": match.experience.domain_relevance,
            },
            "requirement_matches": [
                {
                    "requirement":        r.requirement,
                    "match_level":        r.match_level,
                    "candidate_evidence": r.candidate_evidence,
                }
                for r in match.requirement_matches
            ],
        }

        # ParsedCV v2: dùng markdown sections thay vì structured fields
        # Lấy phần relevant từ markdown (Education, Skills, Certifications)
        relevant_sections = []
        if candidate.markdown:
            lines = candidate.markdown.splitlines()
            capture = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    section_name = stripped.lstrip("#").strip().lower()
                    capture = any(kw in section_name for kw in [
                        "education", "skill", "certif", "language",
                        "award", "achievement", "summary", "profile",
                    ])
                if capture:
                    relevant_sections.append(line)

        candidate_summary = {
            "candidate_name": candidate.candidate_name,
            "file_name":      candidate.file_name,
            "detected_sections": candidate.sections,
            "cv_excerpt":     "\n".join(relevant_sections[:60]) if relevant_sections
                              else "(No structured sections detected — see full CV in ChromaDB)",
        }

        # Format dimensions cần score
        dims_text = "\n".join([
            f"- {name} (max {cfg.weight} pts): {cfg.description}"
            for name, cfg in llm_dims.items()
        ])

        return SCORER_USER_TEMPLATE.format(
            match_result_json  = json.dumps(match_summary,     ensure_ascii=False, indent=2),
            rubric_json        = json.dumps(
                {"job_title": rubric.job_title},
                ensure_ascii=False, indent=2
            ),
            dimensions_to_score = dims_text,
        )

    def _call_llm_with_retry(self, prompt: str) -> dict[str, Any]:
        """Gọi LLM với retry — pattern giống 2 agent trước."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_text = self._call_llm(prompt)
                return self._parse_json_response(raw_text)

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "LLM invalid JSON (attempt %d/%d): %s", attempt, MAX_RETRIES, e
                )
                if attempt < MAX_RETRIES:
                    prompt += (
                        "\n\nIMPORTANT: Output ONLY the raw JSON object. "
                        "No markdown. No explanation."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(
                        f"LLM failed to return valid JSON after {MAX_RETRIES} attempts"
                    ) from e

        raise RuntimeError("Unexpected state in _call_llm_with_retry")

    def _call_llm(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model    = self.model_name,
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                system_instruction = self.system_prompt,
                temperature        = 0.0,
                max_output_tokens  = 2048,
            ),
        )
        return response.text

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """Strip fences và parse JSON — pattern nhất quán với 2 agent trước."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$",           "", text, flags=re.MULTILINE)
        text = text.strip()

        if not text.startswith("{"):
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in LLM response")
            text = text[start:]

        return json.loads(text)

    def _parse_llm_scores(
        self,
        data:     dict[str, Any],
        llm_dims: dict,
        rubric:   ScoringRubric,
    ) -> tuple[dict[str, DimensionScore], list[str], list[str], str]:
        """Validate và convert LLM output thành DimensionScore objects."""
        scores: dict[str, DimensionScore] = {}

        for item in data.get("dimension_scores", []):
            dim_name = item.get("dimension")
            if dim_name not in llm_dims:
                continue  # LLM trả về dimension không yêu cầu → bỏ qua

            max_w = rubric.dimensions[dim_name].weight
            raw   = float(item.get("raw_score", 0))
            raw   = max(0.0, min(float(max_w), raw))  # Clamp

            scores[dim_name] = DimensionScore(
                dimension  = dim_name,
                max_score  = max_w,
                raw_score  = round(raw, 2),
                percentage = round(raw / max_w * 100, 2) if max_w > 0 else 0.0,
                rationale  = item.get("rationale", ""),
                scored_by  = "llm",
            )

        # Fallback: nếu LLM bỏ sót 1 dimension → cho điểm giữa
        for dim_name, cfg in llm_dims.items():
            if dim_name not in scores:
                logger.warning("LLM missed dimension '%s' — using fallback", dim_name)
                scores[dim_name] = DimensionScore(
                    dimension  = dim_name,
                    max_score  = cfg.weight,
                    raw_score  = round(cfg.weight * 0.4, 2),
                    percentage = 40.0,
                    rationale  = "LLM did not score this dimension — scored conservatively.",
                    scored_by  = "llm",
                )

        strengths      = data.get("strengths",      [])[:3]
        concerns       = data.get("concerns",       [])[:3]
        recommendation = data.get("recommendation", "No recommendation provided.")

        return scores, strengths, concerns, recommendation

    # ──────────────────────────────────────────────────────────────────────────
    # Private: Ranking
    # ──────────────────────────────────────────────────────────────────────────

    def _rank(self, scores: list[CandidateScore]) -> list[RankedCandidate]:
        """
        Sắp xếp tất cả CandidateScore từ cao → thấp và tính percentile.

        Percentile = (rank từ cuối / total) × 100
        → Candidate điểm cao nhất = percentile 100
        → Candidate điểm thấp nhất = percentile gần 0
        """
        if not scores:
            return []

        # Sắp xếp giảm dần theo total_score
        sorted_scores = sorted(scores, key=lambda s: s.total_score, reverse=True)
        total         = len(sorted_scores)

        ranked: list[RankedCandidate] = []
        for rank_idx, score in enumerate(sorted_scores, 1):
            # Percentile: rank từ cuối / total × 100
            percentile = round((total - rank_idx + 1) / total * 100, 1)
            ranked.append(
                RankedCandidate(
                    rank       = rank_idx,
                    percentile = percentile,
                    score      = score,
                )
            )

        return ranked

    # ──────────────────────────────────────────────────────────────────────────
    # Private: Fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _fallback_score(
        self,
        candidate_name: str,
        job_title:      str,
        error:          str,
    ) -> CandidateScore:
        """Trả về score 0 khi scoring fail — không crash batch."""
        return CandidateScore(
            candidate_name = candidate_name,
            job_title      = job_title,
            total_score    = 0.0,
            low_confidence = True,
            warnings       = [f"Scoring failed: {error}"],
            rubric_used    = "none",
        )