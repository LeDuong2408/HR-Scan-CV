"""
Agent 2 (v2): JD Matcher Agent

Pipeline mới — 3 bước rõ ràng:

  BƯỚC 1 — Parse JD (LLM call #1, cache được)
    JD raw text → ParsedJD (required_skills, experience, education...)
    JD parsing chỉ cần gọi 1 lần cho toàn bộ batch.

  BƯỚC 2 — Query ChromaDB per skill (không dùng LLM)
    Với MỖI required_skill trong ParsedJD:
      → query cv_chunks collection với filter cv_id của ứng viên này
      → lấy top-K chunks liên quan nhất
    Kết quả: dict[skill → list[CV chunks]]
    Nếu skill không tìm thấy → evidence = empty → MISSING

  BƯỚC 3 — Analyze match (LLM call #2)
    Input: ParsedJD + cv_evidence_per_skill
    Output: MatchResult (requirement_matches, skill_gap, experience, summary)

Tại sao hướng query đúng hơn v1?
  v1 (sai): CV skills → query JD requirements
    → Miss requirements ứng viên không có
  v2 (đúng): JD skills → query CV chunks
    → Luôn check TẤT CẢ requirements
    → Evidence rỗng → LLM kết luận MISSING
    → Không bao giờ bỏ sót gap
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from agents.cv_parser import ParsedCV
from prompts.matcher_prompt import (
    JD_PARSE_SYSTEM_PROMPT,
    JD_PARSE_USER_TEMPLATE,
    MATCH_ANALYZE_SYSTEM_PROMPT,
    MATCH_ANALYZE_USER_TEMPLATE,
)
from rag.cv_chunker import query_cv_chunks
from schemas.jd_schema import ParsedJD
from schemas.match_schema import MatchResult

try:
    from langsmith import traceable as _traceable
    _TRACE = True
except ImportError:
    _TRACE = False

logger = logging.getLogger(__name__)

MAX_RETRIES   = 3
RETRY_DELAY   = 2
TOP_K_CHUNKS  = 3    # Số chunks lấy per skill — đủ evidence mà không quá dài
MIN_SCORE     = 0.0  # Không filter — evidence rỗng = MISSING, LLM tự kết luận


class JDMatcherAgent:
    """
    Agent 2 (v2): Match ParsedCV against JD using structured parsing + ChromaDB evidence.

    Usage:
        agent = JDMatcherAgent(api_key="GEMINI_KEY")

        # Parse JD 1 lần, cache lại
        parsed_jd = agent.parse_jd(jd_text="Full JD text here...", job_title="Senior BE")

        # Match từng candidate với JD đã parse
        result = agent.match(candidate=parsed_cv, parsed_jd=parsed_jd)

        # Hoặc batch (auto-cache JD parsing)
        results = agent.match_batch(candidates=[...], jd_text="...", job_title="...")
    """

    def __init__(
        self,
        api_key:    str,
        model:      str = "gemini-3.1-flash-lite", # gemini-3.1-flash-lite | gemini-2.5-flash-lite
    ) -> None:
        self.client     = genai.Client(api_key=api_key)
        self.model_name = model
        self._jd_cache: dict[str, ParsedJD] = {}  # cache JD parsing theo job_id

    # ── Public API ─────────────────────────────────────────────────────────

    def parse_jd(self, jd_text: str, job_title: str, job_id: str = "") -> ParsedJD:
        """
        BƯỚC 1: Parse JD raw text → ParsedJD.

        Kết quả được cache theo job_id.
        Gọi hàm này 1 lần trước batch, reuse cho tất cả candidates.

        Args:
            jd_text:   Full JD text (copy-paste từ job posting)
            job_title: Tên vị trí (fallback nếu LLM không extract được)
            job_id:    Key để cache (dùng job_id từ ChromaDB)
        """
        cache_key = job_id or job_title

        if cache_key in self._jd_cache:
            logger.info("JD parse cache hit: '%s'", cache_key)
            return self._jd_cache[cache_key]

        logger.info("Parsing JD: '%s'", job_title)

        prompt   = JD_PARSE_USER_TEMPLATE.format(jd_text=jd_text)
        raw_json = self._call_llm_with_retry(
            prompt        = prompt,
            system_prompt = JD_PARSE_SYSTEM_PROMPT,
        )

        parsed           = ParsedJD.model_validate(raw_json)
        parsed.job_title = parsed.job_title or job_title  # fallback

        self._jd_cache[cache_key] = parsed
        logger.info(
            "JD parsed: '%s' | %d required skills | %d nice-to-have",
            parsed.job_title,
            len(parsed.required_skills),
            len(parsed.nice_to_have),
        )
        return parsed

    def match(
        self,
        candidate:  ParsedCV,
        parsed_jd:  ParsedJD,
    ) -> MatchResult:
        """
        BƯỚC 2 + 3: Query evidence + Analyze match cho 1 candidate.

        Args:
            candidate: ParsedCV từ Agent 1 (cần cv_id để query ChromaDB)
            parsed_jd: ParsedJD từ parse_jd()
        """
        if _TRACE:
            from langsmith import traceable
            return traceable(
                name="JDMatcherAgent.match",
                run_type="chain",
                tags=["jd-matcher", "agent-2"],
                metadata={
                    "candidate": candidate.candidate_name,
                    "job":       parsed_jd.job_title,
                    "cv_id":     candidate.cv_id,
                },
            )(self._match_internal)(candidate, parsed_jd)
        return self._match_internal(candidate, parsed_jd)

    def match_batch(
        self,
        candidates: list[ParsedCV],
        jd_text:    str,
        job_title:  str,
        job_id:     str = "",
    ) -> list[MatchResult]:
        """
        Parse JD 1 lần, match toàn bộ batch.

        Args:
            candidates: list ParsedCV từ Agent 1
            jd_text:    Full JD text
            job_title:  Tên vị trí
            job_id:     ID của JD
        """
        # Parse JD 1 lần cho toàn batch
        parsed_jd = self.parse_jd(jd_text, job_title, job_id)

        results = []
        for i, candidate in enumerate(candidates, 1):
            logger.info(
                "Batch match %d/%d: %s",
                i, len(candidates), candidate.candidate_name,
            )
            try:
                result = self.match(candidate, parsed_jd)
            except Exception as e:
                logger.error("Match failed for %s: %s", candidate.candidate_name, e)
                result = self._empty_result(candidate.candidate_name, parsed_jd.job_title)
                result.warnings.append(f"Match failed: {e}")

            results.append(result)

            # Rate limit: 4s giữa các LLM calls (Gemini free tier)
            if i < len(candidates):
                time.sleep(4)

        return results

    # ── Private: Core pipeline ─────────────────────────────────────────────

    def _match_internal(self, candidate: ParsedCV, parsed_jd: ParsedJD) -> MatchResult:
        """Bước 2 + 3: Query ChromaDB → Analyze với LLM."""
        # Bước 2: Query evidence từ ChromaDB per skill
        cv_evidence = self._query_evidence(candidate.cv_id, parsed_jd)

        # Tính average similarity score để debug
        all_scores = [
            chunk["score"]
            for chunks in cv_evidence.values()
            for chunk in chunks
        ]
        avg_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

        # Bước 3: Gọi LLM analyze
        prompt   = self._build_match_prompt(candidate, parsed_jd, cv_evidence)
        raw_json = self._call_llm_with_retry(
            prompt        = prompt,
            system_prompt = MATCH_ANALYZE_SYSTEM_PROMPT,
        )

        result = MatchResult.model_validate(raw_json)

        # Enrich meta
        result.raw_similarity_score = avg_score

        # Low confidence nếu CV có ít chunks hoặc parse failed
        if candidate.chunk_count < 3 or candidate.parse_method == "failed":
            result.low_confidence = True
            result.warnings.append(
                f"CV has only {candidate.chunk_count} chunks — results may be incomplete"
            )

        logger.info(
            "Match: %s | matched=%d | missing=%d | score=%.2f",
            candidate.candidate_name,
            len(result.skill_gap.matched),
            len(result.skill_gap.missing_critical),
            avg_score,
        )
        return result

    # ── Private: ChromaDB evidence retrieval ──────────────────────────────

    def _query_evidence(
        self,
        cv_id:     str,
        parsed_jd: ParsedJD,
    ) -> dict[str, list[dict]]:
        """
        Query ChromaDB: với MỖI required skill → lấy top chunks từ CV.

        Đây là điểm khác biệt cốt lõi so với v1:
          - Query direction: JD skills → CV chunks (đúng)
          - Tất cả required skills đều được query, kể cả khi candidate không có
          - Evidence rỗng → match_level = MISSING trong bước LLM

        Returns:
            dict mapping: skill_text → [{"text", "section", "score", ...}]
        """
        evidence: dict[str, list[dict]] = {}

        all_skills = (
            [(s, "required")  for s in parsed_jd.required_skills] +
            [(s, "nice")      for s in parsed_jd.nice_to_have]
        )

        for skill, priority in all_skills:
            chunks = query_cv_chunks(
                query_text = skill,
                cv_id      = cv_id,
                top_k      = TOP_K_CHUNKS,
                min_score  = MIN_SCORE,
            )
            evidence[skill] = chunks
            logger.debug(
                "Evidence for '%s' [%s]: %d chunks (scores: %s)",
                skill, priority,
                len(chunks),
                [round(c["score"], 2) for c in chunks],
            )

        return evidence

    # ── Private: Prompt building ───────────────────────────────────────────

    def _build_match_prompt(
        self,
        candidate:   ParsedCV,
        parsed_jd:   ParsedJD,
        cv_evidence: dict[str, list[dict]],
    ) -> str:
        """Format prompt cho LLM analyze call."""

        # Format JD structured
        jd_dict = {
            "job_title":                  parsed_jd.job_title,
            "required_skills":            parsed_jd.required_skills,
            "nice_to_have":               parsed_jd.nice_to_have,
            "required_experience_years":  parsed_jd.required_experience_years,
            "required_experience_domain": parsed_jd.required_experience_domain,
            "required_education_level":   parsed_jd.required_education_level,
            "required_education_major":   parsed_jd.required_education_major,
            "seniority_level":            parsed_jd.seniority_level,
        }

        # Format CV evidence per skill
        evidence_lines = []

        for skill, chunks in cv_evidence.items():
            is_nice = skill in parsed_jd.nice_to_have
            tag     = "[NICE-TO-HAVE]" if is_nice else "[REQUIRED]"
            evidence_lines.append(f"\n### {tag} {skill}")

            if not chunks:
                evidence_lines.append("  (No relevant content found in CV)")
            else:
                for j, chunk in enumerate(chunks, 1):
                    section = chunk.get("section", "")
                    score   = chunk.get("score",   0)
                    text    = chunk.get("text",    "").strip()
                    # Truncate very long chunks
                    if len(text) > 400:
                        text = text[:400] + "…"
                    evidence_lines.append(
                        f"  [{j}] Section: {section} | Similarity: {score:.2f}\n"
                        f"      {text}"
                    )

        # Thêm basic info từ CV (không cần LLM vì đã có heuristic)
        cv_meta_lines = [
            f"File: {candidate.file_name}",
            f"Email: {candidate.email or 'N/A'}",
            f"Detected sections: {', '.join(candidate.sections) or 'None'}",
        ]

        return MATCH_ANALYZE_USER_TEMPLATE.format(
            parsed_jd_json   = json.dumps(jd_dict, ensure_ascii=False, indent=2),
            cv_evidence_text = "\n".join(evidence_lines),
            candidate_name   = candidate.candidate_name,
        ) + "\n\n## CV Meta\n" + "\n".join(cv_meta_lines)

    # ── Private: LLM ──────────────────────────────────────────────────────

    def _call_llm_with_retry(
        self,
        prompt:        str,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Gọi LLM với retry khi JSON invalid."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw  = self._call_llm(prompt, system_prompt)
                return self._parse_json(raw)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("LLM invalid JSON (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    prompt += "\n\nIMPORTANT: Output ONLY raw JSON. No markdown. No text."
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"LLM failed after {MAX_RETRIES} attempts: {e}") from e
        raise RuntimeError("Unexpected state")

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        response = self.client.models.generate_content(
            model    = self.model_name,
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                system_instruction = system_prompt,
                temperature        = 0.0,
                max_output_tokens  = 4096,
            ),
        )
        return response.text

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$",           "", text, flags=re.MULTILINE)
        text = text.strip()
        if not text.startswith("{"):
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON object found")
            text = text[start:]
        return json.loads(text)

    def _empty_result(self, candidate_name: str, job_title: str) -> MatchResult:
        return MatchResult(
            candidate_name = candidate_name,
            job_title      = job_title,
            low_confidence = True,
            warnings       = ["Match failed — no result available"],
        )