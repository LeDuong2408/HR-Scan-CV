"""
Agent 2: JD Matcher Agent

Nhận CandidateProfile từ CV Parser Agent (Agent 1)
→ Retrieve JD requirements từ ChromaDB bằng RAG
→ Gọi Gemini để phân tích gap chi tiết
→ Trả về MatchResult có cấu trúc cho Scorer Agent (Agent 3)

Tại sao cần RAG thay vì paste toàn bộ JD vào prompt?
  1. JD thực tế có thể rất dài (2000+ words)
  2. Chỉ retrieve top-K requirements LIÊN QUAN đến CV này
     → Context window gọn hơn, LLM focus hơn, ít hallucination hơn
  3. Rubric scoring được tách ra và luôn nhất quán
  4. 1 JD có thể match nhiều CV khác nhau mà không cần paste lại

Pipeline:
  CandidateProfile
       │
       ▼
  Build query text (skills + experience tóm tắt)
       │
       ▼
  ChromaDB retrieve top-K JD requirements     ← RAG
       │
       ▼
  Format prompt (CV profile JSON + requirements)
       │
       ▼
  Gemini → phân tích gap → raw JSON
       │
       ▼
  Parse + validate → MatchResult              ← Pydantic
       │
       ▼
  Enrich (similarity score, warnings)
       │
       ▼
  Return MatchResult → Scorer Agent
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from prompts.matcher_prompt import MATCHER_SYSTEM_PROMPT, MATCHER_USER_TEMPLATE
from rag.retriever import RetrievedChunk, search_jd_requirements
from schemas.cv_schema import CandidateProfile
from schemas.match_schema import MatchResult

logger = logging.getLogger(__name__)

MAX_RETRIES   = 3
RETRY_DELAY   = 2
DEFAULT_TOP_K = 12  # Số JD chunks retrieve mỗi lần


class JDMatcherAgent:
    """
    Agent 2: Match CV profile against Job Description using RAG + LLM.

    Usage:
        agent = JDMatcherAgent(api_key="GEMINI_KEY")
        result = agent.match(
            candidate=profile,        # CandidateProfile từ CVParserAgent
            job_id="backend-2025-01",
            job_title="Senior Backend Engineer",
        )
        print(result.skill_gap.missing_critical)
        print(result.match_summary)
    """

    def __init__(
        self,
        api_key:       str,
        model:         str = "gemini-3.1-flash-lite-preview",
        system_prompt: str = MATCHER_SYSTEM_PROMPT,
    ) -> None:
        self.client        = genai.Client(api_key=api_key)
        self.model_name    = model
        self.system_prompt = system_prompt

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def match(
        self,
        candidate:  CandidateProfile,
        job_id:     str,
        job_title:  str,
        top_k:      int = DEFAULT_TOP_K,
    ) -> MatchResult:
        """
        Full pipeline: CV profile + job_id → MatchResult.

        Args:
            candidate:  Output của CVParserAgent
            job_id:     ID của JD đã được ingest vào ChromaDB
            job_title:  Tên vị trí (dùng trong prompt và output)
            top_k:      Số JD requirement chunks retrieve từ ChromaDB

        Returns:
            MatchResult — input của Scorer Agent
        """
        logger.info(
            "Matching '%s' against job '%s' (%s)",
            candidate.full_name, job_title, job_id,
        )

        # Bước 1: Build query text từ CV
        query_text = self._build_query_text(candidate)

        # Bước 2: Retrieve JD requirements từ ChromaDB (RAG)
        chunks = search_jd_requirements(
            query_text=query_text,
            job_id=job_id,
            top_k=top_k,
        )

        if not chunks:
            logger.warning(
                "No JD chunks found for job_id='%s'. "
                "Did you run ingest_job_description() first?",
                job_id,
            )
            return self._empty_result(candidate.full_name, job_title, job_id)

        # Bước 3: Gọi LLM để phân tích gap (có retry)
        raw_json = self._call_llm_with_retry(candidate, chunks, job_title)

        # Bước 4: Validate + enrich
        result = self._build_result(raw_json, chunks, candidate)

        logger.info(
            "Match complete: %s | matched=%d | missing_critical=%d | summary='%s'",
            candidate.full_name,
            len(result.skill_gap.matched),
            len(result.skill_gap.missing_critical),
            (result.match_summary or "")[:80],
        )
        return result

    def match_batch(
        self,
        candidates: list[CandidateProfile],
        job_id:     str,
        job_title:  str,
    ) -> list[MatchResult]:
        """
        Match nhiều ứng viên với cùng 1 job.
        Sequential để tránh vượt Gemini free tier rate limit.
        """
        results = []
        for i, candidate in enumerate(candidates, 1):
            logger.info("Batch match %d/%d: %s", i, len(candidates), candidate.full_name)
            try:
                result = self.match(candidate, job_id, job_title)
            except Exception as e:
                logger.error("Match failed for %s: %s", candidate.full_name, e)
                result = self._empty_result(candidate.full_name, job_title, job_id)
                result.warnings.append(f"Match failed: {e}")

            results.append(result)

            # Rate limit: 15 req/min → ~4s giữa các calls
            if i < len(candidates):
                time.sleep(4)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Private: Query building
    # ──────────────────────────────────────────────────────────────────────────

    def _build_query_text(self, candidate: CandidateProfile) -> str:
        """
        Tóm tắt CV thành 1 đoạn text để embed và search ChromaDB.

        Tại sao không embed toàn bộ raw_text?
          - Raw CV text có thể 2000+ words → embedding bị diluted
          - Chỉ lấy skills + experience là đủ để match JD requirements
        """
        parts: list[str] = []

        # Skills (quan trọng nhất cho matching)
        if candidate.technical_skills:
            parts.append("Technical skills: " + ", ".join(candidate.technical_skills))

        # Roles gần nhất (tối đa 3)
        recent_roles = candidate.work_history[:3]
        for work in recent_roles:
            role_desc = f"{work.role} at {work.company}"
            if work.technologies:
                role_desc += " using " + ", ".join(work.technologies[:5])
            parts.append(role_desc)

        # Education
        for edu in candidate.education[:1]:  # Chỉ lấy bằng cao nhất
            parts.append(f"{edu.degree} in {edu.major or 'unknown major'}")

        # Certifications
        if candidate.certifications:
            parts.append("Certifications: " + ", ".join(candidate.certifications))

        return ". ".join(parts)

    # ──────────────────────────────────────────────────────────────────────────
    # Private: LLM call
    # ──────────────────────────────────────────────────────────────────────────

    def _call_llm_with_retry(
        self,
        candidate:  CandidateProfile,
        chunks:     list[RetrievedChunk],
        job_title:  str,
    ) -> dict[str, Any]:
        """Gọi LLM với retry logic — giống CVParserAgent."""
        prompt = self._build_prompt(candidate, chunks, job_title)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self._call_llm(prompt)
                return self._parse_json_response(raw)

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "LLM invalid JSON (attempt %d/%d): %s", attempt, MAX_RETRIES, e
                )
                if attempt < MAX_RETRIES:
                    prompt += (
                        "\n\nIMPORTANT: Previous response was not valid JSON. "
                        "Output ONLY the raw JSON object. No markdown. No text."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(
                        f"LLM failed to return valid JSON after {MAX_RETRIES} attempts"
                    ) from e

        raise RuntimeError("Unexpected state in _call_llm_with_retry")

    def _call_llm(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.0,        # Deterministic — không cần creativity
                max_output_tokens=4096,
            ),
        )
        return response.text

    def _build_prompt(
        self,
        candidate:  CandidateProfile,
        chunks:     list[RetrievedChunk],
        job_title:  str,
    ) -> str:
        """Format prompt với CV profile và JD requirements."""
        # CV profile → JSON string (chỉ lấy fields quan trọng, bỏ raw_text dài)
        cv_dict = {
            "full_name":              candidate.full_name,
            "total_experience_years": candidate.total_experience_years,
            "technical_skills":       candidate.technical_skills,
            "soft_skills":            candidate.soft_skills,
            "certifications":         candidate.certifications,
            "work_history": [
                {
                    "company":        w.company,
                    "role":           w.role,
                    "duration_months": w.duration_months,
                    "technologies":   w.technologies,
                    "achievements":   w.achievements,
                }
                for w in candidate.work_history
            ],
            "education": [
                {
                    "institution":       e.institution,
                    "degree":            e.degree,
                    "level":             e.level,
                    "graduation_year":   e.graduation_year,
                }
                for e in candidate.education
            ],
            "languages": [
                {"language": l.language, "proficiency": l.proficiency}
                for l in candidate.languages
            ],
        }

        # Format JD requirements với priority và score để LLM biết ngữ cảnh
        jd_lines = []
        for i, chunk in enumerate(chunks, 1):
            priority_tag = "[REQUIRED]" if chunk.priority == "required" else "[NICE]"
            jd_lines.append(
                f"{i}. {priority_tag} {chunk.text} (relevance: {chunk.score:.2f})"
            )

        return MATCHER_USER_TEMPLATE.format(
            cv_profile_json=json.dumps(cv_dict, ensure_ascii=False, indent=2),
            jd_requirements_text="\n".join(jd_lines),
            job_title=job_title,
        )

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """Strip markdown fences và parse JSON — giống CVParserAgent."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        if not text.startswith("{"):
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in LLM response")
            text = text[start:]

        return json.loads(text)

    # ──────────────────────────────────────────────────────────────────────────
    # Private: Build validated MatchResult
    # ──────────────────────────────────────────────────────────────────────────

    def _build_result(
        self,
        data:      dict[str, Any],
        chunks:    list[RetrievedChunk],
        candidate: CandidateProfile,
    ) -> MatchResult:
        """Validate LLM output + enrich với metadata."""
        result = MatchResult.model_validate(data)

        # Enrich: average similarity score từ top chunks
        if chunks:
            result.raw_similarity_score = round(
                sum(c.score for c in chunks) / len(chunks), 4
            )

        # Enrich: lưu chunk IDs đã dùng để audit
        result.jd_chunks_used = [c.chunk_id for c in chunks]

        # Inherit low_confidence từ CV parse nếu cần
        from schemas.cv_schema import ParseConfidence
        if candidate.confidence == ParseConfidence.LOW:
            result.low_confidence = True
            result.warnings.append(
                "CV was parsed with LOW confidence — results may be inaccurate"
            )

        return result

    def _empty_result(
        self,
        candidate_name: str,
        job_title:      str,
        job_id:         str,
    ) -> MatchResult:
        """Trả về MatchResult rỗng khi không có JD chunks — không crash batch."""
        return MatchResult(
            candidate_name=candidate_name,
            job_title=job_title,
            low_confidence=True,
            warnings=[f"No JD chunks found for job_id='{job_id}'. Run ingestor first."],
        )