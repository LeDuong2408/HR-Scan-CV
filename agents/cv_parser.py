"""
Agent 1: CV Parser Agent

Responsibilities:
  - Extract raw text from PDF / DOCX (with OCR fallback)
  - Call Gemini Flash to parse raw text into structured CandidateProfile
  - Validate output with Pydantic schema
  - Retry up to MAX_RETRIES if LLM returns invalid JSON
  - Flag low-confidence extractions for HR review

Why Gemini Flash (free tier) over Claude here?
  - We call this agent once per CV in a batch of 50-200 CVs
  - Gemini Flash free tier: 15 req/min, 1500 req/day — sufficient for demo
  - If you have Claude credits, swap _call_llm() — interface is identical
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

from dotenv import load_dotenv
load_dotenv()

from prompts.parser_prompt import PARSER_SYSTEM_PROMPT, PARSER_USER_TEMPLATE
from schemas.cv_schema import (
    CandidateProfile,
    ParseConfidence,
)
from tools.docx_extractor import extract_docx_text
from tools.pdf_extractor import extract_pdf_text

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


class CVParserAgent:
    """
    Agent 1: Parse a CV file into a validated CandidateProfile.

    Usage:
        agent = CVParserAgent(api_key="YOUR_GEMINI_KEY")
        profile = agent.parse("path/to/cv.pdf")
        print(profile.full_name)
        print(profile.total_experience_years)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        system_prompt: str = PARSER_SYSTEM_PROMPT,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str) -> CandidateProfile:
        """
        Full pipeline: file → raw text → LLM → validated CandidateProfile.

        Raises:
            FileNotFoundError: if file does not exist.
            ValueError: if file format is unsupported.
            RuntimeError: if LLM fails after MAX_RETRIES.
        """
        path = Path(file_path)
        logger.info("Parsing CV: %s", path.name)

        # Step 1 — Extract raw text
        raw_text, extraction_method = self._extract_text(path)
        logger.debug("Extracted %d chars via %s", len(raw_text), extraction_method)

        # Step 2 — Call LLM with retry
        llm_json = self._call_llm_with_retry(raw_text)

        # Step 3 — Validate and enrich with meta fields
        profile = self._build_profile(
            llm_json,
            extraction_method=extraction_method,
            raw_text_length=len(raw_text),
        )

        logger.info(
            "Parsed: %s | %.1f yrs exp | %d skills | confidence=%s",
            profile.full_name,
            profile.total_experience_years or 0,
            len(profile.technical_skills),
            profile.confidence,
        )
        return profile

    def parse_batch(self, file_paths: list[str]) -> list[CandidateProfile]:
        """
        Parse multiple CVs sequentially (respects Gemini free tier rate limit).
        For parallel processing, use async version or LangGraph batch node.
        """
        results: list[CandidateProfile] = []
        for i, fp in enumerate(file_paths, 1):
            logger.info("Batch progress: %d/%d — %s", i, len(file_paths), Path(fp).name)
            try:
                profile = self.parse(fp)
                results.append(profile)
            except Exception as e:
                logger.error("Failed to parse %s: %s — skipping", fp, e)
                # Return a minimal profile so downstream agents don't crash
                results.append(self._fallback_profile(fp, str(e)))

            # Respect Gemini free tier: 15 req/min → ~4s between calls
            if i < len(file_paths):
                time.sleep(4)

        return results

    # ------------------------------------------------------------------
    # Private: Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, path: Path) -> tuple[str, str]:
        """Returns (raw_text, extraction_method)."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            result = extract_pdf_text(str(path))
            return result.text, result.method  # "native" | "ocr"

        elif suffix in {".docx", ".doc"}:
            result = extract_docx_text(str(path))
            return result.text, result.method  # "docx" | "doc_converted"

        else:
            raise ValueError(
                f"Unsupported format: {suffix}. Supported: .pdf, .docx, .doc"
            )

    # ------------------------------------------------------------------
    # Private: LLM call
    # ------------------------------------------------------------------

    def _call_llm_with_retry(self, raw_text: str) -> dict[str, Any]:
        """Call Gemini and retry on JSON parse failure."""
        user_message = PARSER_USER_TEMPLATE.format(raw_text=raw_text)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_json = self._call_llm(user_message)
                return self._parse_json_response(raw_json)

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "LLM returned invalid JSON (attempt %d/%d): %s",
                    attempt, MAX_RETRIES, e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
                    # Add explicit correction hint on retry
                    user_message = (
                        user_message
                        + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                        "Output ONLY the raw JSON object. No markdown. No explanation."
                    )
                else:
                    raise RuntimeError(
                        f"LLM failed to return valid JSON after {MAX_RETRIES} attempts: {e}"
                    ) from e

        # Should never reach here
        raise RuntimeError("Unexpected state in _call_llm_with_retry")

    def _call_llm(self, user_message: str) -> str:
        """Single LLM call — returns raw text response."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )
        return response.text

    def _parse_json_response(self, raw_response: str) -> dict[str, Any]:
        """
        Strip any markdown fences the LLM might add despite instructions,
        then parse JSON.
        """
        text = raw_response.strip()

        # Remove ```json ... ``` fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Ensure it starts with {
        if not text.startswith("{"):
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in LLM response")
            text = text[start:]

        return json.loads(text)

    # ------------------------------------------------------------------
    # Private: Build validated CandidateProfile
    # ------------------------------------------------------------------

    def _build_profile(
        self,
        data: dict[str, Any],
        extraction_method: str,
        raw_text_length: int,
    ) -> CandidateProfile:
        """
        Validate LLM output against Pydantic schema.
        Determine confidence level based on extraction quality.
        """
        # Pydantic will coerce types and fill defaults
        profile = CandidateProfile.model_validate(data)

        # Enrich meta fields
        profile.extraction_method = extraction_method
        profile.raw_text_length = raw_text_length

        # Determine confidence
        profile.confidence = self._assess_confidence(profile, extraction_method)

        return profile

    def _assess_confidence(
        self,
        profile: CandidateProfile,
        extraction_method: str,
    ) -> ParseConfidence:
        """
        Heuristic confidence based on what we actually extracted.
        This feeds into the Scorer Agent — low confidence → HR should verify.
        """
        score = 100

        if extraction_method == "ocr":
            score -= 20  # OCR is less reliable

        if not profile.full_name or profile.full_name == "Unknown":
            score -= 20

        if not profile.contact.email:
            score -= 10

        if not profile.work_history:
            score -= 20

        if not profile.technical_skills:
            score -= 15

        if len(profile.missing_fields) > 3:
            score -= 10

        if score >= 75:
            return ParseConfidence.HIGH
        elif score >= 45:
            return ParseConfidence.MEDIUM
        else:
            return ParseConfidence.LOW

    def _fallback_profile(self, file_path: str, error: str) -> CandidateProfile:
        """Minimal profile returned when parsing fails — never crash the batch."""
        return CandidateProfile(
            full_name=f"[PARSE ERROR] {Path(file_path).name}",
            confidence=ParseConfidence.LOW,
            missing_fields=["all"],
            parse_warnings=[f"Parsing failed: {error}"],
            extraction_method="failed",
            raw_text_length=0,
        )

if __name__ == '__main__':
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    agent = CVParserAgent(api_key=GOOGLE_API_KEY)
    profile = agent.parse("C:/Users/DuongLe(EXT)/WorkSapce/hr-cv-scanner/data/CV_18_Bui_Thi_Zung.pdf")
    print(profile.full_name, profile.total_experience_years, profile.education)