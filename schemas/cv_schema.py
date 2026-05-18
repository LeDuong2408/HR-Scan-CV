"""
Pydantic models for CV structured output.
These schemas act as the contract between CV Parser Agent
and all downstream agents (JD Matcher, Scorer).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class EducationLevel(str, Enum):
    PHD = "phd"
    MASTER = "master"
    BACHELOR = "bachelor"
    ASSOCIATE = "associate"
    HIGHSCHOOL = "highschool"
    CERTIFICATION = "certification"
    UNKNOWN = "unknown"


class EducationEntry(BaseModel):
    institution: str = Field(default="Unknown")
    degree: str = Field(default="Unknown")
    major: Optional[str] = None
    level: EducationLevel = EducationLevel.UNKNOWN
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None

    @field_validator("graduation_year")
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        if v and not (1950 <= v <= 2030):
            return None
        return v


class WorkEntry(BaseModel):
    company: str = Field(default="Unknown")
    role: str = Field(default="Unknown")
    start_year: Optional[int] = None
    end_year: Optional[int] = None  # None = current
    duration_months: Optional[int] = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class LanguageEntry(BaseModel):
    language: str
    proficiency: str = "unknown"  # e.g. "Native", "B2", "Conversational"


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None


class ParseConfidence(str, Enum):
    HIGH = "high"      # Clean PDF, all fields extracted
    MEDIUM = "medium"  # Some fields missing or inferred
    LOW = "low"        # Scan/image, OCR used, many fields unclear


class CandidateProfile(BaseModel):
    """
    Structured output of CV Parser Agent.
    All downstream agents consume this schema.
    """
    # Identity
    full_name: str = Field(default="Unknown")
    contact: ContactInfo = Field(default_factory=ContactInfo)

    # Experience
    total_experience_years: Optional[float] = None
    work_history: list[WorkEntry] = Field(default_factory=list)

    # Skills
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    # Education
    education: list[EducationEntry] = Field(default_factory=list)
    highest_education_level: EducationLevel = EducationLevel.UNKNOWN

    # Languages
    languages: list[LanguageEntry] = Field(default_factory=list)

    # Meta — used by Scorer Agent
    confidence: ParseConfidence = ParseConfidence.MEDIUM
    extraction_method: str = "native_pdf"  # native_pdf | ocr | docx
    missing_fields: list[str] = Field(default_factory=list)
    raw_text_length: int = 0
    parse_warnings: list[str] = Field(default_factory=list)