"""
Database schemas
"""
from typing import TypedDict, Optional, List
from datetime import datetime

class ParticipantDB(TypedDict):
    """Participant table schema"""
    id: str
    name: str
    email: str
    mobile: str
    created_at: datetime
    updated_at: datetime

class ApplicationDB(TypedDict):
    """Application table schema"""
    id: str
    participant_id: str
    score: float
    skills_count: int
    experience_years: float
    matched_skills_count: int
    plagiarism_score: float
    keyword_similarity: float
    resume_quality_score: float
    created_at: datetime

class ResumeCorpusDB(TypedDict):
    """Resume corpus table schema"""
    id: str
    participant_id: str
    resume_text: str
    created_at: datetime

class ParsedResume(TypedDict):
    """Parsed resume structure"""
    skills: List[str]
    experience_years: float
    projects_section: str
    education_section: str
    experience_section: str
