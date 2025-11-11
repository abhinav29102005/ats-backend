"""Pydantic models"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict

class ParticipantRegistration(BaseModel):
    name: str = Field(..., min_length=3, max_length=200)
    email: EmailStr
    mobile: str = Field(..., min_length=10, max_length=20)

class ParticipantResponse(BaseModel):
    id: str
    name: str
    email: str
    mobile: str
    upload_count: int
    message: Optional[str] = None

class ScoreBreakdown(BaseModel):
    skills_match: float
    experience: float
    keyword_relevance: float
    education: float
    resume_quality: float
    projects: float

class ScoreResponse(BaseModel):
    score: float
    breakdown: ScoreBreakdown
    skills: List[str]
    matched_skills: List[str]
    experience_years: float
    feedback: List[str]
    penalties: List[str]
    plagiarism_score: float
    keyword_similarity: float
    upload_count: int
    verdict: str

class HealthResponse(BaseModel):
    status: str
    competition: str
    organization: str
    version: str
    nlp_loaded: bool
    database_connected: bool
    scoring_weights: Dict[str, int]
