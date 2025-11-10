"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict
from datetime import datetime

class ParticipantRegistration(BaseModel):
    """Participant registration request"""
    name: str = Field(..., min_length=3, max_length=200)
    email: EmailStr
    mobile: str = Field(..., min_length=10, max_length=20)

class ParticipantResponse(BaseModel):
    """Participant registration response"""
    id: str
    name: str
    email: str
    mobile: str
    upload_count: int
    message: Optional[str] = None

class ScoreBreakdown(BaseModel):
    """Score breakdown by category"""
    skills_match: float
    experience: float
    keyword_relevance: float
    education: float
    resume_quality: float
    projects: float

class ScoreResponse(BaseModel):
    """Resume scoring response"""
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

class LeaderboardEntry(BaseModel):
    """Leaderboard entry"""
    rank: int
    email: str
    name: str
    score: float
    skills_count: int
    experience_years: float
    created_at: Optional[str] = None

class CompetitionStats(BaseModel):
    """Competition statistics"""
    total_participants: int
    total_submissions: int
    avg_score: float
    median_score: float
    top_score: float
    high_scorers: int
    score_distribution: List[Dict[str, int]]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    competition: str
    organization: str
    version: str
    nlp_loaded: bool
    database_connected: bool
    scoring_weights: Dict[str, int]
