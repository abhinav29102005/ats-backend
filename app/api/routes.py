"""
API route handlers
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
import time
import logging
from typing import Dict

from app.models import (
    ParticipantRegistration, ParticipantResponse, ScoreResponse,
    HealthResponse, ScoreBreakdown
)
from app.config import settings
from app.database import db
from app.core.pdf_parser import extract_text_from_pdf, validate_pdf
from app.core.resume_parser import parse_resume
from app.core.scoring_engine import calculate_ats_score
from app.core.plagiarism_checker import check_plagiarism, calculate_plagiarism_penalty
from app.utils.validators import sanitize_input, validate_email, validate_mobile, validate_job_description
from app.utils.helpers import generate_uuid, get_verdict

logger = logging.getLogger(__name__)
router = APIRouter()

# Rate limiting storage
last_submissions: Dict[str, float] = {}

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="active",
        competition=settings.APP_NAME,
        organization="Microsoft Learn Student Chapter @ TIET",
        version=settings.VERSION,
        nlp_loaded=True,  # Updated from resume_parser
        database_connected=db.is_connected,
        scoring_weights=settings.SCORING_WEIGHTS
    )

@router.post("/api/register", response_model=ParticipantResponse)
async def register_participant(participant: ParticipantRegistration):
    """Register new participant"""
    
    # Validate
    if not validate_mobile(participant.mobile):
        raise HTTPException(status_code=400, detail="Invalid mobile number")
    
    try:
        # Check existing
        existing = db.get_participant_by_email(participant.email)
        
        if existing:
            upload_count = db.get_upload_count(existing['id'])
            return ParticipantResponse(
                id=existing['id'],
                name=existing['name'],
                email=existing['email'],
                mobile=existing['mobile'],
                upload_count=upload_count,
                message="Welcome back! Already registered."
            )
        
        # Register new
        participant_id = generate_uuid()
        data = {
            'id': participant_id,
            'name': sanitize_input(participant.name, 200),
            'email': sanitize_input(participant.email, 200).lower(),
            'mobile': sanitize_input(participant.mobile, 20)
        }
        
        db.register_participant(data)
        
        return ParticipantResponse(
            id=participant_id,
            name=participant.name,
            email=participant.email,
            mobile=participant.mobile,
            upload_count=0,
            message="Registration successful!"
        )
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/submit", response_model=ScoreResponse)
async def submit_resume(
    participant_id: str = Form(...),
    job_description: str = Form(...),
    jd_education: str = Form(""),
    resume: UploadFile = File(...)
):
    """Submit resume for ATS scoring"""
    
    # Check upload limit
    upload_count = db.get_upload_count(participant_id)
    if upload_count >= settings.MAX_UPLOADS:
        raise HTTPException(
            status_code=400,
            detail=f"Upload limit of {settings.MAX_UPLOADS} reached"
        )
    
    # Rate limiting
    current_time = time.time()
    if participant_id in last_submissions:
        time_since_last = current_time - last_submissions[participant_id]
        if time_since_last < settings.RATE_LIMIT_SECONDS:
            wait_time = int(settings.RATE_LIMIT_SECONDS - time_since_last)
            raise HTTPException(
                status_code=429,
                detail=f"Wait {wait_time} seconds before next submission"
            )
    
    # Validate job description
    is_valid, msg = validate_job_description(job_description)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)
    
    # Read and validate PDF
    content = await resume.read()
    is_valid, msg = validate_pdf(content, resume.filename, settings.MAX_FILE_SIZE_BYTES)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)
    
    try:
        # Extract text
        text = extract_text_from_pdf(content)
        
        if len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Resume text too short")
        
        # Parse resume
        parsed = parse_resume(text)
        
        # Get reference corpus
        reference_corpus = db.get_reference_corpus()
        
        # Check plagiarism
        plagiarism_score, _ = check_plagiarism(text, reference_corpus)
        
        # Calculate score
        result = calculate_ats_score(text, job_description, parsed, jd_education)
        
        # Apply plagiarism penalty
        penalty, penalty_msg = calculate_plagiarism_penalty(plagiarism_score)
        if penalty > 0:
            result['penalties'].append(penalty_msg)
            result['score'] -= penalty
            result['score'] = max(0, result['score'])
        
        # Save to database
        db.save_application({
            'participant_id': participant_id,
            'score': result['score'],
            'skills_count': len(parsed['skills']),
            'experience_years': parsed['experience_years'],
            'matched_skills_count': len(result['matched_skills']),
            'plagiarism_score': plagiarism_score,
            'keyword_similarity': result['keyword_similarity'],
            'resume_quality_score': result['breakdown']['resume_quality']
        })
        
        # Save to corpus
        db.save_to_corpus(participant_id, text)
        
        # Update rate limiting
        last_submissions[participant_id] = current_time
        
        return ScoreResponse(
            score=result['score'],
            breakdown=ScoreBreakdown(**result['breakdown']),
            skills=parsed['skills'],
            matched_skills=result['matched_skills'],
            experience_years=parsed['experience_years'],
            feedback=result['feedback'],
            penalties=result['penalties'],
            plagiarism_score=plagiarism_score,
            keyword_similarity=result['keyword_similarity'],
            upload_count=upload_count + 1,
            verdict=get_verdict(result['score'])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/participant/{participant_id}/scores")
async def get_participant_scores(participant_id: str):
    """Get all scores for participant"""
    try:
        scores_df = db.get_participant_scores(participant_id)
        
        if scores_df.empty:
            return {
                "scores": [],
                "best_score": None,
                "average_score": None,
                "total_submissions": 0
            }
        
        scores_list = scores_df.to_dict('records')
        for score in scores_list:
            if 'created_at' in score:
                score['created_at'] = str(score['created_at'])
        
        return {
            "scores": scores_list,
            "best_score": float(scores_df['score'].max()),
            "average_score": float(scores_df['score'].mean()),
            "total_submissions": len(scores_list)
        }
    except Exception as e:
        logger.error(f"Get scores error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/participant/{participant_id}/upload-count")
async def get_upload_count(participant_id: str):
    """Get upload count"""
    try:
        count = db.get_upload_count(participant_id)
        return {
            "upload_count": count,
            "max_uploads": settings.MAX_UPLOADS,
            "remaining": settings.MAX_UPLOADS - count
        }
    except Exception as e:
        logger.error(f"Get upload count error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/leaderboard")
async def get_leaderboard():
    """Get top 10 leaderboard"""
    try:
        data = db.get_leaderboard(limit=10)
        return {"leaderboard": data}
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/stats")
async def get_stats():
    """Get competition statistics"""
    try:
        data = db.get_statistics()
        
        if not data:
            return {
                "total_participants": 0,
                "total_submissions": 0,
                "avg_score": 0,
                "median_score": 0,
                "top_score": 0,
                "high_scorers": 0,
                "score_distribution": []
            }
        
        return data
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
