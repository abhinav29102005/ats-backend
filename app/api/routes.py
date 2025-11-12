"""
API route handlers - Updated with fixed job description
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
from app.utils.validators import sanitize_input, validate_mobile
from app.utils.helpers import generate_uuid, get_verdict

logger = logging.getLogger(__name__)
router = APIRouter()

# Rate limiting storage
last_submissions: Dict[str, float] = {}

# Fixed Job Description
FIXED_JOB_DESCRIPTION = """
Job Description — Software Engineer (Cloud & Automation Systems)
Company: CloudAlly Systems Pvt. Ltd.
Location: Bengaluru, India (Hybrid)
Experience Level: Entry–Mid (0–2 years)
CTC: ₹8–12 LPA

About CloudAlly Systems
CloudAlly Systems is a fast-growing technology company specializing in intelligent automation and cloud-based analytics. We build scalable enterprise platforms that integrate AI, data pipelines, and real-time automation to streamline operations for global clients. Our engineering culture emphasizes innovation, autonomy, and building reliable systems at scale.

Role Overview
We are looking for a passionate Software Engineer to join our Cloud and Automation division. The ideal candidate will have a strong understanding of backend development, distributed systems, and an interest in automating business workflows through data-driven engineering. The role involves designing APIs, deploying containerized applications, and contributing to ML-powered feature pipelines.

Responsibilities
* Design, develop, and deploy scalable microservices using Python (FastAPI / Flask) or Node.js.
* Implement, optimize, and maintain RESTful APIs and backend logic for automation workflows.
* Work with cloud platforms (AWS / Azure) to manage CI/CD pipelines, monitoring tools, and deployment infrastructure.
* Integrate machine learning models into production systems (basic exposure preferred, not mandatory).
* Collaborate with frontend developers to deliver complete, maintainable web applications using React or Next.js.
* Ensure system reliability through testing, containerization (Docker), and observability setups (Grafana / Prometheus).
* Contribute to code reviews, architecture discussions, and documentation best practices.

Requirements
* Strong proficiency in Python, Node.js, or TypeScript.
* Experience with FastAPI / Flask, PostgreSQL, and REST API design principles.
* Knowledge of Docker, Git, and CI/CD pipelines (GitHub Actions / Jenkins).
* Familiarity with AWS (ECS, Lambda, S3) or Microsoft Azure deployment workflows.
* Understanding of microservice architecture, data-driven systems, and asynchronous programming.
* Problem-solving mindset with good debugging and performance optimization skills.

Preferred Skills
* Exposure to machine learning pipelines, MLOps, or basic AI model integration.
* Knowledge of React, Next.js, or other frontend frameworks.
* Prior experience in developing or contributing to open-source projects.
* Strong documentation habits and familiarity with Agile/Scrum practices.

What You'll Gain
* Work on real-time data systems handling large-scale automation workloads.
* Collaborate with an interdisciplinary team of backend, cloud, and AI engineers.
* Opportunity to learn deployment automation, cloud infrastructure, and production-scale systems.
* Supportive environment with mentorship, upskilling sessions, and quarterly project showcases.
"""

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="active",
        competition=settings.APP_NAME,
        organization="Microsoft Learn Student Chapter @ TIET",
        version=settings.VERSION,
        nlp_loaded=True,
        database_connected=db.is_connected,
        scoring_weights=settings.SCORING_WEIGHTS
    )

@router.get("/health")
async def health():
    """Fast health check without heavy operations"""
    return {"status": "ok"}

@router.get("/api/job-description")
async def get_job_description():
    """Get the fixed job description"""
    return {"job_description": FIXED_JOB_DESCRIPTION}

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
    resume: UploadFile = File(...)
):
    """Submit resume for ATS scoring - Uses fixed job description"""
    try:
        # Validate inputs
        if not participant_id:
            raise HTTPException(status_code=400, detail="Participant ID required")
        
        # Validate file
        if not resume.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        if resume.size and resume.size > settings.MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB")
        
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
        
        # Read and validate PDF
        content = await resume.read()
        is_valid, msg = validate_pdf(content, resume.filename, settings.MAX_FILE_SIZE_BYTES)
        if not is_valid:
            raise HTTPException(status_code=400, detail=msg)
        
        # Extract text
        text = extract_text_from_pdf(content)
        
        if len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Resume text too short")
        
        # Parse resume
        parsed = parse_resume(text)
        
        # Calculate score using fixed JD
        result = calculate_ats_score(text, FIXED_JOB_DESCRIPTION, parsed)
        
        # Save to database
        db.save_application({
            'participant_id': participant_id,
            'score': result['score'],
            'skills_count': len(parsed['skills']),
            'experience_years': parsed['experience_years'],
            'matched_skills_count': len(result['matched_skills']),
            'keyword_similarity': result['keyword_similarity'],
            'resume_quality_score': result['breakdown']['resume_quality']
        })
        
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
        
        # Handle empty dataframe
        if scores_df is None or scores_df.empty:
            return {
                "scores": [],
                "best_score": 0,
                "average_score": 0,
                "total_submissions": 0
            }
        
        # Convert to dict
        scores_list = scores_df.to_dict('records')
        
        # Format dates
        for score in scores_list:
            if 'created_at' in score and score['created_at']:
                score['created_at'] = str(score['created_at'])
        
        return {
            "scores": scores_list,
            "best_score": float(scores_df['score'].max()),
            "average_score": float(scores_df['score'].mean()),
            "total_submissions": len(scores_list)
        }
        
    except Exception as e:
        logger.error(f"Get scores error: {e}")
        # Return empty instead of raising error
        return {
            "scores": [],
            "best_score": 0,
            "average_score": 0,
            "total_submissions": 0
        }

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