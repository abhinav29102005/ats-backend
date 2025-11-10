"""
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.api.routes import router
from app.database import db
from app.core.resume_parser import load_nlp_model

# Logging configuration
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="ATS Resume Scoring System - Microsoft Learn Student Chapter @ TIET",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.ENVIRONMENT == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.VERSION}")
    
    # Validate settings
    try:
        settings.validate()
        logger.info("✅ Settings validated")
    except Exception as e:
        logger.error(f"❌ Settings validation failed: {e}")
    
    # Load NLP model
    load_nlp_model()
    
    # Check database connection
    if db.is_connected:
        logger.info("✅ Database connected")
    else:
        logger.error("❌ Database connection failed")
    
    logger.info(f"🌐 Environment: {settings.ENVIRONMENT}")
    logger.info(f"📊 Max uploads: {settings.MAX_UPLOADS}")
    logger.info("✅ Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down application")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )
