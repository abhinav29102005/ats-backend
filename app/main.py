"""Main FastAPI application"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes import router
from app.database import db
from app.core.resume_parser import load_nlp_model
from app.middleware import APIKeyMiddleware

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="ATS Resume Scoring System"
)

# CORS - UPDATED FOR FILE UPLOADS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# API Key Middleware - Add security layer
if settings.ENABLE_API_KEY_AUTH:
    app.add_middleware(APIKeyMiddleware)
    logger.info("✅ API Key authentication enabled")
else:
    logger.warning("⚠️ API Key authentication disabled")

# Include routes
app.include_router(router)

@app.on_event("startup")
async def startup():
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.VERSION}")
    try:
        settings.validate()
        logger.info("✅ Settings validated")
    except Exception as e:
        logger.error(f"❌ Settings validation failed: {e}")
    
    load_nlp_model()
    
    if db.is_connected:
        logger.info("✅ Database connected")
    else:
        logger.error("❌ Database connection failed")
    
    logger.info(f"🌐 Environment: {settings.ENVIRONMENT}")
    logger.info(f"📊 Max uploads: {settings.MAX_UPLOADS}")
    logger.info("✅ Application started successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("👋 Shutting down application")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
