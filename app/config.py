"""Configuration"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME = "MLSC Perfect CV Match 2025"
    VERSION = "2.0.0"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    PORT = int(os.getenv("PORT", 8000))
    
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    
    MAX_UPLOADS = int(os.getenv("MAX_UPLOADS", 10))
    RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 30))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 20))
    
    SCORING_WEIGHTS = {
        'skills_match': 35,
        'experience': 25,
        'keyword_relevance': 15,
        'education': 10,
        'resume_quality': 10,
        'projects': 5
    }
    
    # ADDED THE LIVE FRONTEND EXACT ORIGIN HERE
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "https://perfect-cv.mlsctiet.com", "*"]
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    ENABLE_API_KEY_AUTH = os.getenv("ENABLE_API_KEY_AUTH", "true").lower() == "true"
    MASTER_API_KEY = os.getenv("MASTER_API_KEY", "")
    API_KEYS = [key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()]
    
    @property
    def MAX_FILE_SIZE_BYTES(self):
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

settings = Settings()
