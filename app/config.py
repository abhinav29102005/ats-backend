"""
Configuration management for the application
"""
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    """Application settings"""
    
    # Application
    APP_NAME: str = "MLSC Perfect CV Match 2025"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # Competition Settings
    MAX_UPLOADS: int = int(os.getenv("MAX_UPLOADS", 5))
    RATE_LIMIT_SECONDS: int = int(os.getenv("RATE_LIMIT_SECONDS", 30))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 20))
    
    # Scoring Weights
    SCORING_WEIGHTS = {
        'skills_match': 35,
        'experience': 25,
        'keyword_relevance': 15,
        'education': 10,
        'resume_quality': 10,
        'projects': 5
    }
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://mlsc-tiet.vercel.app"
    ]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def validate(self) -> bool:
        """Validate critical settings"""
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        return True

settings = Settings()
