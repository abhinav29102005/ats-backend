"""
API Key middleware for securing endpoints
"""
import os
import logging
from typing import Callable
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API key for protected endpoints
    
    Protected endpoints require an API key in the header:
    X-API-Key: your-api-key
    """
    
    # Public endpoints that don't require API key
    PUBLIC_ENDPOINTS = [
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc"
    ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> object:
        """Process request and validate API key if required"""
        
        # Allow public endpoints
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)
        
        # Skip preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check for API key in protected endpoints
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            logger.warning(f"Missing API key for {request.method} {request.url.path}")
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Please provide X-API-Key header."
            )
        
        # Validate API key
        if not self._validate_api_key(api_key):
            logger.warning(f"Invalid API key attempt for {request.method} {request.url.path}")
            raise HTTPException(
                status_code=403,
                detail="Invalid API key."
            )
        
        # Add API key info to request state for logging
        request.state.api_key_valid = True
        
        response = await call_next(request)
        return response
    
    @staticmethod
    def _validate_api_key(api_key: str) -> bool:
        """
        Validate the provided API key
        
        Args:
            api_key: The API key to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Get valid API keys from environment variable
        valid_keys = os.getenv("API_KEYS", "").split(",")
        valid_keys = [key.strip() for key in valid_keys if key.strip()]
        
        # Also check for a master key
        master_key = os.getenv("MASTER_API_KEY")
        
        if master_key and api_key == master_key:
            return True
        
        if api_key in valid_keys:
            return True
        
        return False


def require_api_key(endpoint: Callable) -> Callable:
    """
    Decorator to require API key for specific endpoint
    
    Usage:
        @router.post("/secure-endpoint")
        @require_api_key
        async def secure_endpoint(request: Request):
            ...
    """
    async def wrapper(request: Request, *args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Please provide X-API-Key header."
            )
        
        if not APIKeyMiddleware._validate_api_key(api_key):
            raise HTTPException(
                status_code=403,
                detail="Invalid API key."
            )
        
        return await endpoint(request, *args, **kwargs)
    
    return wrapper
