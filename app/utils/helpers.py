"""
Helper functions
"""
from typing import Dict
import uuid

def generate_uuid() -> str:
    """Generate unique UUID"""
    return str(uuid.uuid4())

def get_verdict(score: float) -> str:
    """Determine verdict based on score"""
    if score >= 80:
        return "Excellent Match ⭐"
    elif score >= 60:
        return "Good Match ✓"
    elif score >= 40:
        return "Fair Match"
    else:
        return "Needs Improvement"

def format_datetime(dt) -> str:
    """Format datetime for API response"""
    return str(dt) if dt else None
