"""Helpers"""
import uuid

def generate_uuid() -> str:
    return str(uuid.uuid4())

def get_verdict(score: float) -> str:
    if score >= 80:
        return "Excellent Match ⭐"
    elif score >= 60:
        return "Good Match ✓"
    elif score >= 40:
        return "Fair Match"
    else:
        return "Needs Improvement"
