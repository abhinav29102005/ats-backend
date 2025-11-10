"""
Input validation utilities
"""
import re
from typing import Tuple

def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    text = text.strip()[:max_length]
    text = re.sub(r'[<>"\';]', '', text)
    return text

def validate_email(email: str) -> bool:
    """Validate email format"""
    if not email:
        return False
    email = email.strip().lower()
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_mobile(mobile: str) -> bool:
    """Validate mobile number"""
    if not mobile:
        return False
    mobile_clean = re.sub(r'[\s\-\+()]', '', mobile)
    return len(mobile_clean) >= 10 and mobile_clean.isdigit()

def validate_name(name: str) -> Tuple[bool, str]:
    """Validate participant name"""
    if not name or len(name) < 3:
        return False, "Name must be at least 3 characters"
    if len(name) > 200:
        return False, "Name too long"
    return True, "Valid"

def validate_job_description(jd: str) -> Tuple[bool, str]:
    """Validate job description"""
    if not jd or len(jd.strip()) < 50:
        return False, "Job description must be at least 50 characters"
    return True, "Valid"
