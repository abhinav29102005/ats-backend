"""
Validation utilities
"""
import re

def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    text = text.strip()[:max_length]
    # Remove potentially dangerous characters
    return re.sub(r'[<>"\';]', '', text)

def validate_email(email: str) -> bool:
    """Validate email format"""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip().lower()) is not None

def validate_mobile(mobile: str) -> bool:
    """Validate mobile number"""
    if not mobile:
        return False
    # Remove spaces, dashes, plus, parentheses
    mobile_clean = re.sub(r'[\s\-\+()]', '', mobile)
    # Check if it's at least 10 digits and contains only digits
    return len(mobile_clean) >= 10 and mobile_clean.isdigit()

def validate_job_description(jd: str) -> bool:
    """Validate job description"""
    if not jd:
        return False
    # Must be at least 50 characters
    return len(jd.strip()) >= 50

def validate_file_size(file_size: int, max_size_mb: int = 20) -> bool:
    """Validate file size"""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes

def validate_pdf_file(filename: str) -> bool:
    """Validate PDF file extension"""
    if not filename:
        return False
    return filename.lower().endswith('.pdf')