"""
PDF text extraction utilities
"""
import pymupdf
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extract text from PDF bytes
    
    Args:
        pdf_content: PDF file content as bytes
        
    Returns:
        Extracted text as string
        
    Raises:
        Exception: If PDF extraction fails
    """
    try:
        doc = pymupdf.open(stream=pdf_content, filetype="pdf")
        text = "".join([page.get_text() + "\n" for page in doc])
        doc.close()
        
        if not text.strip():
            raise Exception("PDF appears to be empty or contains only images")
        
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise Exception(f"PDF extraction failed: {str(e)}")

def validate_pdf(content: bytes, filename: str, max_size_bytes: int) -> tuple[bool, str]:
    """
    Validate PDF file
    
    Args:
        content: File content as bytes
        filename: Name of the file
        max_size_bytes: Maximum allowed file size
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filename.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    if len(content) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        return False, f"File size exceeds {max_mb}MB limit"
    
    return True, "Valid"
