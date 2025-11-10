# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
import time
import pymupdf
import spacy
import re
from datetime import datetime
import pandas as pd
from supabase import create_client, Client
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import os
from dotenv import load_dotenv
from dateutil.parser import parse as date_parse

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="MLSC Perfect CV Match 2025",
    description="ATS Resume Scoring System - Microsoft Learn Student Chapter @ TIET",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
nlp = None
supabase: Client = None
last_submissions = {}

# Configuration from environment
MAX_UPLOADS = int(os.getenv("MAX_UPLOADS", 5))
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 30))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 20))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Scoring weights (total = 100 points)
SCORING_WEIGHTS = {
    'skills_match': 35,
    'experience': 25,
    'keyword_relevance': 15,
    'education': 10,
    'resume_quality': 10,
    'projects': 5
}

# ============================================================================
# STARTUP AND INITIALIZATION
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize NLP model and Supabase connection"""
    global nlp, supabase
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ SpaCy model loaded successfully")
    except OSError:
        logger.error("❌ SpaCy model not found. Install: python -m spacy download en_core_web_sm")
    except Exception as e:
        logger.error(f"❌ Error loading spaCy: {e}")
    
    # Initialize Supabase
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        
        supabase = create_client(url, key)
        
        # Test connection
        supabase.table('participants').select("count", count='exact').limit(1).execute()
        logger.info("✅ Supabase connected successfully")
        
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        supabase = None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ParticipantRegistration(BaseModel):
    name: str = Field(..., min_length=3, max_length=200)
    email: EmailStr
    mobile: str = Field(..., min_length=10, max_length=20)

class ParticipantResponse(BaseModel):
    id: str
    name: str
    email: str
    mobile: str
    upload_count: int
    message: Optional[str] = None

class ScoreResponse(BaseModel):
    score: float
    breakdown: Dict[str, float]
    skills: List[str]
    matched_skills: List[str]
    experience_years: float
    feedback: List[str]
    penalties: List[str]
    plagiarism_score: float
    keyword_similarity: float
    upload_count: int
    verdict: str

class LeaderboardEntry(BaseModel):
    rank: int
    email: str
    name: str
    score: float
    skills_count: int
    experience_years: float

# ============================================================================
# HELPER FUNCTIONS - TEXT EXTRACTION
# ============================================================================

def validate_pdf_file(content: bytes, filename: str) -> tuple[bool, str]:
    """Validate PDF file"""
    if not filename.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    if len(content) > MAX_FILE_SIZE_BYTES:
        return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
    
    return True, "Valid"

def extract_pdf_text(pdf_content: bytes) -> str:
    """Extract text from PDF"""
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

# ============================================================================
# HELPER FUNCTIONS - RESUME PARSING
# ============================================================================

def extract_section_advanced(text: str, section_type: str) -> str:
    """Enhanced section extraction with multiple header variations"""
    
    section_headers = {
        'experience': [
            r'(?:work\s+)?experience', r'employment\s+history', 
            r'professional\s+experience', r'career\s+history',
            r'work\s+history', r'relevant\s+experience'
        ],
        'education': [
            r'education(?:al\s+background)?', r'academic\s+(?:background|qualifications?)',
            r'qualifications?', r'degrees?'
        ],
        'skills': [
            r'(?:technical\s+)?skills?', r'core\s+competencies', 
            r'expertise', r'proficiencies?', r'technologies?'
        ],
        'projects': [
            r'projects?', r'portfolio', r'key\s+projects?',
            r'selected\s+projects?'
        ]
    }
    
    lines = text.split('\n')
    section_text = []
    capturing = False
    
    headers = section_headers.get(section_type.lower(), [])
    
    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        
        # Check if this line is a section header
        is_section_start = any(re.match(f'^{h}s?:?$', line_stripped) for h in headers)
        
        if is_section_start:
            capturing = True
            continue
        
        # Detect next section (stop capturing)
        if capturing:
            all_headers = [h for headers_list in section_headers.values() for h in headers_list]
            is_new_section = any(re.match(f'^{h}s?:?$', line_stripped) for h in all_headers)
            
            if is_new_section:
                break
            
            if line.strip():
                section_text.append(line)
    
    return '\n'.join(section_text)

def parse_skills_advanced(text: str) -> List[str]:
    """Enhanced skills detection with word boundaries"""
    
    skills_patterns = {
        'Python': r'\bpython\b',
        'Java': r'\bjava\b(?!script)',
        'JavaScript': r'\b(javascript|js)\b',
        'TypeScript': r'\btypescript\b',
        'C++': r'\bc\+\+\b',
        'C#': r'\bc#\b',
        'Ruby': r'\bruby\b',
        'PHP': r'\bphp\b',
        'Swift': r'\bswift\b',
        'Kotlin': r'\bkotlin\b',
        'Go': r'\b(golang|go)\b',
        'Rust': r'\brust\b',
        'SQL': r'\bsql\b',
        'React': r'\breact(\.js)?\b',
        'Angular': r'\bangular(\.js)?\b',
        'Vue.js': r'\b(vue|vuejs|vue\.js)\b',
        'Node.js': r'\bnode(\.js)?\b',
        'Django': r'\bdjango\b',
        'Flask': r'\bflask\b',
        'Spring': r'\bspring\b',
        'Express': r'\bexpress(\.js)?\b',
        'FastAPI': r'\bfastapi\b',
        'Docker': r'\bdocker\b',
        'Kubernetes': r'\b(kubernetes|k8s)\b',
        'AWS': r'\baws\b',
        'Azure': r'\bazure\b',
        'GCP': r'\b(gcp|google cloud)\b',
        'Git': r'\bgit\b',
        'CI/CD': r'\b(ci/cd|cicd)\b',
        'Jenkins': r'\bjenkins\b',
        'MongoDB': r'\bmongodb\b',
        'PostgreSQL': r'\b(postgresql|postgres)\b',
        'MySQL': r'\bmysql\b',
        'Redis': r'\bredis\b',
        'TensorFlow': r'\btensorflow\b',
        'PyTorch': r'\bpytorch\b',
        'Pandas': r'\bpandas\b',
        'NumPy': r'\bnumpy\b',
        'Scikit-learn': r'\b(scikit-learn|sklearn)\b',
        'Machine Learning': r'\b(machine\s+learning|ml)\b',
        'Data Science': r'\bdata\s+science\b',
        'HTML': r'\bhtml5?\b',
        'CSS': r'\bcss3?\b',
        'REST API': r'\b(rest|restful)\s+api\b',
        'GraphQL': r'\bgraphql\b',
        'Microservices': r'\bmicroservices\b',
        'Agile': r'\bagile\b',
        'Scrum': r'\bscrum\b',
    }
    
    text_lower = text.lower()
    detected_skills = []
    seen_skills = set()
    
    for skill_name, pattern in skills_patterns.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            if skill_name.lower() not in seen_skills:
                detected_skills.append(skill_name)
                seen_skills.add(skill_name.lower())
    
    return detected_skills

def extract_experience_enhanced(text: str) -> float:
    """Advanced experience extraction with date intelligence"""
    
    total_months = 0
    text_lower = text.lower()
    current_year = datetime.now().year
    
    # Pattern 1: Direct years mention
    direct_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
    ]
    
    max_direct_years = 0
    for pattern in direct_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            years = [int(m) for m in matches if 0 < int(m) <= 50]
            if years:
                max_direct_years = max(years)
    
    if max_direct_years > 0:
        return float(max_direct_years)
    
    # Pattern 2: Date ranges with month names
    date_range_pattern = r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*[-–—to]\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|present|current)'
    
    ranges = re.findall(date_range_pattern, text_lower, re.IGNORECASE)
    
    for start_str, end_str in ranges:
        try:
            start_date = date_parse(start_str, fuzzy=True)
            
            if 'present' in end_str or 'current' in end_str:
                end_date = datetime.now()
            else:
                end_date = date_parse(end_str, fuzzy=True)
            
            months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            
            if 0 < months_diff <= 600:
                total_months += months_diff
        except:
            continue
    
    # Pattern 3: Year-only ranges (2020-2023)
    year_ranges = re.findall(r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)', text)
    
    for start_year, end_year in year_ranges:
        try:
            start = int(start_year)
            
            if 'present' in str(end_year).lower() or 'current' in str(end_year).lower():
                end = current_year
            else:
                end = int(end_year)
            
            if 1980 <= start <= current_year and start <= end <= current_year:
                total_months += (end - start) * 12
        except:
            continue
    
    # Pattern 4: Month mentions
    month_pattern = r'(\d+)\s*months?\s+(?:of\s+)?experience'
    month_matches = re.findall(month_pattern, text_lower)
    
    if month_matches and total_months == 0:
        months = sum(int(m) for m in month_matches if int(m) <= 240)
        total_months = months
    
    return round(total_months / 12, 1) if total_months > 0 else 0.0

def extract_achievements(text: str) -> tuple[int, int]:
    """Detect quantifiable achievements"""
    achievements = []
    
    achievement_patterns = [
        r'(?:increased|improved|reduced|achieved|delivered|generated|saved)\s+[^.]*?(\d+)%',
        r'(\d+)%\s+(?:increase|improvement|reduction|growth)',
        r'(?:managed|led|supervised)\s+(?:team\s+of\s+)?(\d+)',
        r'\$(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|m|k|thousand)?',
        r'(\d+)\+?\s+(?:projects?|clients?|users?|customers?)'
    ]
    
    for pattern in achievement_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        achievements.extend(matches)
    
    achievement_count = len(achievements)
    
    if achievement_count >= 5:
        score = 5
    elif achievement_count >= 3:
        score = 3
    elif achievement_count >= 1:
        score = 2
    else:
        score = 0
    
    return score, achievement_count

def assess_resume_quality(text: str) -> tuple[float, List[str]]:
    """Evaluate resume formatting and structure - max 10 points"""
    score = 0.0
    issues = []
    
    # Word count check
    word_count = len(text.split())
    if 300 <= word_count <= 1000:
        score += 3
    elif word_count < 200:
        issues.append("Resume too brief (< 200 words)")
        score -= 1
    elif word_count > 1500:
        issues.append("Resume too verbose (> 1500 words)")
    
    # Check for essential sections
    required_sections = ['experience', 'education', 'skills']
    text_lower = text.lower()
    found_sections = sum(1 for sec in required_sections if sec in text_lower)
    score += found_sections * 2
    
    if found_sections < 3:
        issues.append(f"Missing {3 - found_sections} essential section(s)")
    
    # Check for bullet points
    bullet_count = len(re.findall(r'[\n\r]\s*[•▪▸■●\-\*]\s+', text))
    if bullet_count >= 5:
        score += 2
    elif bullet_count == 0:
        issues.append("No bullet points - poor structure")
    
    # Check for contact info
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'[\+]?[\d\s\-\(\)]{10,}', text))
    
    if has_email:
        score += 1.5
    else:
        issues.append("No email detected")
    
    if has_phone:
        score += 1.5
    else:
        issues.append("No phone detected")
    
    return min(score, 10.0), issues

def calculate_keyword_relevance(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Calculate semantic similarity using TF-IDF - max 15 points"""
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        jd_keywords = vectorizer.get_feature_names_out()
        resume_lower = resume_text.lower()
        
        matched_keywords = [kw for kw in jd_keywords if kw in resume_lower]
        match_rate = len(matched_keywords) / len(jd_keywords) if len(jd_keywords) > 0 else 0
        
        # Combined score (70% similarity + 30% keyword match)
        combined_score = (similarity * 0.7) + (match_rate * 0.3)
        final_score = combined_score * 15  # Scale to 15 points
        
        return {
            'score': min(final_score, 15.0),
            'similarity': round(similarity * 100, 2),
            'matched_keywords': matched_keywords[:10],
            'match_rate': round(match_rate * 100, 2)
        }
        
    except Exception as e:
        logger.error(f"Keyword relevance error: {e}")
        return {
            'score': 0.0,
            'similarity': 0.0,
            'matched_keywords': [],
            'match_rate': 0.0
        }

def parse_resume(text: str) -> Dict[str, Any]:
    """Main resume parsing function"""
    if not nlp:
        raise Exception("NLP model not available")
    
    if not text or len(text.strip()) < 100:
        raise Exception("Resume text is too short or empty")
    
    # Extract skills
    skills = parse_skills_advanced(text)
    
    # Extract experience
    experience_years = extract_experience_enhanced(text)
    
    # Extract sections
    projects_section = extract_section_advanced(text, 'projects')
    education_section = extract_section_advanced(text, 'education')
    experience_section = extract_section_advanced(text, 'experience')
    
    return {
        'skills': skills,
        'experience_years': experience_years,
        'projects_section': projects_section,
        'education_section': education_section,
        'experience_section': experience_section
    }

def validate_projects(projects_section: str, skills: List[str]) -> tuple[float, List[str]]:
    """Validate that projects demonstrate claimed skills"""
    if not projects_section or not skills:
        return 0.0, []
    
    projects_lower = projects_section.lower()
    verified_skills = [s for s in skills if s.lower() in projects_lower]
    verification_rate = len(verified_skills) / len(skills) if skills else 0.0
    
    return verification_rate, verified_skills

def validate_education(education_section: str, jd_education: str = "") -> tuple[float, List[str]]:
    """Validate education section - max 10 points"""
    score = 0.0
    penalties = []
    
    if not education_section:
        return 0.0, ["No education section found"]
    
    edu_lower = education_section.lower()
    jd_lower = jd_education.lower() if jd_education else ""
    
    # Check for degrees
    degrees = ['bachelor', 'b.tech', 'b.e.', 'bsc', 'master', 'm.tech', 'm.sc', 'phd', 'mba']
    jd_degrees = [d for d in degrees if d in jd_lower]
    resume_degrees = [d for d in degrees if d in edu_lower]
    
    if jd_degrees and resume_degrees:
        if any(jd_deg in resume_degrees for jd_deg in jd_degrees):
            score += 5
        else:
            score += 2
    elif resume_degrees and not jd_degrees:
        score += 3
    
    # CGPA validation (fixed - no penalty for excellence)
    cgpa_pattern = r'(?:cgpa|gpa|grade)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?'
    cgpa_matches = re.findall(cgpa_pattern, edu_lower)
    
    for match in cgpa_matches:
        try:
            cgpa_value = float(match[0])
            max_scale = float(match[1]) if match[1] else 10.0
            
            if max_scale <= 0:
                continue
            
            if cgpa_value > max_scale:
                penalties.append(f"Invalid CGPA: {cgpa_value}/{max_scale}")
                score -= 3
            elif cgpa_value >= max_scale * 0.7:
                score += 2  # Reward good academic performance
        except (ValueError, ZeroDivisionError):
            continue
    
    # Field of study
    fields = [
        'computer science', 'software engineering', 'information technology',
        'electrical engineering', 'electronics', 'data science', 'artificial intelligence'
    ]
    
    if jd_lower:
        jd_fields = [f for f in fields if f in jd_lower]
        resume_fields = [f for f in fields if f in edu_lower]
        
        if jd_fields and resume_fields:
            if any(jf in resume_fields for jf in jd_fields):
                score += 3
            else:
                score += 1
    
    return min(max(score, 0.0), 10.0), penalties

def check_plagiarism(resume_text: str, reference_corpus: List[str] = None) -> tuple[float, str]:
    """Check for plagiarism against reference corpus"""
    if not reference_corpus or len(reference_corpus) == 0:
        return 0.0, "No reference data"
    
    try:
        corpus = [resume_text] + reference_corpus
        
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        max_similarity = float(np.max(similarities)) if similarities.size > 0 else 0.0
        plagiarism_score = round(max_similarity * 100, 2)
        
        return plagiarism_score, "Checked"
        
    except Exception as e:
        logger.error(f"Plagiarism check error: {e}")
        return 0.0, f"Error: {str(e)}"

# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def calculate_ats_score(
    resume_text: str,
    job_description: str,
    jd_education: str = "",
    reference_corpus: List[str] = None
) -> Dict[str, Any]:
    """Calculate comprehensive ATS score with improved algorithm"""
    
    if not resume_text or not job_description:
        raise Exception("Resume text and job description are required")
    
    if len(resume_text.strip()) < 100:
        raise Exception("Resume text is too short")
    
    # Parse resume
    parsed = parse_resume(resume_text)
    
    score = 0.0
    feedback = []
    penalties = []
    breakdown = {}
    
    jd_lower = job_description.lower()
    
    # 1. Skills Matching (35 points)
    matched_skills = [s for s in parsed['skills'] if s.lower() in jd_lower]
    
    if parsed['skills']:
        skills_score = (len(matched_skills) / len(parsed['skills'])) * 35
        score += skills_score
        breakdown['skills_match'] = round(skills_score, 2)
        feedback.append(f"Matched {len(matched_skills)}/{len(parsed['skills'])} skills ({skills_score:.1f}/35)")
    else:
        breakdown['skills_match'] = 0.0
        feedback.append("No skills detected")
    
    # 2. Experience (25 points)
    exp_match = re.search(r'(\d+)\+?\s*years?', jd_lower)
    required_exp = int(exp_match.group(1)) if exp_match else 2
    
    if parsed['experience_years'] >= required_exp:
        exp_score = 25.0
    elif parsed['experience_years'] > 0:
        exp_score = (parsed['experience_years'] / required_exp) * 25
    else:
        exp_score = 0.0
    
    # Achievements bonus (within experience)
    ach_score, ach_count = extract_achievements(resume_text)
    if ach_count > 0:
        feedback.append(f"Found {ach_count} quantifiable achievements")
    
    score += exp_score
    breakdown['experience'] = round(exp_score, 2)
    feedback.append(f"Experience: {parsed['experience_years']} years ({exp_score:.1f}/25)")
    
    # 3. Keyword Relevance (15 points)
    keyword_result = calculate_keyword_relevance(resume_text, job_description)
    score += keyword_result['score']
    breakdown['keyword_relevance'] = round(keyword_result['score'], 2)
    feedback.append(f"Keyword match: {keyword_result['similarity']}% similarity ({keyword_result['score']:.1f}/15)")
    
    # 4. Education (10 points)
    education_score, edu_penalties = validate_education(parsed['education_section'], jd_education)
    score += education_score
    breakdown['education'] = round(education_score, 2)
    penalties.extend(edu_penalties)
    feedback.append(f"Education: {education_score:.1f}/10")
    
    # 5. Resume Quality (10 points)
    quality_score, quality_issues = assess_resume_quality(resume_text)
    score += quality_score
    breakdown['resume_quality'] = round(quality_score, 2)
    if quality_issues:
        penalties.extend(quality_issues)
    feedback.append(f"Resume quality: {quality_score:.1f}/10")
    
    # 6. Projects (5 points)
    project_verification, verified_skills = validate_projects(
        parsed['projects_section'],
        parsed['skills']
    )
    project_score = project_verification * 5
    score += project_score
    breakdown['projects'] = round(project_score, 2)
    feedback.append(f"Projects: {project_score:.1f}/5")
    
    # Plagiarism check (penalty only)
    plagiarism_score = 0.0
    if reference_corpus:
        plagiarism_score, _ = check_plagiarism(resume_text, reference_corpus)
        
        if plagiarism_score > 80:
            penalties.append(f"Critical plagiarism: {plagiarism_score}% (-20)")
            score -= 20
        elif plagiarism_score > 60:
            penalties.append(f"High plagiarism: {plagiarism_score}% (-10)")
            score -= 10
        elif plagiarism_score > 40:
            penalties.append(f"Moderate plagiarism: {plagiarism_score}% (-5)")
            score -= 5
    
    # Keyword stuffing check
    for skill in parsed['skills'][:5]:
        count = resume_text.lower().count(skill.lower())
        if count > 15:
            penalties.append(f"Keyword stuffing: '{skill}' repeated {count} times (-5)")
            score -= 5
            break
    
    final_score = max(0.0, min(score, 100.0))
    
    return {
        **parsed,
        'score': round(final_score, 2),
        'breakdown': breakdown,
        'matched_skills': matched_skills,
        'feedback': feedback,
        'penalties': penalties,
        'plagiarism_score': plagiarism_score,
        'keyword_similarity': keyword_result['similarity']
    }

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    text = text.strip()[:max_length]
    text = re.sub(r'[<>"\';]', '', text)
    return text

def validate_email(email: str) -> bool:
    """Validate email format - accepts all valid emails"""
    if not email:
        return False
    email = email.strip().lower()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

def validate_mobile(mobile: str) -> bool:
    """Validate mobile number"""
    if not mobile:
        return False
    mobile_clean = re.sub(r'[\s\-\+()]', '', mobile)
    return len(mobile_clean) >= 10 and mobile_clean.isdigit()

def register_participant(name: str, email: str, mobile: str) -> str:
    """Register new participant in database"""
    if not supabase:
        return str(uuid.uuid4())
    
    try:
        name = sanitize_input(name, 200)
        email = sanitize_input(email, 200).lower()
        mobile = sanitize_input(mobile, 20)
        
        if not name or len(name) < 3:
            raise Exception("Invalid name")
        
        if not validate_email(email):
            raise Exception("Invalid email format")
        
        if not validate_mobile(mobile):
            raise Exception("Invalid mobile number")
        
        participant_id = str(uuid.uuid4())
        data = {
            'id': participant_id,
            'name': name,
            'email': email,
            'mobile': mobile
        }
        
        supabase.table('participants').insert(data).execute()
        logger.info(f"✅ Registered participant: {email}")
        return participant_id
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise Exception(f"Registration failed: {str(e)}")

def check_participant_exists(email: str) -> Optional[Dict]:
    """Check if participant already exists"""
    if not supabase:
        return None
    
    try:
        email = sanitize_input(email, 200).lower()
        response = supabase.table('participants').select('*').eq('email', email).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
        
    except Exception as e:
        logger.error(f"Check participant error: {e}")
        return None

def save_participant_application(
    participant_id: str,
    score: float,
    skills_count: int,
    experience_years: float,
    matched_skills_count: int,
    plagiarism_score: float,
    keyword_similarity: float,
    resume_quality_score: float
) -> bool:
    """Save application submission"""
    if not supabase:
        return False
    
    try:
        data = {
            'participant_id': participant_id,
            'score': float(score),
            'skills_count': int(skills_count),
            'experience_years': float(experience_years),
            'matched_skills_count': int(matched_skills_count),
            'plagiarism_score': float(plagiarism_score),
            'keyword_similarity': float(keyword_similarity),
            'resume_quality_score': float(resume_quality_score)
        }
        
        supabase.table('applications').insert(data).execute()
        logger.info(f"✅ Saved application for participant: {participant_id}, Score: {score}")
        return True
        
    except Exception as e:
        logger.error(f"Save application error: {e}")
        raise Exception(f"Failed to save application: {str(e)}")

def get_participant_upload_count(participant_id: str) -> int:
    """Get number of uploads for participant"""
    if not supabase or not participant_id:
        return 0
    
    try:
        response = supabase.table('applications').select('id', count='exact').eq('participant_id', participant_id).execute()
        return response.count if response.count else 0
    except Exception as e:
        logger.error(f"Upload count error: {e}")
        return 0

def get_reference_corpus() -> List[str]:
    """Get reference corpus for plagiarism checking"""
    if not supabase:
        return []
    
    try:
        response = supabase.table('resume_corpus').select('resume_text').limit(100).execute()
        if response.data:
            return [item['resume_text'] for item in response.data]
        return []
    except Exception as e:
        logger.error(f"Reference corpus error: {e}")
        return []

def save_to_corpus(participant_id: str, resume_text: str):
    """Save resume to corpus for plagiarism checking"""
    if not supabase:
        return
    
    try:
        data = {
            'participant_id': participant_id,
            'resume_text': resume_text
        }
        supabase.table('resume_corpus').insert(data).execute()
    except Exception as e:
        logger.error(f"Save to corpus error: {e}")

def get_participant_scores(participant_id: str) -> pd.DataFrame:
    """Get all scores for a participant"""
    if not supabase or not participant_id:
        return pd.DataFrame()
    
    try:
        response = supabase.table('applications').select('*').eq('participant_id', participant_id).order('created_at', desc=True).execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Get scores error: {e}")
        return pd.DataFrame()

def get_leaderboard() -> List[Dict]:
    """Get top 10 leaderboard"""
    if not supabase:
        return []
    
    try:
        # Use the view we created
        response = supabase.table('leaderboard').select('*').limit(10).execute()
        
        if response.data:
            return response.data
        return []
        
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        return []

def get_competition_stats() -> Optional[Dict]:
    """Get competition statistics"""
    if not supabase:
        return None
    
    try:
        apps_response = supabase.table('applications').select('score, experience_years').execute()
        participants_response = supabase.table('participants').select('id', count='exact').execute()
        
        if not apps_response.data:
            return None
        
        df = pd.DataFrame(apps_response.data)
        
        stats = {
            'total_participants': participants_response.count if participants_response.count else 0,
            'total_submissions': len(df),
            'avg_score': float(df['score'].mean()),
            'median_score': float(df['score'].median()),
            'top_score': float(df['score'].max()),
            'high_scorers': int(len(df[df['score'] >= 80])),
            'score_distribution': [
                {'range': '0-60%', 'count': int(len(df[df['score'] < 60]))},
                {'range': '60-80%', 'count': int(len(df[(df['score'] >= 60) & (df['score'] < 80)]))},
                {'range': '80-100%', 'count': int(len(df[df['score'] >= 80]))}
            ]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return None

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "competition": "Perfect CV Match 2025",
        "organization": "Microsoft Learn Student Chapter @ TIET",
        "version": "2.0.0",
        "nlp_loaded": nlp is not None,
        "database_connected": supabase is not None,
        "scoring_weights": SCORING_WEIGHTS
    }

@app.post("/api/register", response_model=ParticipantResponse)
async def api_register(participant: ParticipantRegistration):
    """Register a new participant"""
    
    # Validation
    if len(participant.name) < 3:
        raise HTTPException(status_code=400, detail="Name must be at least 3 characters")
    
    # Uncomment this if you want to restrict to specific domain
    # if '@thapar.edu' not in participant.email.lower():
    #     raise HTTPException(status_code=400, detail="Valid Thapar email required")
    
    mobile_clean = participant.mobile.replace('+', '').replace(' ', '').replace('-', '')
    if len(mobile_clean) < 10:
        raise HTTPException(status_code=400, detail="Valid mobile number required")
    
    try:
        existing = check_participant_exists(participant.email)
        
        if existing:
            participant_id = existing['id']
            upload_count = get_participant_upload_count(participant_id)
            return ParticipantResponse(
                id=participant_id,
                name=existing['name'],
                email=existing['email'],
                mobile=existing['mobile'],
                upload_count=upload_count,
                message="Welcome back! You are already registered."
            )
        
        participant_id = register_participant(
            participant.name,
            participant.email,
            participant.mobile
        )
        
        return ParticipantResponse(
            id=participant_id,
            name=participant.name,
            email=participant.email,
            mobile=participant.mobile,
            upload_count=0,
            message="Registration successful! You can now submit your resume."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/submit", response_model=ScoreResponse)
async def api_submit_resume(
    participant_id: str = Form(...),
    job_description: str = Form(...),
    jd_education: str = Form(""),
    resume: UploadFile = File(...)
):
    """Submit resume and get ATS score"""
    
    if not nlp:
        raise HTTPException(
            status_code=503,
            detail="NLP service unavailable. Please try again later."
        )
    
    # Check upload limit
    upload_count = get_participant_upload_count(participant_id)
    if upload_count >= MAX_UPLOADS:
        raise HTTPException(
            status_code=400,
            detail=f"Upload limit of {MAX_UPLOADS} submissions reached"
        )
    
    # Rate limiting
    current_time = time.time()
    if participant_id in last_submissions:
        time_since_last = current_time - last_submissions[participant_id]
        if time_since_last < RATE_LIMIT_SECONDS:
            wait_time = int(RATE_LIMIT_SECONDS - time_since_last)
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {wait_time} seconds before next submission"
            )
    
    # Validate file
    content = await resume.read()
    is_valid, message = validate_pdf_file(content, resume.filename)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # Validate job description
    if not job_description or len(job_description.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Job description must be at least 50 characters"
        )
    
    try:
        # Extract text
        text = extract_pdf_text(content)
        
        if len(text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from PDF"
            )
        
        # Get reference corpus for plagiarism
        reference_corpus = get_reference_corpus()
        
        # Calculate ATS score
        result = calculate_ats_score(
            text,
            job_description,
            jd_education,
            reference_corpus
        )
        
        # Save application
        save_participant_application(
            participant_id=participant_id,
            score=result['score'],
            skills_count=len(result['skills']),
            experience_years=result['experience_years'],
            matched_skills_count=len(result['matched_skills']),
            plagiarism_score=result['plagiarism_score'],
            keyword_similarity=result['keyword_similarity'],
            resume_quality_score=result['breakdown'].get('resume_quality', 0)
        )
        
        # Save to corpus
        save_to_corpus(participant_id, text)
        
        # Update rate limiting
        last_submissions[participant_id] = current_time
        
        # Determine verdict
        score = result['score']
        if score >= 80:
            verdict = "Excellent Match ⭐"
        elif score >= 60:
            verdict = "Good Match ✓"
        elif score >= 40:
            verdict = "Fair Match"
        else:
            verdict = "Needs Improvement"
        
        return ScoreResponse(
            score=result['score'],
            breakdown=result['breakdown'],
            skills=result['skills'],
            matched_skills=result['matched_skills'],
            experience_years=result['experience_years'],
            feedback=result['feedback'],
            penalties=result['penalties'],
            plagiarism_score=result['plagiarism_score'],
            keyword_similarity=result['keyword_similarity'],
            upload_count=upload_count + 1,
            verdict=verdict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.get("/api/participant/{participant_id}/scores")
async def api_get_scores(participant_id: str):
    """Get all scores for a participant"""
    try:
        scores_df = get_participant_scores(participant_id)
        
        if scores_df.empty:
            return {
                "scores": [],
                "best_score": None,
                "total_submissions": 0,
                "average_score": None
            }
        
        scores_list = scores_df.to_dict('records')
        for score in scores_list:
            if 'created_at' in score:
                score['created_at'] = str(score['created_at'])
        
        return {
            "scores": scores_list,
            "best_score": float(scores_df['score'].max()),
            "average_score": float(scores_df['score'].mean()),
            "total_submissions": len(scores_list)
        }
        
    except Exception as e:
        logger.error(f"Error fetching scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/participant/{participant_id}/upload-count")
async def api_get_upload_count(participant_id: str):
    """Get upload count for a participant"""
    try:
        count = get_participant_upload_count(participant_id)
        return {
            "upload_count": count,
            "max_uploads": MAX_UPLOADS,
            "remaining": MAX_UPLOADS - count
        }
    except Exception as e:
        logger.error(f"Error fetching upload count: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/leaderboard")
async def api_leaderboard():
    """Get top 10 leaderboard"""
    try:
        data = get_leaderboard()
        return {"leaderboard": data}
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def api_stats():
    """Get competition statistics"""
    try:
        data = get_competition_stats()
        
        if not data:
            return {
                "total_participants": 0,
                "total_submissions": 0,
                "avg_score": 0,
                "median_score": 0,
                "top_score": 0,
                "high_scorers": 0,
                "score_distribution": []
            }
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
