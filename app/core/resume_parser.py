"""
Resume parsing and information extraction
"""
import re
import spacy
from typing import List, Dict
from datetime import datetime
from dateutil.parser import parse as date_parse
import logging

logger = logging.getLogger(__name__)

# Global NLP model
nlp = None

def load_nlp_model():
    """Load spaCy NLP model"""
    global nlp
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ SpaCy model loaded")
    except OSError:
        logger.error("❌ SpaCy model not found")
        nlp = None

def extract_section(text: str, section_type: str) -> str:
    """Extract specific section from resume"""
    section_headers = {
        'experience': [
            r'(?:work\s+)?experience', r'employment\s+history',
            r'professional\s+experience', r'career\s+history',
            r'work\s+history'
        ],
        'education': [
            r'education(?:al\s+background)?', r'academic\s+(?:background|qualifications?)',
            r'qualifications?', r'degrees?'
        ],
        'skills': [
            r'(?:technical\s+)?skills?', r'core\s+competencies',
            r'expertise', r'proficiencies?'
        ],
        'projects': [
            r'projects?', r'portfolio', r'key\s+projects?'
        ]
    }
    
    lines = text.split('\n')
    section_text = []
    capturing = False
    headers = section_headers.get(section_type.lower(), [])
    
    for line in lines:
        line_stripped = line.strip().lower()
        is_section_start = any(re.match(f'^{h}s?:?$', line_stripped) for h in headers)
        
        if is_section_start:
            capturing = True
            continue
        
        if capturing:
            all_headers = [h for headers_list in section_headers.values() for h in headers_list]
            is_new_section = any(re.match(f'^{h}s?:?$', line_stripped) for h in all_headers)
            
            if is_new_section:
                break
            
            if line.strip():
                section_text.append(line)
    
    return '\n'.join(section_text)

def extract_skills(text: str) -> List[str]:
    """Extract skills using word boundaries"""
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
        'Deep Learning': r'\bdeep\s+learning\b',
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

def extract_experience(text: str) -> float:
    """Extract years of experience with date intelligence"""
    text_lower = text.lower()
    current_year = datetime.now().year
    
    # Pattern 1: Direct years mention
    direct_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
    ]
    
    for pattern in direct_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            years = [int(m) for m in matches if 0 < int(m) <= 50]
            if years:
                return float(max(years))
    
    # Pattern 2: Date ranges
    total_months = 0
    date_range_pattern = r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*[-–—to]\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|present|current)'
    
    ranges = re.findall(date_range_pattern, text_lower, re.IGNORECASE)
    
    for start_str, end_str in ranges:
        try:
            start_date = date_parse(start_str, fuzzy=True)
            end_date = datetime.now() if 'present' in end_str or 'current' in end_str else date_parse(end_str, fuzzy=True)
            
            months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            if 0 < months_diff <= 600:
                total_months += months_diff
        except:
            continue
    
    # Pattern 3: Year-only ranges
    year_ranges = re.findall(r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)', text)
    
    for start_year, end_year in year_ranges:
        try:
            start = int(start_year)
            end = current_year if 'present' in str(end_year).lower() or 'current' in str(end_year).lower() else int(end_year)
            
            if 1980 <= start <= current_year and start <= end <= current_year:
                total_months += (end - start) * 12
        except:
            continue
    
    return round(total_months / 12, 1) if total_months > 0 else 0.0

def parse_resume(text: str) -> Dict:
    """Main resume parsing function"""
    if not nlp:
        raise Exception("NLP model not loaded")
    
    if len(text.strip()) < 100:
        raise Exception("Resume text too short")
    
    return {
        'skills': extract_skills(text),
        'experience_years': extract_experience(text),
        'projects_section': extract_section(text, 'projects'),
        'education_section': extract_section(text, 'education'),
        'experience_section': extract_section(text, 'experience')
    }
