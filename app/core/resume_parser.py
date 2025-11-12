"""
Resume parsing and information extraction - FIXED
Separates education dates from work experience dates
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
        'Asynchronous Programming': r'\b(asynchronous|async)\b',
        'MLOps': r'\bmlops\b',
        'Next.js': r'\bnext(\.js)?\b',
        'Grafana': r'\bgrafana\b',
        'Prometheus': r'\bprometheus\b',
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
    """
    Extract years of WORK experience ONLY (not education years)
    FIXED: Separates education dates from work experience dates
    """
    text_lower = text.lower()
    current_year = datetime.now().year
    
    # Pattern 1: Direct years mention (most reliable)
    direct_patterns = [
        r'(\d+\.?\d*)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:work\s+)?experience',
        r'(?:work\s+)?experience[:\s]+(\d+\.?\d*)\+?\s*(?:years?|yrs?)',
        r'(\d+\.?\d*)\s*(?:years?|yrs?)\s+in\s+(?:software|development|engineering)',
    ]
    
    for pattern in direct_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            years = [float(m) for m in matches if 0 < float(m) <= 50]
            if years:
                return float(max(years))
    
    # Pattern 2: Extract ONLY from "EXPERIENCE" section
    # Find experience section boundaries
    exp_section_match = re.search(
        r'(?:professional\s+)?(?:work\s+)?experience[:\s]*\n(.*?)(?=\n\s*(?:education|projects|skills|certifications?|$))',
        text_lower,
        re.DOTALL | re.IGNORECASE
    )
    
    if not exp_section_match:
        # Try alternative pattern
        exp_section_match = re.search(
            r'experience.*?\n(.*?)(?=education|projects|skills|$)',
            text_lower,
            re.DOTALL | re.IGNORECASE
        )
    
    if not exp_section_match:
        # No clear experience section found
        return 0.0
    
    exp_section_text = exp_section_match.group(1)
    
    # Pattern 3: Calculate from date ranges ONLY in experience section
    total_months = 0
    
    # Month-Year format (June 2023 - Present)
    date_ranges = re.findall(
        r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*[-–—to]\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|present|current)',
        exp_section_text,
        re.IGNORECASE
    )
    
    for start_str, end_str in date_ranges:
        try:
            start_date = date_parse(start_str, fuzzy=True)
            end_date = datetime.now() if 'present' in end_str or 'current' in end_str else date_parse(end_str, fuzzy=True)
            
            # Only count if start date is after 2000 and reasonable
            if start_date.year >= 2000 and start_date <= datetime.now():
                months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                if 0 < months_diff <= 240:  # Max 20 years
                    total_months += months_diff
        except:
            continue
    
    # Pattern 4: Year-only ranges ONLY if no month-year ranges found
    if not date_ranges:
        year_ranges = re.findall(r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)', exp_section_text)
        
        for start_year, end_year in year_ranges:
            try:
                start = int(start_year)
                end = current_year if 'present' in str(end_year).lower() or 'current' in str(end_year).lower() else int(end_year)
                
                # Skip if it's a typical education duration (3-5 years)
                duration = end - start
                if duration in [3, 4, 5]:
                    # Check if it looks like education
                    context = exp_section_text[max(0, exp_section_text.find(start_year)-100):exp_section_text.find(start_year)+100]
                    if 'bachelor' in context or 'b.tech' in context or 'degree' in context:
                        continue
                
                if 2000 <= start <= current_year and start <= end <= current_year:
                    months = (end - start) * 12
                    if 0 < months <= 240:
                        total_months += months
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
