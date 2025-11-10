"""
ATS scoring engine - main scoring logic
"""
import re
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def calculate_keyword_relevance(resume_text: str, job_description: str) -> Dict:
    """Calculate TF-IDF based keyword relevance"""
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
        
        combined_score = (similarity * 0.7) + (match_rate * 0.3)
        final_score = combined_score * 15
        
        return {
            'score': min(final_score, 15.0),
            'similarity': round(similarity * 100, 2),
            'matched_keywords': matched_keywords[:10],
            'match_rate': round(match_rate * 100, 2)
        }
    except Exception as e:
        logger.error(f"Keyword relevance error: {e}")
        return {'score': 0.0, 'similarity': 0.0, 'matched_keywords': [], 'match_rate': 0.0}

def validate_projects(projects_section: str, skills: List[str]) -> Tuple[float, List[str]]:
    """Validate projects demonstrate claimed skills"""
    if not projects_section or not skills:
        return 0.0, []
    
    projects_lower = projects_section.lower()
    verified_skills = [s for s in skills if s.lower() in projects_lower]
    verification_rate = len(verified_skills) / len(skills) if skills else 0.0
    
    return verification_rate, verified_skills

def validate_education(education_section: str, jd_education: str = "") -> Tuple[float, List[str]]:
    """Validate education section"""
    score = 0.0
    penalties = []
    
    if not education_section:
        return 0.0, ["No education section found"]
    
    edu_lower = education_section.lower()
    jd_lower = jd_education.lower() if jd_education else ""
    
    # Degree matching
    degrees = ['bachelor', 'b.tech', 'b.e.', 'bsc', 'master', 'm.tech', 'm.sc', 'phd', 'mba']
    jd_degrees = [d for d in degrees if d in jd_lower]
    resume_degrees = [d for d in degrees if d in edu_lower]
    
    if jd_degrees and resume_degrees:
        score += 5 if any(jd_deg in resume_degrees for jd_deg in jd_degrees) else 2
    elif resume_degrees:
        score += 3
    
    # CGPA validation (fixed - no penalty for excellence)
    cgpa_pattern = r'(?:cgpa|gpa|grade)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?'
    cgpa_matches = re.findall(cgpa_pattern, edu_lower)
    
    for match in cgpa_matches:
        try:
            cgpa_value = float(match[0])
            max_scale = float(match[1]) if match[1] else 10.0
            
            if cgpa_value > max_scale:
                penalties.append(f"Invalid CGPA: {cgpa_value}/{max_scale}")
                score -= 3
            elif cgpa_value >= max_scale * 0.7:
                score += 2
        except (ValueError, ZeroDivisionError):
            continue
    
    # Field of study
    fields = ['computer science', 'software engineering', 'information technology',
              'electrical engineering', 'electronics', 'data science']
    
    if jd_lower:
        jd_fields = [f for f in fields if f in jd_lower]
        resume_fields = [f for f in fields if f in edu_lower]
        
        if jd_fields and resume_fields:
            score += 3 if any(jf in resume_fields for jf in jd_fields) else 1
    
    return min(max(score, 0.0), 10.0), penalties

def assess_resume_quality(text: str) -> Tuple[float, List[str]]:
    """Assess resume format and quality"""
    score = 0.0
    issues = []
    
    word_count = len(text.split())
    if 300 <= word_count <= 1000:
        score += 3
    elif word_count < 200:
        issues.append("Resume too brief")
        score -= 1
    elif word_count > 1500:
        issues.append("Resume too verbose")
    
    # Essential sections
    required = ['experience', 'education', 'skills']
    text_lower = text.lower()
    found = sum(1 for sec in required if sec in text_lower)
    score += found * 2
    
    if found < 3:
        issues.append(f"Missing {3-found} section(s)")
    
    # Bullet points
    bullets = len(re.findall(r'[\n\r]\s*[•▪▸■●\-\*]\s+', text))
    if bullets >= 5:
        score += 2
    elif bullets == 0:
        issues.append("No bullet points")
    
    # Contact info
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'[\+]?[\d\s\-\(\)]{10,}', text))
    
    if has_email:
        score += 1.5
    else:
        issues.append("No email")
    
    if has_phone:
        score += 1.5
    else:
        issues.append("No phone")
    
    return min(score, 10.0), issues

def extract_achievements(text: str) -> Tuple[int, int]:
    """Extract quantifiable achievements"""
    patterns = [
        r'(?:increased|improved|reduced|achieved|delivered|generated|saved)\s+[^.]*?(\d+)%',
        r'(\d+)%\s+(?:increase|improvement|reduction|growth)',
        r'(?:managed|led|supervised)\s+(?:team\s+of\s+)?(\d+)',
        r'\$(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|m|k|thousand)?',
        r'(\d+)\+?\s+(?:projects?|clients?|users?|customers?)'
    ]
    
    achievements = []
    for pattern in patterns:
        achievements.extend(re.findall(pattern, text, re.IGNORECASE))
    
    count = len(achievements)
    score = 5 if count >= 5 else (3 if count >= 3 else (2 if count >= 1 else 0))
    
    return score, count

def calculate_ats_score(resume_text: str, job_description: str, parsed_resume: Dict,
                       jd_education: str = "") -> Dict:
    """Main ATS scoring function"""
    score = 0.0
    feedback = []
    penalties = []
    breakdown = {}
    
    jd_lower = job_description.lower()
    
    # 1. Skills Matching (35 points)
    matched_skills = [s for s in parsed_resume['skills'] if s.lower() in jd_lower]
    
    if parsed_resume['skills']:
        skills_score = (len(matched_skills) / len(parsed_resume['skills'])) * 35
        score += skills_score
        breakdown['skills_match'] = round(skills_score, 2)
        feedback.append(f"Matched {len(matched_skills)}/{len(parsed_resume['skills'])} skills")
    else:
        breakdown['skills_match'] = 0.0
        feedback.append("No skills detected")
    
    # 2. Experience (25 points)
    exp_match = re.search(r'(\d+)\+?\s*years?', jd_lower)
    required_exp = int(exp_match.group(1)) if exp_match else 2
    
    exp_score = 25.0 if parsed_resume['experience_years'] >= required_exp else \
                (parsed_resume['experience_years'] / required_exp) * 25 if parsed_resume['experience_years'] > 0 else 0.0
    
    score += exp_score
    breakdown['experience'] = round(exp_score, 2)
    feedback.append(f"Experience: {parsed_resume['experience_years']} years")
    
    # 3. Keyword Relevance (15 points)
    keyword_result = calculate_keyword_relevance(resume_text, job_description)
    score += keyword_result['score']
    breakdown['keyword_relevance'] = round(keyword_result['score'], 2)
    feedback.append(f"Keyword similarity: {keyword_result['similarity']}%")
    
    # 4. Education (10 points)
    education_score, edu_penalties = validate_education(parsed_resume['education_section'], jd_education)
    score += education_score
    breakdown['education'] = round(education_score, 2)
    penalties.extend(edu_penalties)
    
    # 5. Resume Quality (10 points)
    quality_score, quality_issues = assess_resume_quality(resume_text)
    score += quality_score
    breakdown['resume_quality'] = round(quality_score, 2)
    penalties.extend(quality_issues)
    
    # 6. Projects (5 points)
    project_verification, verified_skills = validate_projects(
        parsed_resume['projects_section'],
        parsed_resume['skills']
    )
    project_score = project_verification * 5
    score += project_score
    breakdown['projects'] = round(project_score, 2)
    
    # Keyword stuffing check
    for skill in parsed_resume['skills'][:5]:
        count = resume_text.lower().count(skill.lower())
        if count > 15:
            penalties.append(f"Keyword stuffing: '{skill}' repeated {count} times")
            score -= 5
            break
    
    final_score = max(0.0, min(score, 100.0))
    
    return {
        'score': round(final_score, 2),
        'breakdown': breakdown,
        'matched_skills': matched_skills,
        'feedback': feedback,
        'penalties': penalties,
        'keyword_similarity': keyword_result['similarity']
    }
