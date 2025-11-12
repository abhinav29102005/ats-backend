"""
ATS scoring engine - CloudAlly Systems Job Description
Skills: 30, Education: 20, Experience: 20, Skills in Projects: 15, 
Keyword Relevance: 10, Resume Quality: 5

Penalties:
- Missing required skill: -2 each
- Missing preferred skill: -1 each
- Skill not used in project: -1 each
- Repetitive keywords: -5
"""
import re
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# CloudAlly Systems Required Skills
REQUIRED_SKILLS = [
    'python', 'node.js', 'nodejs', 'typescript',
    'fastapi', 'flask',
    'postgresql', 'postgres',
    'rest api', 'restful api',
    'docker',
    'git',
    'ci/cd', 'github actions', 'jenkins',
    'aws', 'azure',
    'microservices', 'microservice'
]

# CloudAlly Systems Preferred Skills
PREFERRED_SKILLS = [
    'machine learning', 'ml', 'mlops',
    'react', 'next.js', 'nextjs',
    'agile', 'scrum'
]

def calculate_keyword_relevance(resume_text: str, job_description: str) -> Dict:
    """Calculate TF-IDF based keyword relevance (10 points)"""
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
        final_score = combined_score * 10
        
        return {
            'score': min(final_score, 10.0),
            'similarity': round(similarity * 100, 2),
            'matched_keywords': matched_keywords[:10],
            'match_rate': round(match_rate * 100, 2)
        }
    except Exception as e:
        logger.error(f"Keyword relevance error: {e}")
        return {'score': 0.0, 'similarity': 0.0, 'matched_keywords': [], 'match_rate': 0.0}

def detect_resume_skills(text: str) -> List[str]:
    """Detect skills from resume text"""
    all_skills = REQUIRED_SKILLS + PREFERRED_SKILLS + [
        'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'sql',
        'angular', 'vue', 'vue.js', 'django', 'spring', 'express',
        'kubernetes', 'k8s', 'gcp', 'mongodb', 'mysql', 'redis',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
        'deep learning', 'data science', 'html', 'css', 'graphql'
    ]
    
    text_lower = text.lower()
    detected = []
    seen = set()
    
    for skill in all_skills:
        # Use word boundaries for accurate matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE):
            skill_normalized = skill.lower()
            if skill_normalized not in seen:
                detected.append(skill)
                seen.add(skill_normalized)
    
    return detected

def validate_projects(projects_section: str, skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Validate skills are used in projects (15 points)
    Returns: (score, verified_skills, unverified_skills)
    """
    if not projects_section or not skills:
        return 0.0, [], skills
    
    projects_lower = projects_section.lower()
    verified_skills = []
    unverified_skills = []
    
    for skill in skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, projects_lower, re.IGNORECASE):
            verified_skills.append(skill)
        else:
            unverified_skills.append(skill)
    
    if not skills:
        return 0.0, [], []
    
    verification_rate = len(verified_skills) / len(skills)
    score = verification_rate * 15
    
    return score, verified_skills, unverified_skills

def count_resume_pages(text: str) -> int:
    """Estimate pages (500 words per page)"""
    word_count = len(text.split())
    pages = max(1, round(word_count / 500))
    return pages

def validate_education(education_section: str) -> Tuple[float, List[str]]:
    """
    Validate education (20 points)
    Penalty: -2 if CGPA > 9.42
    """
    score = 0.0
    penalties = []
    
    if not education_section:
        penalties.append("No education section")
        return 0.0, penalties
    
    edu_lower = education_section.lower()
    
    # Degree matching (10 points)
    degrees = ['bachelor', 'b.tech', 'b.e.', 'bsc', 'master', 'm.tech', 'm.sc', 'phd', 'mba']
    resume_degrees = [d for d in degrees if d in edu_lower]
    
    if resume_degrees:
        score += 10
    
    # CGPA validation (10 points)
    cgpa_pattern = r'(?:cgpa|gpa|grade)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?'
    cgpa_matches = re.findall(cgpa_pattern, edu_lower)
    
    cgpa_valid = False
    for match in cgpa_matches:
        try:
            cgpa_value = float(match[0])
            max_scale = float(match[1]) if match[1] else 10.0
            
            if cgpa_value > 9.42 and max_scale == 10.0:
                penalties.append(f"Invalid CGPA: {cgpa_value}/10.0 (>9.42) (-2)")
                score -= 2
            elif cgpa_value > max_scale:
                penalties.append(f"Invalid CGPA: {cgpa_value}/{max_scale} (-2)")
                score -= 2
            elif cgpa_value >= max_scale * 0.6:
                score += 10
                cgpa_valid = True
        except (ValueError, ZeroDivisionError):
            continue
    
    return min(max(score, 0.0), 20.0), penalties

def assess_resume_quality(text: str) -> Tuple[float, List[str]]:
    """
    Resume quality (5 points)
    Penalty: -2 per extra page beyond 1
    """
    score = 5.0
    issues = []
    
    # Page penalty
    pages = count_resume_pages(text)
    if pages > 1:
        penalty = (pages - 1) * 2
        issues.append(f"{pages} pages (-{penalty} for {pages-1} extra)")
        score -= penalty
    
    # Word count
    word_count = len(text.split())
    if word_count < 200:
        issues.append("Resume too brief")
    elif word_count > 1500:
        issues.append("Resume too verbose")
    
    # Sections
    required = ['experience', 'education', 'skills']
    text_lower = text.lower()
    missing = [s for s in required if s not in text_lower]
    
    if missing:
        issues.append(f"Missing: {', '.join(missing)}")
    
    # Contact
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'[\+]?[\d\s\-\(\)]{10,}', text))
    
    if not has_email:
        issues.append("No email")
    if not has_phone:
        issues.append("No phone")
    
    return max(score, 0.0), issues

def check_keyword_repetition(text: str, skills: List[str]) -> Tuple[int, str]:
    """Check for keyword stuffing"""
    for skill in skills[:10]:
        pattern = r'\b' + re.escape(skill) + r'\b'
        count = len(re.findall(pattern, text.lower(), re.IGNORECASE))
        if count > 15:
            return 5, f"'{skill}' repeated {count} times"
    return 0, ""

def calculate_ats_score(resume_text: str, job_description: str, parsed_resume: Dict) -> Dict:
    """
    CloudAlly Systems ATS Scoring
    
    Scoring: Skills(30) + Education(20) + Experience(20) + Projects(15) + Keywords(10) + Quality(5)
    
    Penalties:
    - Required skill missing: -2 each
    - Preferred skill missing: -1 each
    - Skill not in projects: -1 each
    - Keyword repetition: -5
    - Extra pages: -2 per page
    - Invalid CGPA: -2
    """
    score = 0.0
    feedback = []
    penalties = []
    breakdown = {}
    
    # Detect skills from resume
    resume_skills = detect_resume_skills(resume_text)
    resume_skills_lower = [s.lower() for s in resume_skills]
    
    # 1. Skills Match (30 points)
    if resume_skills:
        # Required skills check
        missing_required = []
        matched_required = []
        
        for req_skill in REQUIRED_SKILLS:
            if req_skill.lower() in resume_skills_lower:
                matched_required.append(req_skill)
            else:
                missing_required.append(req_skill)
        
        # Preferred skills check
        missing_preferred = []
        matched_preferred = []
        
        for pref_skill in PREFERRED_SKILLS:
            if pref_skill.lower() in resume_skills_lower:
                matched_preferred.append(pref_skill)
            else:
                missing_preferred.append(pref_skill)
        
        # Calculate base score
        total_required = len(REQUIRED_SKILLS)
        total_preferred = len(PREFERRED_SKILLS)
        
        required_score = (len(matched_required) / total_required) * 25 if total_required > 0 else 0
        preferred_score = (len(matched_preferred) / total_preferred) * 5 if total_preferred > 0 else 0
        
        skills_score = required_score + preferred_score
        score += skills_score
        breakdown['skills_match'] = round(skills_score, 2)
        
        feedback.append(f"Required: {len(matched_required)}/{total_required}, Preferred: {len(matched_preferred)}/{total_preferred}")
        
        # Penalties
        for skill in missing_required:
            penalties.append(f"Missing required: {skill} (-2)")
            score -= 2
        for skill in missing_preferred:
            penalties.append(f"Missing preferred: {skill} (-1)")
            score -= 1
    else:
        breakdown['skills_match'] = 0.0
        feedback.append("No skills detected")
        penalties.append("No skills found")
    
    # 2. Education (20 points)
    edu_score, edu_penalties = validate_education(parsed_resume['education_section'])
    score += edu_score
    breakdown['education'] = round(edu_score, 2)
    penalties.extend(edu_penalties)
    
    # 3. Experience (20 points)
    # CloudAlly requires 0-2 years
    exp_years = parsed_resume['experience_years']
    
    if 0 <= exp_years <= 2:
        exp_score = 20.0
    elif exp_years > 2:
        exp_score = 15.0  # Overqualified but still good
    else:
        exp_score = 0.0
    
    score += exp_score
    breakdown['experience'] = round(exp_score, 2)
    feedback.append(f"Experience: {exp_years} years")
    
    # 4. Skills in Projects (15 points)
    project_score, verified_skills, unverified_skills = validate_projects(
        parsed_resume['projects_section'],
        resume_skills
    )
    score += project_score
    breakdown['projects'] = round(project_score, 2)
    
    # Penalty: Skills not in projects (-1 each)
    for skill in unverified_skills:
        penalties.append(f"Skill not in projects: {skill} (-1)")
        score -= 1
    
    if verified_skills:
        feedback.append(f"Skills in projects: {len(verified_skills)}/{len(resume_skills)}")
    
    # 5. Keyword Relevance (10 points)
    keyword_result = calculate_keyword_relevance(resume_text, job_description)
    score += keyword_result['score']
    breakdown['keyword_relevance'] = round(keyword_result['score'], 2)
    feedback.append(f"Keyword match: {keyword_result['similarity']}%")
    
    # 6. Resume Quality (5 points)
    quality_score, quality_issues = assess_resume_quality(resume_text)
    score += quality_score
    breakdown['resume_quality'] = round(quality_score, 2)
    penalties.extend(quality_issues)
    
    # Check keyword repetition
    rep_penalty, rep_msg = check_keyword_repetition(resume_text, resume_skills)
    if rep_penalty > 0:
        penalties.append(f"Keyword stuffing: {rep_msg} (-{rep_penalty})")
        score -= rep_penalty
    
    final_score = max(0.0, min(score, 100.0))
    
    return {
        'score': round(final_score, 2),
        'breakdown': breakdown,
        'matched_skills': resume_skills,
        'feedback': feedback,
        'penalties': penalties,
        'keyword_similarity': keyword_result['similarity']
    }