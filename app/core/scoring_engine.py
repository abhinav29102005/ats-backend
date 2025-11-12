"""
ATS scoring engine - Updated scoring logic
Skills: 30, Education: 20, Experience: 20, Skills in Projects: 15, 
Keyword Relevance: 10, Resume Quality: 5

Replace the content of app/core/scoring_engine.py with this code
"""
import re
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

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
        final_score = combined_score * 10  # Out of 10 points
        
        return {
            'score': min(final_score, 10.0),
            'similarity': round(similarity * 100, 2),
            'matched_keywords': matched_keywords[:10],
            'match_rate': round(match_rate * 100, 2)
        }
    except Exception as e:
        logger.error(f"Keyword relevance error: {e}")
        return {'score': 0.0, 'similarity': 0.0, 'matched_keywords': [], 'match_rate': 0.0}

def extract_jd_skills(job_description: str) -> Tuple[List[str], List[str]]:
    """
    Extract required and preferred skills from job description
    Returns: (required_skills, preferred_skills)
    """
    jd_lower = job_description.lower()
    
    # Common technical skills
    all_skills = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php',
        'swift', 'kotlin', 'go', 'rust', 'sql', 'react', 'angular', 'vue',
        'node.js', 'django', 'flask', 'spring', 'express', 'fastapi',
        'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'ci/cd',
        'jenkins', 'mongodb', 'postgresql', 'mysql', 'redis',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
        'machine learning', 'deep learning', 'data science',
        'html', 'css', 'rest api', 'graphql', 'microservices', 'agile', 'scrum'
    ]
    
    required_skills = []
    preferred_skills = []
    
    # Look for required skills section
    required_section = re.search(r'(?:required|must have|essential).*?(?:preferred|nice to have|$)', 
                                 jd_lower, re.DOTALL)
    if required_section:
        required_text = required_section.group(0)
        required_skills = [skill for skill in all_skills if skill in required_text]
    
    # Look for preferred skills section
    preferred_section = re.search(r'(?:preferred|nice to have|desired).*?(?:$)', 
                                  jd_lower, re.DOTALL)
    if preferred_section:
        preferred_text = preferred_section.group(0)
        preferred_skills = [skill for skill in all_skills if skill in preferred_text]
    
    # If no sections found, treat all JD skills as required
    if not required_skills and not preferred_skills:
        required_skills = [skill for skill in all_skills if skill in jd_lower]
    
    return required_skills, preferred_skills

def validate_projects(projects_section: str, skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Validate projects demonstrate claimed skills (15 points)
    Returns: (score, verified_skills, unverified_skills)
    """
    if not projects_section or not skills:
        return 0.0, [], skills
    
    projects_lower = projects_section.lower()
    verified_skills = [s for s in skills if s.lower() in projects_lower]
    unverified_skills = [s for s in skills if s not in verified_skills]
    
    if not skills:
        return 0.0, [], []
    
    verification_rate = len(verified_skills) / len(skills)
    score = verification_rate * 15  # Out of 15 points
    
    return score, verified_skills, unverified_skills

def count_resume_pages(text: str) -> int:
    """
    Estimate number of pages based on text length
    Rough estimate: ~500 words per page
    """
    word_count = len(text.split())
    pages = max(1, round(word_count / 500))
    return pages

def validate_education(education_section: str, jd_education: str = "") -> Tuple[float, List[str]]:
    """
    Validate education section (20 points)
    CGPA validation: -2 if CGPA > 9.42 (invalid)
    """
    score = 0.0
    penalties = []
    
    if not education_section:
        penalties.append("No education section found (-0 points, no penalty)")
        return 0.0, penalties
    
    edu_lower = education_section.lower()
    jd_lower = jd_education.lower() if jd_education else ""
    
    # Degree matching (10 points)
    degrees = ['bachelor', 'b.tech', 'b.e.', 'bsc', 'master', 'm.tech', 'm.sc', 'phd', 'mba']
    jd_degrees = [d for d in degrees if d in jd_lower]
    resume_degrees = [d for d in degrees if d in edu_lower]
    
    if jd_degrees and resume_degrees:
        score += 10 if any(jd_deg in resume_degrees for jd_deg in jd_degrees) else 5
    elif resume_degrees:
        score += 7
    
    # CGPA validation (10 points with penalty check)
    cgpa_pattern = r'(?:cgpa|gpa|grade)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?'
    cgpa_matches = re.findall(cgpa_pattern, edu_lower)
    
    cgpa_valid = False
    for match in cgpa_matches:
        try:
            cgpa_value = float(match[0])
            max_scale = float(match[1]) if match[1] else 10.0
            
            # Check if CGPA is greater than 9.42 (invalid)
            if cgpa_value > 9.42 and max_scale == 10.0:
                penalties.append(f"Invalid CGPA: {cgpa_value}/10.0 (exceeds 9.42) (-2)")
                score -= 2
            elif cgpa_value > max_scale:
                penalties.append(f"Invalid CGPA: {cgpa_value}/{max_scale} (-2)")
                score -= 2
            elif cgpa_value >= max_scale * 0.6:
                score += 10
                cgpa_valid = True
        except (ValueError, ZeroDivisionError):
            continue
    
    if not cgpa_valid and not cgpa_matches:
        # No CGPA found, no points but no penalty
        pass
    
    return min(max(score, 0.0), 20.0), penalties

def assess_resume_quality(text: str) -> Tuple[float, List[str]]:
    """
    Assess resume format and quality (5 points)
    Penalty: -2 per extra page beyond 1 page
    """
    score = 5.0  # Start with full points
    issues = []
    
    # Page count penalty
    pages = count_resume_pages(text)
    if pages > 1:
        penalty = (pages - 1) * 2
        issues.append(f"Resume is {pages} pages (-{penalty} for {pages-1} extra page(s))")
        score -= penalty
    
    # Word count check (no penalty, just feedback)
    word_count = len(text.split())
    if word_count < 200:
        issues.append("Resume too brief (consider adding more details)")
    elif word_count > 1500:
        issues.append("Resume too verbose (consider condensing)")
    
    # Essential sections
    required = ['experience', 'education', 'skills']
    text_lower = text.lower()
    missing = [sec for sec in required if sec not in text_lower]
    
    if missing:
        issues.append(f"Missing section(s): {', '.join(missing)}")
    
    # Contact info
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'[\+]?[\d\s\-\(\)]{10,}', text))
    
    if not has_email:
        issues.append("No email found")
    if not has_phone:
        issues.append("No phone found")
    
    return max(score, 0.0), issues

def calculate_ats_score(resume_text: str, job_description: str, parsed_resume: Dict,
                       jd_education: str = "") -> Dict:
    """
    Main ATS scoring function
    
    Scoring breakdown:
    - Skills Match: 30 points
    - Education: 20 points
    - Experience: 20 points
    - Skills in Projects: 15 points
    - Keyword Relevance: 10 points
    - Resume Quality: 5 points
    Total: 100 points
    
    Penalties:
    - Missing preferred skills: -2 per skill
    - Required skills not used in projects: -2 per skill
    - Invalid CGPA (>9.42): -2
    - Missing parameters: 0 points (no penalty, just no points awarded)
    - Extra pages: -2 per page beyond 1
    """
    score = 0.0
    feedback = []
    penalties = []
    breakdown = {}
    
    jd_lower = job_description.lower()
    
    # Extract required and preferred skills from JD
    required_jd_skills, preferred_jd_skills = extract_jd_skills(job_description)
    
    # 1. Skills Match (30 points)
    resume_skills = parsed_resume['skills']
    
    if resume_skills:
        # Match resume skills with JD skills
        all_jd_skills = list(set(required_jd_skills + preferred_jd_skills))
        matched_skills = [s for s in resume_skills if s.lower() in jd_lower or 
                         any(s.lower() == jd_skill.lower() for jd_skill in all_jd_skills)]
        
        if all_jd_skills:
            skills_score = (len(matched_skills) / len(all_jd_skills)) * 30
        else:
            skills_score = 15.0  # Default if no clear JD skills
        
        score += skills_score
        breakdown['skills_match'] = round(skills_score, 2)
        feedback.append(f"Matched {len(matched_skills)}/{len(all_jd_skills) if all_jd_skills else len(resume_skills)} skills")
        
        # Penalty: Missing preferred skills (-2 each)
        missing_preferred = [s for s in preferred_jd_skills if s not in [rs.lower() for rs in resume_skills]]
        if missing_preferred:
            penalty = len(missing_preferred) * 2
            penalties.append(f"Missing preferred skills: {', '.join(missing_preferred)} (-{penalty})")
            score -= penalty
    else:
        breakdown['skills_match'] = 0.0
        feedback.append("No skills detected")
        penalties.append("No skills found in resume")
    
    # 2. Education (20 points)
    education_score, edu_penalties = validate_education(parsed_resume['education_section'], jd_education)
    score += education_score
    breakdown['education'] = round(education_score, 2)
    penalties.extend(edu_penalties)
    
    # 3. Experience (20 points)
    exp_match = re.search(r'(\d+)\+?\s*years?', jd_lower)
    required_exp = int(exp_match.group(1)) if exp_match else 2
    
    if parsed_resume['experience_years'] > 0:
        exp_score = 20.0 if parsed_resume['experience_years'] >= required_exp else \
                    (parsed_resume['experience_years'] / required_exp) * 20
        score += exp_score
        breakdown['experience'] = round(exp_score, 2)
        feedback.append(f"Experience: {parsed_resume['experience_years']} years")
    else:
        breakdown['experience'] = 0.0
        feedback.append("No experience found")
        penalties.append("No experience section found")
    
    # 4. Skills in Projects (15 points)
    project_score, verified_skills, unverified_skills = validate_projects(
        parsed_resume['projects_section'],
        resume_skills
    )
    score += project_score
    breakdown['projects'] = round(project_score, 2)
    
    # Penalty: Required skills not used in projects (-2 each)
    required_but_not_in_projects = [s for s in required_jd_skills 
                                    if s in [rs.lower() for rs in resume_skills] 
                                    and s not in [vs.lower() for vs in verified_skills]]
    if required_but_not_in_projects:
        penalty = len(required_but_not_in_projects) * 2
        penalties.append(f"Required skills not demonstrated in projects: {', '.join(required_but_not_in_projects)} (-{penalty})")
        score -= penalty
    
    if verified_skills:
        feedback.append(f"Skills verified in projects: {len(verified_skills)}/{len(resume_skills)}")
    
    # 5. Keyword Relevance (10 points)
    keyword_result = calculate_keyword_relevance(resume_text, job_description)
    score += keyword_result['score']
    breakdown['keyword_relevance'] = round(keyword_result['score'], 2)
    feedback.append(f"Keyword similarity: {keyword_result['similarity']}%")
    
    # 6. Resume Quality (5 points)
    quality_score, quality_issues = assess_resume_quality(resume_text)
    score += quality_score
    breakdown['resume_quality'] = round(quality_score, 2)
    penalties.extend(quality_issues)
    
    # Keyword stuffing check
    for skill in resume_skills[:5]:
        count = resume_text.lower().count(skill.lower())
        if count > 15:
            penalties.append(f"Keyword stuffing: '{skill}' repeated {count} times (-5)")
            score -= 5
            break
    
    final_score = max(0.0, min(score, 100.0))
    
    return {
        'score': round(final_score, 2),
        'breakdown': breakdown,
        'matched_skills': [s for s in resume_skills if s.lower() in jd_lower],
        'feedback': feedback,
        'penalties': penalties,
        'keyword_similarity': keyword_result['similarity']
    }