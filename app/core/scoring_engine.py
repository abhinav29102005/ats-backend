"""
ATS Scoring Engine - CloudAlly Systems Fixed JD
FIXED: Uses hardcoded CloudAlly skill lists with proper deduplication
Version: 2.0 - FINAL
"""
import re
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# ============================================================
# FIXED CLOUDALLY SYSTEMS SKILL LISTS - CLEANED (NO DUPLICATES)
# ============================================================

# Required Skills (14 UNIQUE) - Penalty: -2 per missing skill
REQUIRED_SKILLS = [
    'python',
    'node',              # Matches: node.js, nodejs, node
    'typescript',
    'fastapi',
    'flask',
    'postgresql',        # Matches: postgresql, postgres
    'rest api',          # Matches: rest api, restapi, restful api
    'docker',
    'git',
    'ci/cd',             # Matches: ci/cd, cicd
    'aws',
    'azure',
    'microservices',     # Matches: microservices, microservice
    'async',             # Matches: async, asynchronous, asynchronous programming
]

# Preferred Skills (8 UNIQUE) - Penalty: -1 per missing skill
PREFERRED_SKILLS = [
    'machine learning',  # Matches: machine learning, ml
    'mlops',
    'react',             # Matches: react, reactjs
    'next',              # Matches: next.js, nextjs, next
    'agile',
    'scrum',
    'grafana',
    'prometheus',
]

# Total: 22 unique skills (14 required + 8 preferred)

# Fixed JD for CloudAlly Systems
CLOUDALLY_JD = """
CloudAlly Systems Pvt. Ltd. Cloud & Automation Systems Software Engineer Entry-Mid 0-2 years
Bengaluru India cloud automation intelligent analytics scalable enterprise platforms AI data pipelines
real-time automation global clients engineering culture innovation autonomy reliable systems scale
passionate Software Engineer Cloud Automation division backend development distributed systems
automating business workflows data-driven engineering design develop deploy scalable microservices
Python FastAPI Flask Node.js implement optimize maintain RESTful APIs backend logic automation workflows
work cloud platforms AWS Azure manage CI/CD pipelines monitoring tools deployment infrastructure
integrate machine learning models production systems basic exposure collaborate frontend developers
deliver complete maintainable web applications React Next.js ensure system reliability testing
containerization Docker observability setups Grafana Prometheus contribute code reviews architecture
discussions documentation best practices strong proficiency Python Node.js TypeScript experience
FastAPI Flask PostgreSQL REST API design principles knowledge Docker Git CI/CD pipelines GitHub Actions
Jenkins familiarity AWS ECS Lambda S3 Microsoft Azure deployment workflows understanding microservice
architecture data-driven systems asynchronous programming problem-solving debugging performance
optimization skills exposure machine learning pipelines MLOps basic AI model integration knowledge
React Next.js frontend frameworks prior experience developing contributing open-source projects
strong documentation habits Agile Scrum practices work real-time data systems large-scale automation
workloads collaborate interdisciplinary team backend cloud AI engineers opportunity learn deployment
automation cloud infrastructure production-scale systems supportive environment mentorship upskilling
sessions quarterly project showcases
"""


def skill_matches(resume_skill: str, jd_skill: str) -> bool:
    """
    Check if resume skill matches JD skill (flexible matching with aliases)
    """
    resume_lower = resume_skill.lower()
    jd_lower = jd_skill.lower()
    
    # Exact match
    if resume_lower == jd_lower:
        return True
    
    # Skill aliases for flexible matching
    skill_aliases = {
        'node': ['node.js', 'nodejs', 'node js', 'node'],
        'postgresql': ['postgres', 'postgresql', 'psql'],
        'rest api': ['restapi', 'rest api', 'restful api', 'rest', 'restful'],
        'ci/cd': ['cicd', 'ci/cd', 'ci cd', 'ci-cd'],
        'microservices': ['microservice', 'microservices', 'micro services', 'micro-services'],
        'async': ['asynchronous', 'async', 'asynchronous programming', 'asyncio'],
        'machine learning': ['ml', 'machine learning', 'machine-learning'],
        'react': ['reactjs', 'react', 'react.js'],
        'next': ['nextjs', 'next.js', 'next js', 'next'],
    }
    
    # Check if JD skill has aliases
    for base_skill, aliases in skill_aliases.items():
        if jd_lower == base_skill or jd_lower in aliases:
            # Check if resume skill matches any alias
            if any(alias in resume_lower or resume_lower in alias for alias in aliases):
                return True
    
    # Substring match (fallback)
    if jd_lower in resume_lower or resume_lower in jd_lower:
        return True
    
    return False


def calculate_keyword_relevance(resume_text: str, job_description: str) -> Dict:
    """Calculate TF-IDF keyword relevance (10 points)"""
    try:
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Use CloudAlly JD instead of provided JD for consistency
        tfidf_matrix = vectorizer.fit_transform([CLOUDALLY_JD, resume_text])
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


def validate_projects(projects_section: str, skills: List[str], required_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """Validate projects (15 points)"""
    if not projects_section or not skills:
        return 0.0, [], required_skills
    
    projects_lower = projects_section.lower()
    verified_skills = [s for s in skills if s.lower() in projects_lower]
    
    if not skills:
        return 0.0, [], []
    
    verification_rate = len(verified_skills) / len(skills)
    score = verification_rate * 15
    
    resume_skills_lower = [s.lower() for s in skills]
    verified_skills_lower = [s.lower() for s in verified_skills]
    
    unverified_required = [
        req_skill for req_skill in required_skills
        if req_skill in resume_skills_lower
        and req_skill not in verified_skills_lower
    ]
    
    return score, verified_skills, unverified_required


def count_resume_pages(text: str) -> int:
    """Estimate pages (500 words per page)"""
    word_count = len(text.split())
    pages = max(1, round(word_count / 500))
    return pages


def validate_education(education_section: str) -> Tuple[float, List[str], float]:
    """Validate education (20 points)"""
    score = 0.0
    penalties = []
    cgpa_value = 0.0
    
    if not education_section:
        penalties.append("No education section found")
        return 0.0, penalties, 0.0
    
    edu_lower = education_section.lower()
    
    # Degree matching (15 points)
    cs_degrees = ['computer science', 'software engineering', 'information technology', 'computer engineering']
    general_degrees = ['bachelor', 'b.tech', 'b.e.', 'bsc', 'b.sc']
    
    has_cs_degree = any(deg in edu_lower for deg in cs_degrees)
    has_bachelor = any(deg in edu_lower for deg in general_degrees)
    
    if has_cs_degree and has_bachelor:
        score += 15
    elif has_bachelor:
        score += 10
    elif any(d in edu_lower for d in ['master', 'm.tech', 'm.sc', 'phd', 'mba']):
        score += 7
    else:
        penalties.append("No clear degree found")
    
    # CGPA (5 points)
    cgpa_pattern = r'(?:cgpa|gpa|grade)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?'
    cgpa_matches = re.findall(cgpa_pattern, edu_lower)
    
    for match in cgpa_matches:
        try:
            cgpa_value = float(match[0])
            max_scale = float(match[1]) if match[1] else 10.0
            
            if cgpa_value > 9.42 and max_scale == 10.0:
                penalties.append(f"Invalid CGPA: {cgpa_value}/10.0 (-2)")
                score -= 2
            elif cgpa_value > max_scale:
                penalties.append(f"Invalid CGPA: {cgpa_value}/{max_scale} (-2)")
                score -= 2
            elif cgpa_value >= max_scale * 0.70:
                score += 5
        except (ValueError, ZeroDivisionError):
            continue
    
    return min(max(score, 0.0), 20.0), penalties, cgpa_value


def assess_resume_quality(text: str) -> Tuple[float, List[str]]:
    """Assess quality (5 points)"""
    score = 5.0
    issues = []
    
    pages = count_resume_pages(text)
    if pages > 1:
        penalty = (pages - 1) * 2
        issues.append(f"Resume is {pages} pages (-{penalty})")
        score -= penalty
    
    required = ['experience', 'education', 'skills']
    text_lower = text.lower()
    missing = [sec for sec in required if sec not in text_lower]
    
    if missing:
        issues.append(f"Missing: {', '.join(missing)}")
    
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'[\+]?[\d\s\-\(\)]{10,}', text))
    
    if not has_email:
        issues.append("No email")
    if not has_phone:
        issues.append("No phone")
    
    return max(score, 0.0), issues


def calculate_ats_score(resume_text: str, job_description: str, parsed_resume: Dict,
                       jd_education: str = "", required_skills: List[str] = None,
                       preferred_skills: List[str] = None) -> Dict:
    """
    CloudAlly Systems ATS Scoring - FIXED v2.0
    Uses hardcoded skill lists (22 unique skills) with flexible matching
    """
    
    # Use CloudAlly fixed skills (ignore parameters)
    req_skills = REQUIRED_SKILLS
    pref_skills = PREFERRED_SKILLS
    
    score = 0.0
    feedback = []
    penalties = []
    breakdown = {}
    
    resume_skills = parsed_resume.get('skills', [])
    resume_skills_lower = [s.lower() for s in resume_skills]
    
    # ========== SKILLS MATCH (30 points) ==========
    all_jd_skills = req_skills + pref_skills  # 22 unique skills
    matched_skills = []
    matched_jd_skills = set()
    
    # Match each resume skill against all JD skills using flexible matching
    for skill in resume_skills:
        for jd_skill in all_jd_skills:
            if skill_matches(skill, jd_skill):
                if jd_skill not in matched_jd_skills:
                    matched_skills.append(skill)
                    matched_jd_skills.add(jd_skill)
                break
    
    if all_jd_skills:
        skills_score = (len(matched_jd_skills) / len(all_jd_skills)) * 30
    else:
        skills_score = 0.0
    
    score += skills_score
    breakdown['skills_match'] = round(skills_score, 2)
    feedback.append(f"Matched {len(matched_jd_skills)}/{len(all_jd_skills)} skills")
    
    # Check missing required (-2 each)
    missing_required = []
    for req_skill in req_skills:
        found = False
        for resume_skill in resume_skills:
            if skill_matches(resume_skill, req_skill):
                found = True
                break
        if not found:
            missing_required.append(req_skill)
    
    if missing_required:
        penalty = len(missing_required) * 2
        penalties.append(f"Missing required: {', '.join(missing_required[:5])} (-{penalty})")
        score -= penalty
    
    # Check missing preferred (-1 each)
    missing_preferred = []
    for pref_skill in pref_skills:
        found = False
        for resume_skill in resume_skills:
            if skill_matches(resume_skill, pref_skill):
                found = True
                break
        if not found:
            missing_preferred.append(pref_skill)
    
    if missing_preferred:
        penalty = len(missing_preferred) * 1
        penalties.append(f"Missing preferred: {', '.join(missing_preferred[:5])} (-{penalty})")
        score -= penalty
    
    # ========== EDUCATION (20 points) ==========
    education_score, edu_penalties, cgpa = validate_education(parsed_resume.get('education_section', ''))
    score += education_score
    breakdown['education'] = round(education_score, 2)
    penalties.extend(edu_penalties)
    
    if education_score >= 15:
        feedback.append("Education: Excellent match")
    elif education_score >= 10:
        feedback.append("Education: Good match")
    
    # ========== EXPERIENCE (20 points) ==========
    exp_years = parsed_resume.get('experience_years', 0)
    
    if 0 <= exp_years <= 2:
        exp_score = 20.0
        feedback.append(f"Experience: {exp_years} years ")
    elif 2 < exp_years <= 5:
        exp_score = 15.0
        feedback.append(f"Experience: {exp_years} years ")
    elif exp_years > 5:
        exp_score = 10.0
        feedback.append(f"Experience: {exp_years} years ")
    else:
        exp_score = 0.0
        feedback.append("Experience: None detected")
    
    score += exp_score
    breakdown['experience'] = round(exp_score, 2)
    
    # ========== PROJECTS (15 points) ==========
    project_score, verified_skills, unverified_required = validate_projects(
        parsed_resume.get('projects_section', ''),
        resume_skills,
        req_skills
    )
    
    score += project_score
    breakdown['projects'] = round(project_score, 2)
    
    if unverified_required:
        penalty = len(set(unverified_required)) * 2
        penalties.append(f"Skills not in projects: {', '.join(list(set(unverified_required))[:3])} (-{penalty})")
        score -= penalty
    
    if verified_skills:
        feedback.append(f"Skills in projects: {len(verified_skills)}/{len(resume_skills)}")
    
    # ========== KEYWORDS (10 points) ==========
    keyword_result = calculate_keyword_relevance(resume_text, CLOUDALLY_JD)
    score += keyword_result['score']
    breakdown['keyword_relevance'] = round(keyword_result['score'], 2)
    feedback.append(f"Keyword similarity: {keyword_result['similarity']}%")
    
    # ========== QUALITY (5 points) ==========
    quality_score, quality_issues = assess_resume_quality(resume_text)
    score += quality_score
    breakdown['resume_quality'] = round(quality_score, 2)
    penalties.extend(quality_issues)
    
    # Keyword stuffing check
    for skill in resume_skills[:5]:
        count = resume_text.lower().count(skill.lower())
        if count > 15:
            penalties.append(f"Keyword stuffing: '{skill}' ({count} times) (-5)")
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
