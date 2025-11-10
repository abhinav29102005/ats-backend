"""
Plagiarism detection using TF-IDF similarity
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def check_plagiarism(resume_text: str, reference_corpus: List[str]) -> Tuple[float, str]:
    """
    Check plagiarism against reference corpus
    
    Args:
        resume_text: Resume text to check
        reference_corpus: List of reference resume texts
        
    Returns:
        Tuple of (plagiarism_score, status)
    """
    if not reference_corpus or len(reference_corpus) == 0:
        return 0.0, "No reference data"
    
    try:
        corpus = [resume_text] + reference_corpus
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        max_similarity = float(np.max(similarities)) if similarities.size > 0 else 0.0
        plagiarism_score = round(max_similarity * 100, 2)
        
        return plagiarism_score, "Checked"
        
    except Exception as e:
        logger.error(f"Plagiarism check error: {e}")
        return 0.0, f"Error: {str(e)}"

def calculate_plagiarism_penalty(plagiarism_score: float) -> Tuple[float, str]:
    """
    Calculate penalty based on plagiarism score
    
    Args:
        plagiarism_score: Plagiarism score (0-100)
        
    Returns:
        Tuple of (penalty_points, penalty_message)
    """
    if plagiarism_score > 80:
        return 20.0, f"Critical plagiarism: {plagiarism_score}% (-20)"
    elif plagiarism_score > 60:
        return 10.0, f"High plagiarism: {plagiarism_score}% (-10)"
    elif plagiarism_score > 40:
        return 5.0, f"Moderate plagiarism: {plagiarism_score}% (-5)"
    else:
        return 0.0, ""
