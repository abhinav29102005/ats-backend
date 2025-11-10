# MLSC Perfect CV Match 2025 - ATS Scorer

Advanced ATS (Applicant Tracking System) Resume Scoring Platform for Microsoft Learn Student Chapter @ TIET.

## 🚀 Features

- **Smart Resume Parsing**: Extract skills, experience, education, and projects
- **TF-IDF Keyword Matching**: Semantic similarity with job descriptions
- **Plagiarism Detection**: Compare against corpus of submitted resumes
- **Detailed Scoring**: 6-category breakdown with feedback
- **Rate Limiting**: Prevent abuse with upload limits
- **Leaderboard**: Real-time competition standings
- **Statistics Dashboard**: Track competition metrics

## 📊 Scoring System

- **Skills Match** (35 points): Keyword matching with JD
- **Experience** (25 points): Years + relevance
- **Keyword Relevance** (15 points): TF-IDF similarity
- **Education** (10 points): Degree + field matching
- **Resume Quality** (10 points): Format + structure
- **Projects** (5 points): Skill verification

## 🛠️ Tech Stack

- **FastAPI**: High-performance Python web framework
- **Supabase**: PostgreSQL database
- **spaCy**: NLP for resume parsing
- **Scikit-learn**: TF-IDF and ML algorithms
- **Railway**: Deployment platform

## 📁 Project Structure

