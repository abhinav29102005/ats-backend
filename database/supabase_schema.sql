-- ============================================
-- MLSC Perfect CV Match 2025 - Database Schema
-- Supabase PostgreSQL Database
-- ============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- TABLES
-- ============================================

-- Participants table
CREATE TABLE IF NOT EXISTS participants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    email VARCHAR(200) UNIQUE NOT NULL,
    mobile VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Applications table
CREATE TABLE IF NOT EXISTS applications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    participant_id UUID NOT NULL REFERENCES participants(id) ON DELETE CASCADE,
    score DECIMAL(5,2) NOT NULL CHECK (score >= 0 AND score <= 100),
    skills_count INTEGER DEFAULT 0,
    experience_years DECIMAL(4,1) DEFAULT 0,
    matched_skills_count INTEGER DEFAULT 0,
    plagiarism_score DECIMAL(5,2) DEFAULT 0,
    keyword_similarity DECIMAL(5,2) DEFAULT 0,
    resume_quality_score DECIMAL(4,2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Resume corpus table (for plagiarism detection)
CREATE TABLE IF NOT EXISTS resume_corpus (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    participant_id UUID REFERENCES participants(id) ON DELETE CASCADE,
    resume_text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_applications_participant_id ON applications(participant_id);
CREATE INDEX IF NOT EXISTS idx_applications_score_desc ON applications(score DESC);
CREATE INDEX IF NOT EXISTS idx_applications_created_at ON applications(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_participants_email ON participants(email);
CREATE INDEX IF NOT EXISTS idx_resume_corpus_participant_id ON resume_corpus(participant_id);
CREATE INDEX IF NOT EXISTS idx_applications_score_participant ON applications(participant_id, score DESC);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Auto-update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- TRIGGERS
-- ============================================

-- Auto-update participants.updated_at
DROP TRIGGER IF EXISTS update_participants_updated_at ON participants;
CREATE TRIGGER update_participants_updated_at 
    BEFORE UPDATE ON participants 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- VIEWS
-- ============================================

-- Leaderboard view (top 100 participants by best score)
CREATE OR REPLACE VIEW leaderboard AS
SELECT 
    ROW_NUMBER() OVER (ORDER BY a.score DESC, a.created_at ASC) as rank,
    p.email,
    p.name,
    a.score,
    a.skills_count,
    a.experience_years,
    a.matched_skills_count,
    a.created_at
FROM (
    SELECT DISTINCT ON (participant_id) 
        participant_id,
        score,
        skills_count,
        experience_years,
        matched_skills_count,
        created_at
    FROM applications
    ORDER BY participant_id, score DESC, created_at DESC
) a
JOIN participants p ON p.id = a.participant_id
ORDER BY a.score DESC, a.created_at ASC
LIMIT 100;

-- ============================================
-- ROW LEVEL SECURITY (RLS) - OPTIONAL
-- ============================================
-- Uncomment if you want to enable RLS for additional security

-- ALTER TABLE participants ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE applications ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE resume_corpus ENABLE ROW LEVEL SECURITY;

-- Allow service role to bypass RLS
-- CREATE POLICY "Service role can do everything on participants" 
--     ON participants FOR ALL 
--     TO service_role 
--     USING (true) 
--     WITH CHECK (true);

-- CREATE POLICY "Service role can do everything on applications" 
--     ON applications FOR ALL 
--     TO service_role 
--     USING (true) 
--     WITH CHECK (true);

-- CREATE POLICY "Service role can do everything on resume_corpus" 
--     ON resume_corpus FOR ALL 
--     TO service_role 
--     USING (true) 
--     WITH CHECK (true);

-- ============================================
-- COMMENTS
-- ============================================

COMMENT ON TABLE participants IS 'Stores participant registration information';
COMMENT ON TABLE applications IS 'Stores ATS scoring results for each resume submission';
COMMENT ON TABLE resume_corpus IS 'Stores resume text for plagiarism detection';
COMMENT ON VIEW leaderboard IS 'Top 100 participants ranked by their best score';

-- ============================================
-- VERIFICATION QUERIES
-- ============================================

-- Run these queries to verify your setup:

-- 1. Check all tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- 2. Check all indexes exist
SELECT indexname, tablename 
FROM pg_indexes 
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- 3. Check view exists
SELECT table_name 
FROM information_schema.views 
WHERE table_schema = 'public';

-- 4. Test insert (optional - remove after testing)
-- INSERT INTO participants (name, email, mobile) 
-- VALUES ('Test User', 'test@example.com', '1234567890');

-- 5. View table structures
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'public' 
  AND table_name = 'participants'
ORDER BY ordinal_position;