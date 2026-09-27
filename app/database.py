"""Database operations"""
import logging
from typing import Optional, List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from app.config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        try:
            if settings.DATABASE_URL:
                self.conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
                self.conn.autocommit = True
                logger.info("✅ Database connected via local PostgreSQL")
            else:
                logger.error("❌ DATABASE_URL not configured")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            self.conn = None
            
    def _execute(self, query, params=None, fetch=None):
        if not self.conn or self.conn.closed:
            self.connect()
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                if fetch == 'all':
                    return cur.fetchall()
                elif fetch == 'one':
                    return cur.fetchone()
                return None
        except Exception as e:
            logger.error(f"Query error: {e}")
            return None
    
    @property
    def is_connected(self):
        return self.conn is not None and not self.conn.closed
    
    def register_participant(self, data: Dict) -> str:
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO participants ({columns}) VALUES ({placeholders}) RETURNING id"
        res = self._execute(query, tuple(data.values()), fetch='one')
        return str(res['id']) if res else data['id']
    
    def get_participant_by_email(self, email: str) -> Optional[Dict]:
        query = "SELECT * FROM participants WHERE email = %s"
        return self._execute(query, (email,), fetch='one')
    
    def save_application(self, data: Dict):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO applications ({columns}) VALUES ({placeholders})"
        self._execute(query, tuple(data.values()))
    
    def get_upload_count(self, participant_id: str) -> int:
        query = "SELECT COUNT(id) as count FROM applications WHERE participant_id = %s"
        res = self._execute(query, (participant_id,), fetch='one')
        return res['count'] if res else 0
    
    def get_participant_scores(self, participant_id: str):
        query = "SELECT * FROM applications WHERE participant_id = %s ORDER BY created_at DESC"
        data = self._execute(query, (participant_id,), fetch='all')
        return pd.DataFrame(data) if data else pd.DataFrame()
    
    def get_leaderboard(self, limit: int = 10):
        query = "SELECT * FROM leaderboard LIMIT %s"
        data = self._execute(query, (limit,), fetch='all')
        return data if data else []
    
    def get_statistics(self):
        apps_query = "SELECT score, experience_years FROM applications"
        apps_data = self._execute(apps_query, fetch='all')
        
        part_query = "SELECT COUNT(id) as count FROM participants"
        part_res = self._execute(part_query, fetch='one')
        
        if not apps_data:
            return None
            
        df = pd.DataFrame(apps_data)
        return {
            'total_participants': part_res['count'] if part_res else 0,
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
    
    def save_to_corpus(self, participant_id: str, resume_text: str):
        query = "INSERT INTO resume_corpus (participant_id, resume_text) VALUES (%s, %s)"
        self._execute(query, (participant_id, resume_text))
    
    def get_reference_corpus(self, limit: int = 100):
        query = "SELECT resume_text FROM resume_corpus LIMIT %s"
        data = self._execute(query, (limit,), fetch='all')
        return [item['resume_text'] for item in data] if data else []

db = Database()
