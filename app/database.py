"""Database operations"""
import logging
from typing import Optional, List, Dict
from supabase import create_client, Client
import pandas as pd
from app.config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client: Optional[Client] = None
        self.connect()
    
    def connect(self):
        try:
            if settings.SUPABASE_URL and settings.SUPABASE_KEY:
                self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                self.client.table('participants').select("count", count='exact').limit(1).execute()
                logger.info("✅ Database connected")
            else:
                logger.error("❌ Supabase credentials not configured")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            self.client = None
    
    @property
    def is_connected(self):
        return self.client is not None
    
    def register_participant(self, data: Dict) -> str:
        self.client.table('participants').insert(data).execute()
        return data['id']
    
    def get_participant_by_email(self, email: str) -> Optional[Dict]:
        response = self.client.table('participants').select('*').eq('email', email).execute()
        return response.data[0] if response.data else None
    
    def save_application(self, data: Dict):
        self.client.table('applications').insert(data).execute()
    
    def get_upload_count(self, participant_id: str) -> int:
        response = self.client.table('applications').select('id', count='exact').eq('participant_id', participant_id).execute()
        return response.count if response.count else 0
    
    def get_participant_scores(self, participant_id: str):
        response = self.client.table('applications').select('*').eq('participant_id', participant_id).order('created_at', desc=True).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    
    def get_leaderboard(self, limit: int = 10):
        response = self.client.table('leaderboard').select('*').limit(limit).execute()
        return response.data if response.data else []
    
    def get_statistics(self):
        apps_response = self.client.table('applications').select('score, experience_years').execute()
        participants_response = self.client.table('participants').select('id', count='exact').execute()
        
        if not apps_response.data:
            return None
        
        df = pd.DataFrame(apps_response.data)
        return {
            'total_participants': participants_response.count or 0,
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
        data = {'participant_id': participant_id, 'resume_text': resume_text}
        self.client.table('resume_corpus').insert(data).execute()
    
    def get_reference_corpus(self, limit: int = 100):
        response = self.client.table('resume_corpus').select('resume_text').limit(limit).execute()
        return [item['resume_text'] for item in response.data] if response.data else []

db = Database()
