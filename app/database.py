"""
Database operations and Supabase client
"""
import logging
from typing import Optional, List, Dict
from supabase import create_client, Client
import pandas as pd
from app.config import settings

logger = logging.getLogger(__name__)

class Database:
    """Database handler for Supabase operations"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.connect()
    
    def connect(self):
        """Initialize Supabase connection"""
        try:
            if settings.SUPABASE_URL and settings.SUPABASE_KEY:
                self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                # Test connection
                self.client.table('participants').select("count", count='exact').limit(1).execute()
                logger.info("✅ Database connected successfully")
            else:
                logger.error("❌ Supabase credentials not configured")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            self.client = None
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.client is not None
    
    def register_participant(self, data: Dict) -> str:
        """Register new participant"""
        if not self.client:
            raise Exception("Database not connected")
        
        try:
            response = self.client.table('participants').insert(data).execute()
            return data['id']
        except Exception as e:
            logger.error(f"Registration error: {e}")
            raise Exception(f"Failed to register participant: {str(e)}")
    
    def get_participant_by_email(self, email: str) -> Optional[Dict]:
        """Get participant by email"""
        if not self.client:
            return None
        
        try:
            response = self.client.table('participants').select('*').eq('email', email).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Get participant error: {e}")
            return None
    
    def save_application(self, data: Dict) -> bool:
        """Save application submission"""
        if not self.client:
            return False
        
        try:
            self.client.table('applications').insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Save application error: {e}")
            raise Exception(f"Failed to save application: {str(e)}")
    
    def get_upload_count(self, participant_id: str) -> int:
        """Get participant upload count"""
        if not self.client:
            return 0
        
        try:
            response = self.client.table('applications').select('id', count='exact').eq('participant_id', participant_id).execute()
            return response.count if response.count else 0
        except Exception as e:
            logger.error(f"Get upload count error: {e}")
            return 0
    
    def get_participant_scores(self, participant_id: str) -> pd.DataFrame:
        """Get all scores for participant"""
        if not self.client:
            return pd.DataFrame()
        
        try:
            response = self.client.table('applications').select('*').eq('participant_id', participant_id).order('created_at', desc=True).execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            logger.error(f"Get scores error: {e}")
            return pd.DataFrame()
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top leaderboard entries"""
        if not self.client:
            return []
        
        try:
            response = self.client.table('leaderboard').select('*').limit(limit).execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Get leaderboard error: {e}")
            return []
    
    def get_statistics(self) -> Optional[Dict]:
        """Get competition statistics"""
        if not self.client:
            return None
        
        try:
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
        except Exception as e:
            logger.error(f"Get statistics error: {e}")
            return None
    
    def save_to_corpus(self, participant_id: str, resume_text: str):
        """Save resume to corpus for plagiarism checking"""
        if not self.client:
            return
        
        try:
            data = {
                'participant_id': participant_id,
                'resume_text': resume_text
            }
            self.client.table('resume_corpus').insert(data).execute()
        except Exception as e:
            logger.error(f"Save to corpus error: {e}")
    
    def get_reference_corpus(self, limit: int = 100) -> List[str]:
        """Get reference corpus for plagiarism checking"""
        if not self.client:
            return []
        
        try:
            response = self.client.table('resume_corpus').select('resume_text').limit(limit).execute()
            return [item['resume_text'] for item in response.data] if response.data else []
        except Exception as e:
            logger.error(f"Get corpus error: {e}")
            return []

# Global database instance
db = Database()
