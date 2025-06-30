"""
Complete ZeroVault Database Module
Production-ready database functions for Supabase integration
No SQLAlchemy dependency - uses Supabase directly
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Generator, Union
from datetime import datetime
import uuid
import json

# Import your existing Supabase service
try:
    from .services.supabase_service import supabase_service
except ImportError:
    from services.supabase_service import supabase_service

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Complete database manager for ZeroVault
    Handles all database operations using Supabase
    """
    
    def __init__(self):
        self.supabase = supabase_service
        self.connection_pool = {}
        
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            return await self.supabase.health_check()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def initialize_database(self) -> bool:
        """Initialize database with required tables"""
        try:
            # Check if main tables exist
            tables_to_check = [
                'profiles', 'llm_scans', 'vulnerabilities', 
                'test_results', 'scan_progress', 'security_audit_logs'
            ]
            
            for table in tables_to_check:
                exists = await self.supabase.check_table_exists(table)
                if exists:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    logger.warning(f"⚠️ Table '{table}' missing")
            
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    # Scan Management Functions
    async def create_scan(self, scan_data: Dict[str, Any]) -> Optional[str]:
        """Create a new scan record"""
        try:
            scan_record = {
                'id': str(uuid.uuid4()),
                'company_id': scan_data.get('company_id'),
                'user_id': scan_data.get('user_id', scan_data.get('company_id')),
                'llm_name': scan_data.get('llm_name'),
                'endpoint': scan_data.get('endpoint'),
                'api_key': scan_data.get('api_key', ''),
                'model_type': scan_data.get('model_type', 'unknown'),
                'model_name': scan_data.get('model_name'),
                'description': scan_data.get('description'),
                'testing_scope': scan_data.get('testing_scope', 'basic'),
                'status': scan_data.get('status', 'queued'),
                'progress': scan_data.get('progress', 0.0),
                'risk_score': scan_data.get('risk_score', 0.0),
                'compliance_score': scan_data.get('compliance_score', 100.0),
                'vulnerability_count': scan_data.get('vulnerability_count', 0),
                'authenticity_verified': scan_data.get('authenticity_verified', False),
                'real_ai_testing': scan_data.get('real_ai_testing', False),
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.client.table('llm_scans').insert(scan_record).execute()
            if result.data:
                logger.info(f"Created scan: {scan_record['id']}")
                return scan_record['id']
            return None
            
        except Exception as e:
            logger.error(f"Failed to create scan: {e}")
            return None
    
    async def get_scan_by_id(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get scan by ID"""
        try:
            result = self.supabase.client.table('llm_scans').select('*').eq('id', scan_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get scan {scan_id}: {e}")
            return None
    
    async def update_scan(self, scan_id: str, update_data: Dict[str, Any]) -> bool:
        """Update scan record"""
        try:
            update_data['updated_at'] = datetime.utcnow().isoformat()
            
            result = self.supabase.client.table('llm_scans').update(update_data).eq('id', scan_id).execute()
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Failed to update scan {scan_id}: {e}")
            return False
    
    async def get_scans_by_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get scans for a user"""
        try:
            result = self.supabase.client.table('llm_scans').select('*').eq('user_id', user_id).limit(limit).order('created_at', desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get scans for user {user_id}: {e}")
            return []
    
    # Vulnerability Management Functions
    async def create_vulnerability(self, vuln_data: Dict[str, Any]) -> Optional[str]:
        """Create vulnerability record"""
        try:
            vuln_record = {
                'id': str(uuid.uuid4()),
                'scan_id': vuln_data.get('scan_id'),
                'vulnerability_type': vuln_data.get('vulnerability_type'),
                'severity': vuln_data.get('severity'),
                'title': vuln_data.get('title'),
                'description': vuln_data.get('description'),
                'evidence': vuln_data.get('evidence'),
                'attack_vector': vuln_data.get('attack_vector'),
                'impact': vuln_data.get('impact'),
                'recommendation': vuln_data.get('recommendation'),
                'confidence_score': vuln_data.get('confidence_score', 0.0),
                'owasp_category': vuln_data.get('owasp_category'),
                'cve_reference': vuln_data.get('cve_reference'),
                'remediation_effort': vuln_data.get('remediation_effort', 'medium'),
                'created_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.client.table('vulnerabilities').insert(vuln_record).execute()
            if result.data:
                logger.info(f"Created vulnerability: {vuln_record['id']}")
                return vuln_record['id']
            return None
            
        except Exception as e:
            logger.error(f"Failed to create vulnerability: {e}")
            return None
    
    async def get_vulnerabilities_by_scan(self, scan_id: str) -> List[Dict[str, Any]]:
        """Get vulnerabilities for a scan"""
        try:
            result = self.supabase.client.table('vulnerabilities').select('*').eq('scan_id', scan_id).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get vulnerabilities for scan {scan_id}: {e}")
            return []
    
    # Test Results Management
    async def create_test_result(self, test_data: Dict[str, Any]) -> Optional[str]:
        """Create test result record"""
        try:
            test_record = {
                'id': str(uuid.uuid4()),
                'scan_id': test_data.get('scan_id'),
                'test_type': test_data.get('test_type'),
                'test_name': test_data.get('test_name'),
                'prompt': test_data.get('prompt'),
                'response': test_data.get('response'),
                'vulnerable': test_data.get('vulnerable', False),
                'severity': test_data.get('severity'),
                'confidence': test_data.get('confidence', 0.0),
                'explanation': test_data.get('explanation'),
                'mitigation': test_data.get('mitigation'),
                'execution_time': test_data.get('execution_time', 0.0),
                'agent_used': test_data.get('agent_used'),
                'attack_category': test_data.get('attack_category'),
                'created_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.client.table('test_results').insert(test_record).execute()
            if result.data:
                return test_record['id']
            return None
            
        except Exception as e:
            logger.error(f"Failed to create test result: {e}")
            return None
    
    async def get_test_results_by_scan(self, scan_id: str) -> List[Dict[str, Any]]:
        """Get test results for a scan"""
        try:
            result = self.supabase.client.table('test_results').select('*').eq('scan_id', scan_id).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get test results for scan {scan_id}: {e}")
            return []
    
    # Progress Tracking
    async def update_scan_progress(self, scan_id: str, progress: int, message: str, phase: str = None) -> bool:
        """Update scan progress"""
        try:
            progress_data = {
                'progress': progress,
                'status_message': message,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            if phase:
                progress_data['current_phase'] = phase
            
            # Update main scan record
            scan_updated = await self.update_scan(scan_id, progress_data)
            
            # Also create/update progress record
            progress_record = {
                'scan_id': scan_id,
                'progress_percentage': progress,
                'current_phase': phase or 'unknown',
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            try:
                self.supabase.client.table('scan_progress').insert(progress_record).execute()
            except:
                pass  # Progress table is optional
            
            return scan_updated
            
        except Exception as e:
            logger.error(f"Failed to update progress for scan {scan_id}: {e}")
            return False
    
    # User Profile Management
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        try:
            result = self.supabase.client.table('profiles').select('*').eq('id', user_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get user profile {user_id}: {e}")
            return None
    
    async def create_user_profile(self, profile_data: Dict[str, Any]) -> Optional[str]:
        """Create user profile"""
        try:
            profile_record = {
                'id': profile_data.get('id', str(uuid.uuid4())),
                'user_type': profile_data.get('user_type', 'individual'),
                'full_name': profile_data.get('full_name'),
                'company_name': profile_data.get('company_name'),
                'industry': profile_data.get('industry'),
                'company_size': profile_data.get('company_size'),
                'security_maturity_level': profile_data.get('security_maturity_level'),
                'subscription_tier': profile_data.get('subscription_tier', 'basic'),
                'api_quota': profile_data.get('api_quota', 100),
                'api_usage_count': profile_data.get('api_usage_count', 0),
                'notification_preferences': profile_data.get('notification_preferences', {}),
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.client.table('profiles').insert(profile_record).execute()
            if result.data:
                return profile_record['id']
            return None
            
        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            return None
    
    # Security Audit Logging
    async def log_security_event(self, event_type: str, event_data: Dict[str, Any], severity: str = 'INFO') -> bool:
        """Log security event"""
        try:
            audit_record = {
                'id': str(uuid.uuid4()),
                'event_type': event_type,
                'event_data': json.dumps(event_data),
                'severity': severity,
                'user_id': event_data.get('user_id'),
                'ip_address': event_data.get('ip_address'),
                'user_agent': event_data.get('user_agent'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.client.table('security_audit_logs').insert(audit_record).execute()
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            return False
    
    # Statistics and Analytics
    async def get_platform_statistics(self) -> Dict[str, Any]:
        """Get platform usage statistics"""
        try:
            # Total scans
            total_scans = self.supabase.client.table('llm_scans').select('id').execute()
            
            # Completed scans
            completed_scans = self.supabase.client.table('llm_scans').select('id').eq('status', 'completed').execute()
            
            # Total vulnerabilities
            total_vulns = self.supabase.client.table('vulnerabilities').select('id').execute()
            
            # Active users
            active_users = self.supabase.client.table('profiles').select('id').execute()
            
            return {
                'total_scans': len(total_scans.data) if total_scans.data else 0,
                'completed_scans': len(completed_scans.data) if completed_scans.data else 0,
                'total_vulnerabilities': len(total_vulns.data) if total_vulns.data else 0,
                'active_users': len(active_users.data) if active_users.data else 0,
                'success_rate': (len(completed_scans.data) / max(len(total_scans.data), 1)) * 100 if total_scans.data else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get platform statistics: {e}")
            return {}
    
    # Cleanup and Maintenance
    async def cleanup_old_scans(self, days_old: int = 30) -> int:
        """Clean up old scan records"""
        try:
            cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - days_old)
            
            # Get old scans
            old_scans = self.supabase.client.table('llm_scans').select('id').lt('created_at', cutoff_date.isoformat()).execute()
            
            if old_scans.data:
                scan_ids = [scan['id'] for scan in old_scans.data]
                
                # Delete related records first
                for scan_id in scan_ids:
                    self.supabase.client.table('vulnerabilities').delete().eq('scan_id', scan_id).execute()
                    self.supabase.client.table('test_results').delete().eq('scan_id', scan_id).execute()
                
                # Delete scans
                self.supabase.client.table('llm_scans').delete().in_('id', scan_ids).execute()
                
                logger.info(f"Cleaned up {len(scan_ids)} old scans")
                return len(scan_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup old scans: {e}")
            return 0

# Global database manager instance
db_manager = DatabaseManager()

# Legacy compatibility functions
def get_db_session() -> Generator[DatabaseManager, None, None]:
    """
    Provide database manager for dependency injection
    Compatible with FastAPI Depends()
    """
    yield db_manager

def create_db_session() -> DatabaseManager:
    """
    Create database manager instance
    """
    return db_manager

async def init_db() -> bool:
    """
    Initialize database
    """
    return await db_manager.initialize_database()

# Convenience functions
async def health_check() -> bool:
    """Quick health check"""
    return await db_manager.health_check()

async def get_scan(scan_id: str) -> Optional[Dict[str, Any]]:
    """Quick scan lookup"""
    return await db_manager.get_scan_by_id(scan_id)

async def create_scan(scan_data: Dict[str, Any]) -> Optional[str]:
    """Quick scan creation"""
    return await db_manager.create_scan(scan_data)

async def update_progress(scan_id: str, progress: int, message: str) -> bool:
    """Quick progress update"""
    return await db_manager.update_scan_progress(scan_id, progress, message)

# Export all important functions
__all__ = [
    'DatabaseManager', 'db_manager', 'get_db_session', 'create_db_session', 
    'init_db', 'health_check', 'get_scan', 'create_scan', 'update_progress'
]
