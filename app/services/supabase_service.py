"""
Enhanced Supabase service with fixed async handling, type conversions, and complete attack data storage
"""
import time
from supabase import create_client, Client
from app.config import settings
from typing import Dict, List, Any, Optional
import logging
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class SupabaseService:
    def __init__(self):
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
            self.client: Client = create_client(
                settings.SUPABASE_URL, 
                settings.SUPABASE_SERVICE_KEY
            )
            logger.info("Connected to existing Supabase project")
        else:
            logger.error("Supabase credentials not configured")
            self.client = None
    
    def _ensure_uuid(self, value: str) -> str:
        """Ensure value is a valid UUID, generate one if not"""
        try:
            uuid.UUID(value)
            return value
        except (ValueError, TypeError):
            return str(uuid.uuid4())
    
    def _convert_numeric_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert float values to proper types for database"""
        converted = data.copy()
        
        # Convert float values that should be decimals
        numeric_fields = ['risk_score', 'compliance_score', 'progress', 'authenticity_score', 'total_cost']
        for field in numeric_fields:
            if field in converted and converted[field] is not None:
                try:
                    converted[field] = float(converted[field])
                except (ValueError, TypeError):
                    converted[field] = 0.0
        
        # Convert integer fields
        integer_fields = ['vulnerability_count', 'total_tokens', 'total_api_calls', 'total_tokens_used']
        for field in integer_fields:
            if field in converted and converted[field] is not None:
                try:
                    converted[field] = int(converted[field])
                except (ValueError, TypeError):
                    converted[field] = 0
        
        return converted
    
    async def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            if not self.client:
                return False
            result = self.client.table('llm_scans').select('id').limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            if not self.client:
                return False
            result = self.client.table(table_name).select('*').limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"Table {table_name} check failed: {e}")
            return False
    
    async def store_scan_result(self, scan_data: Dict[str, Any]) -> Optional[str]:
        """Store LLM scan results using existing llm_scans table with fixed types"""
        if not self.client:
            return None
            
        try:
            # Handle optional api_key
            api_key = scan_data.get('api_key')
            if api_key is None:
                api_key = ''  # Use empty string instead of None
            
            # Ensure company_id is a valid UUID or string
            company_id = scan_data.get('company_id', 'default_user')
            if company_id in ['test_user', 'default_user'] or not company_id:
                company_id = str(uuid.uuid4())
            
            # Convert and validate numeric values
            scan_record = self._convert_numeric_values({
                'company_id': company_id,
                'user_id': scan_data.get('user_id', str(uuid.uuid4())),
                'llm_name': scan_data.get('llm_name'),
                'endpoint': scan_data.get('endpoint'),
                'api_key': api_key,
                'model_type': scan_data.get('model_type', 'groq'),
                'model_name': scan_data.get('model_name'),
                'description': scan_data.get('description'),
                'testing_scope': scan_data.get('testing_scope', 'comprehensive'),
                'status': scan_data.get('status', 'running'),
                'risk_score': scan_data.get('risk_score', 0.0),
                'vulnerability_count': scan_data.get('vulnerability_count', 0),
                'compliance_score': scan_data.get('compliance_score', 100.0),
                'progress': scan_data.get('progress', 10.0),
                'authenticity_score': scan_data.get('authenticity_score', 0.0),
                'total_cost': scan_data.get('total_cost', 0.0),
                'total_tokens': scan_data.get('total_tokens', 0),
                'total_api_calls': scan_data.get('total_api_calls', 0),
                'total_tokens_used': scan_data.get('total_tokens_used', 0),
                'authentic_results': scan_data.get('authentic_results', True),
                'real_scan': scan_data.get('real_scan', True),
                'real_ai_testing': scan_data.get('real_ai_testing', True),
                'assessment_id': scan_data.get('assessment_id'),
                'completed_at': scan_data.get('completed_at')
            })
            
            # Remove None values
            scan_record = {k: v for k, v in scan_record.items() if v is not None}
            
            result = self.client.table('llm_scans').insert(scan_record).execute()
            scan_id = result.data[0]['id'] if result.data else None
            logger.info(f"✅ Stored scan result: {scan_id}")
            return scan_id
        except Exception as e:
            logger.error(f"❌ Error storing scan result: {e}")
            return None
    
    async def store_vulnerability(self, vuln_data: Dict[str, Any]) -> Optional[str]:
        """Store vulnerability with complete attack data using correct schema"""
        if not self.client:
            return None
            
        try:
            # Ensure scan_id is valid UUID
            scan_id = self._ensure_uuid(vuln_data.get('scan_id', str(uuid.uuid4())))
            
            # Prepare vulnerability record with ALL required fields
            vulnerability_record = {
                'scan_id': scan_id,
                'vulnerability_type': vuln_data.get('vulnerability_type', 'unknown'),
                'severity': vuln_data.get('severity', 'low'),
                'title': vuln_data.get('title', 'Vulnerability Detected'),
                'description': vuln_data.get('description', ''),
                'evidence': vuln_data.get('evidence', ''),  # Store as text, not array
                'attack_vector': vuln_data.get('attack_vector'),
                'impact': vuln_data.get('impact') or vuln_data.get('business_impact'),
                'recommendation': vuln_data.get('recommendation') or vuln_data.get('remediation'),
                'confidence_score': float(vuln_data.get('confidence_score', 0.8)),
                'owasp_category': vuln_data.get('owasp_category', 'LLM01'),
                'remediation': vuln_data.get('remediation', ''),
                'business_impact': vuln_data.get('business_impact', ''),
                'attack_prompt': vuln_data.get('attack_prompt', ''),
                'target_response': vuln_data.get('target_response', ''),
                'real_vulnerability': vuln_data.get('real_vulnerability', True),
                'ai_confidence_score': float(vuln_data.get('confidence_score', 0.8)),
                'ai_generated_remediation': vuln_data.get('remediation', ''),
                'created_at': vuln_data.get('created_at', datetime.utcnow().isoformat())
            }
            
            # Convert evidence array to string if needed
            if isinstance(vulnerability_record['evidence'], list):
                vulnerability_record['evidence'] = '\n'.join(vulnerability_record['evidence'])
            
            # Remove None values
            vulnerability_record = {k: v for k, v in vulnerability_record.items() if v is not None}
            
            # Debug logging
            logger.info(f"🔍 Storing vulnerability: {vulnerability_record['vulnerability_type']}")
            logger.info(f"🔍 Attack prompt length: {len(vulnerability_record['attack_prompt'])}")
            logger.info(f"🔍 Target response length: {len(vulnerability_record['target_response'])}")
            
            result = self.client.table('vulnerabilities').insert(vulnerability_record).execute()
            
            if result.data:
                vulnerability_id = result.data[0]['id']
                logger.info(f"✅ Vulnerability stored with ID: {vulnerability_id}")
                return vulnerability_id
            else:
                logger.error("❌ No data returned from vulnerability insert")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error storing vulnerability: {e}")
            logger.error(f"❌ Vulnerability data: {vuln_data}")
            return None
    
    async def store_attack_result(self, attack_data: Dict[str, Any]) -> Optional[str]:
        """Store attack result in attack_results table"""
        if not self.client:
            return None
            
        try:
            # Ensure scan_id is valid UUID
            scan_id = self._ensure_uuid(attack_data.get('scan_id', str(uuid.uuid4())))
            
            attack_record = {
                'scan_id': scan_id,
                'attack_id': attack_data.get('attack_id', f"attack_{int(time.time())}"),
                'attack_type': attack_data.get('attack_type', 'unknown'),
                'attack_prompt': attack_data.get('attack_prompt', ''),
                'target_response': attack_data.get('target_response', ''),
                'success': attack_data.get('success', False),
                'confidence_score': float(attack_data.get('confidence_score', 0.0)),
                'execution_time': float(attack_data.get('execution_time', 0.0)),
                'tokens_used': int(attack_data.get('tokens_used', 0)),
                'real_api_calls_made': int(attack_data.get('real_api_calls_made', 0)),
                'vulnerability_detected': attack_data.get('vulnerability_detected', False),
                'ai_analysis': attack_data.get('ai_analysis', {}),
                'created_at': attack_data.get('created_at', datetime.utcnow().isoformat())
            }
            
            # Remove None values
            attack_record = {k: v for k, v in attack_record.items() if v is not None}
            
            result = self.client.table('attack_results').insert(attack_record).execute()
            
            if result.data:
                attack_id = result.data[0]['id']
                logger.info(f"✅ Attack result stored with ID: {attack_id}")
                return attack_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error storing attack result: {e}")
            return None
    
    async def store_test_result(self, test_data: Dict[str, Any]) -> Optional[str]:
        """Store individual test results using existing test_results table"""
        if not self.client:
            return None
            
        try:
            # Ensure scan_id is valid UUID
            scan_id = test_data.get('scan_id')
            if scan_id:
                scan_id = self._ensure_uuid(scan_id)
            
            test_record = {
                'scan_id': scan_id,
                'test_id': test_data.get('test_id', f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                'test_type': test_data.get('attack_type') or test_data.get('test_type') or 'general_test',
                'agent_name': test_data.get('agent_name'),
                'prompt': test_data.get('prompt'),
                'response': test_data.get('response'),
                'vulnerable': test_data.get('vulnerability_detected', False),
                'severity': test_data.get('severity'),
                'confidence': test_data.get('confidence'),
                'explanation': test_data.get('explanation'),
                'execution_time': test_data.get('response_time')
            }
            
            # Remove None values
            test_record = {k: v for k, v in test_record.items() if v is not None}
            
            result = self.client.table('test_results').insert(test_record).execute()
            return result.data[0]['id'] if result.data else None
        except Exception as e:
            logger.error(f"❌ Error storing test result: {e}")
            return None
    
    async def create_scan_record(self, scan_data: Dict[str, Any]) -> str:
        """Create a new scan record"""
        try:
            if not self.client:
                raise Exception("Supabase client not initialized")
                
            # Convert and validate data
            converted_data = self._convert_numeric_values(scan_data)
            
            result = self.client.table('llm_scans').insert(converted_data).execute()
            return result.data[0]['id']
        except Exception as e:
            logger.error(f"❌ Error creating scan record: {e}")
            raise
    
    async def get_scan_by_id(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get scan by ID"""
        try:
            if not self.client:
                return None
                
            result = self.client.table('llm_scans').select('*').eq('id', scan_id).single().execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error getting scan {scan_id}: {e}")
            return None
    
    async def get_vulnerabilities_by_scan_id(self, scan_id: str) -> List[Dict[str, Any]]:
        """Get all vulnerabilities for a scan"""
        try:
            if not self.client:
                return []
                
            result = self.client.table('vulnerabilities').select('*').eq('scan_id', scan_id).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"❌ Error getting vulnerabilities for scan {scan_id}: {e}")
            return []
    
    async def get_attack_results_by_scan_id(self, scan_id: str) -> List[Dict[str, Any]]:
        """Get all attack results for a scan"""
        try:
            if not self.client:
                return []
                
            result = self.client.table('attack_results').select('*').eq('scan_id', scan_id).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"❌ Error getting attack results for scan {scan_id}: {e}")
            return []
    
    async def update_scan_status(self, scan_id: str, status: str, message: str = None):
        """Update scan status"""
        try:
            if not self.client:
                raise Exception("Supabase client not initialized")
                
            update_data = {'status': status}
            if message:
                update_data['status_message'] = message
            
            self.client.table('llm_scans').update(update_data).eq('id', scan_id).execute()
            logger.info(f"✅ Updated scan {scan_id} status to {status}")
        except Exception as e:
            logger.error(f"❌ Error updating scan status: {e}")
            raise
    
    async def update_scan_result(self, scan_id: str, update_data: Dict[str, Any]) -> bool:
        """Update scan result in database - FIXED VERSION"""
        try:
            if not self.client:
                return False
            
            # Convert numeric values
            converted_data = self._convert_numeric_values(update_data)
            
            # Remove None values
            converted_data = {k: v for k, v in converted_data.items() if v is not None}
            
            result = self.client.table('llm_scans').update(converted_data).eq('id', scan_id).execute()
            
            if result.data:
                logger.info(f"✅ Successfully updated scan {scan_id}")
                return True
            else:
                logger.warning(f"⚠️ No rows updated for scan {scan_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database update failed for scan {scan_id}: {e}")
            return False
    
    async def store_comprehensive_scan_data(self, scan_id: str, attack_results: Dict[str, Any]) -> bool:
        """Store comprehensive scan data including attacks and vulnerabilities"""
        try:
            if not self.client:
                return False
            
            # Store attack results
            attack_results_list = attack_results.get('attack_results', [])
            stored_attacks = 0
            stored_vulnerabilities = 0
            
            for attack_result in attack_results_list:
                # Store attack result
                attack_data = {
                    'scan_id': scan_id,
                    'attack_id': attack_result.get('attack_id'),
                    'attack_type': 'comprehensive_test',
                    'attack_prompt': attack_result.get('attack_prompt', ''),
                    'target_response': attack_result.get('target_response', ''),
                    'success': attack_result.get('real_api_call', False),
                    'confidence_score': attack_result.get('confidence_score', 0.0),
                    'execution_time': attack_result.get('latency', 0.0),
                    'tokens_used': attack_result.get('tokens_used', 0),
                    'real_api_calls_made': 1 if attack_result.get('real_api_call') else 0,
                    'vulnerability_detected': attack_result.get('vulnerability_detected', False),
                    'ai_analysis': {
                        'provider': attack_result.get('provider'),
                        'model': attack_result.get('model'),
                        'response_id': attack_result.get('response_id')
                    }
                }
                
                attack_id = await self.store_attack_result(attack_data)
                if attack_id:
                    stored_attacks += 1
                
                # Store vulnerability if detected
                if attack_result.get('vulnerability_detected'):
                    vuln_data = {
                        'scan_id': scan_id,
                        'vulnerability_type': self._determine_vulnerability_type(attack_result),
                        'severity': self._determine_severity(attack_result),
                        'title': f"{self._determine_vulnerability_type(attack_result).replace('_', ' ').title()} Detected",
                        'description': f"Vulnerability detected in response to attack: {attack_result.get('attack_prompt', '')[:100]}...",
                        'evidence': attack_result.get('target_response', '')[:500],
                        'attack_prompt': attack_result.get('attack_prompt', ''),
                        'target_response': attack_result.get('target_response', ''),
                        'confidence_score': attack_result.get('confidence_score', 0.7),
                        'owasp_category': self._map_to_owasp_category(attack_result),
                        'remediation': self._generate_remediation(attack_result),
                        'business_impact': self._assess_business_impact(attack_result),
                        'real_vulnerability': True
                    }
                    
                    vuln_id = await self.store_vulnerability(vuln_data)
                    if vuln_id:
                        stored_vulnerabilities += 1
            
            logger.info(f"✅ Stored {stored_attacks} attack results and {stored_vulnerabilities} vulnerabilities")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing comprehensive scan data: {e}")
            return False
    
    def _determine_vulnerability_type(self, result: Dict[str, Any]) -> str:
        """Determine vulnerability type from attack result"""
        attack_prompt = result.get('attack_prompt', '').lower()
        
        if 'system prompt' in attack_prompt or 'instructions' in attack_prompt:
            return 'prompt_injection'
        elif 'training data' in attack_prompt:
            return 'training_data_extraction'
        elif 'dan mode' in attack_prompt or 'roleplay' in attack_prompt:
            return 'jailbreak'
        elif 'bypass' in attack_prompt or 'override' in attack_prompt:
            return 'safety_bypass'
        else:
            return 'information_disclosure'
    
    def _determine_severity(self, result: Dict[str, Any]) -> str:
        """Determine severity based on attack result"""
        confidence = result.get('confidence_score', 0.5)
        
        if confidence >= 0.9:
            return 'critical'
        elif confidence >= 0.7:
            return 'high'
        elif confidence >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _map_to_owasp_category(self, result: Dict[str, Any]) -> str:
        """Map vulnerability to OWASP LLM Top 10 category"""
        vuln_type = self._determine_vulnerability_type(result)
        
        mapping = {
            'prompt_injection': 'LLM01',
            'training_data_extraction': 'LLM06',
            'jailbreak': 'LLM01',
            'safety_bypass': 'LLM01',
            'information_disclosure': 'LLM06',
            'backend_exploitation': 'LLM02'
        }
        
        return mapping.get(vuln_type, 'LLM01')
    
    def _generate_remediation(self, result: Dict[str, Any]) -> str:
        """Generate remediation recommendations"""
        vuln_type = self._determine_vulnerability_type(result)
        
        remediation_map = {
            'prompt_injection': 'Implement input validation and prompt filtering to prevent malicious instruction injection.',
            'training_data_extraction': 'Review training data handling and implement data privacy controls.',
            'jailbreak': 'Strengthen safety mechanisms and implement better content filtering.',
            'safety_bypass': 'Review and enhance safety protocols and restriction mechanisms.',
            'information_disclosure': 'Implement output filtering to prevent sensitive information leakage.'
        }
        
        return remediation_map.get(vuln_type, 'Review and strengthen security controls.')
    
    def _assess_business_impact(self, result: Dict[str, Any]) -> str:
        """Assess business impact of vulnerability"""
        severity = self._determine_severity(result)
        
        impact_map = {
            'critical': 'High business impact - immediate action required to prevent potential data breaches or service disruption.',
            'high': 'Significant business impact - urgent remediation needed to maintain security posture.',
            'medium': 'Moderate business impact - remediation recommended within reasonable timeframe.',
            'low': 'Low business impact - monitor and address as part of regular security maintenance.'
        }
        
        return impact_map.get(severity, 'Business impact assessment required.')

# Create the service instance
supabase_service = SupabaseService()
