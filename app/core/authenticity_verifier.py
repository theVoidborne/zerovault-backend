import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from app.services.supabase_service import supabase_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class AuthenticityReport:
    scan_id: str
    authenticity_score: float
    is_authentic: bool
    verification_checks: Dict[str, bool]
    fake_indicators: List[str]
    real_indicators: List[str]
    confidence_level: str
    verification_timestamp: datetime

class AuthenticityVerifier:
    """Verify that scan results are genuine and not simulated"""
    
    def __init__(self):
        self.supabase = supabase_service
        
    async def verify_scan_authenticity(self, scan_id: str) -> AuthenticityReport:
        """Comprehensive verification that scan results are genuine"""
        
        verification_checks = {
            "real_api_calls_made": await self._verify_real_api_calls(scan_id),
            "genuine_responses_received": await self._verify_genuine_responses(scan_id),
            "actual_costs_incurred": await self._verify_actual_costs(scan_id),
            "authentic_vulnerabilities": await self._verify_authentic_vulnerabilities(scan_id),
            "real_progress_tracking": await self._verify_real_progress(scan_id),
            "ai_analysis_authenticity": await self._verify_ai_analysis(scan_id),
            "no_hardcoded_responses": await self._verify_no_hardcoded_responses(scan_id),
            "timestamp_consistency": await self._verify_timestamp_consistency(scan_id),
            "cost_response_correlation": await self._verify_cost_response_correlation(scan_id)
        }
        
        # Calculate authenticity score
        authenticity_score = sum(verification_checks.values()) / len(verification_checks)
        
        # Identify fake and real indicators
        fake_indicators = await self._identify_fake_indicators(scan_id, verification_checks)
        real_indicators = await self._identify_real_indicators(scan_id, verification_checks)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(authenticity_score, verification_checks)
        
        report = AuthenticityReport(
            scan_id=scan_id,
            authenticity_score=authenticity_score,
            is_authentic=authenticity_score > 0.8,
            verification_checks=verification_checks,
            fake_indicators=fake_indicators,
            real_indicators=real_indicators,
            confidence_level=confidence_level,
            verification_timestamp=datetime.now()
        )
        
        # Store verification report
        await self._store_authenticity_report(report)
        
        return report
    
    async def _verify_real_api_calls(self, scan_id: str) -> bool:
        """Verify that real API calls were made to external services"""
        
        try:
            # Check for API communication logs
            api_logs = await self.supabase.table("api_communication_logs").select("*").eq("scan_id", scan_id).execute()
            
            if not api_logs.data:
                return False
            
            # Verify API calls have real response data
            for log in api_logs.data:
                if not log.get("response_content") or not log.get("http_status_code"):
                    return False
                
                # Check for realistic response times
                response_time = log.get("response_time_ms", 0)
                if response_time < 100 or response_time > 60000:  # Unrealistic response times
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying real API calls: {str(e)}")
            return False
    
    async def _verify_genuine_responses(self, scan_id: str) -> bool:
        """Verify that responses are genuine and not templated"""
        
        try:
            api_logs = await self.supabase.table("api_communication_logs").select("response_content").eq("scan_id", scan_id).execute()
            
            if not api_logs.data:
                return False
            
            responses = [log["response_content"] for log in api_logs.data if log.get("response_content")]
            
            # Check for response diversity
            unique_responses = set(responses)
            if len(unique_responses) < len(responses) * 0.7:  # Too many duplicate responses
                return False
            
            # Check for templated response patterns
            templated_patterns = [
                "Target model response to:",
                "Simulated response:",
                "Mock response:",
                "Test response:",
                "Placeholder response"
            ]
            
            for response in responses:
                if any(pattern in response for pattern in templated_patterns):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying genuine responses: {str(e)}")
            return False
    
    async def _verify_actual_costs(self, scan_id: str) -> bool:
        """Verify that actual costs were incurred and tracked"""
        
        try:
            cost_records = await self.supabase.table("real_cost_tracking").select("*").eq("scan_id", scan_id).execute()
            
            if not cost_records.data:
                return False
            
            total_cost = sum(record["actual_cost_usd"] for record in cost_records.data)
            
            # Verify costs are realistic (not zero or suspiciously low)
            if total_cost < 0.001:  # Less than $0.001 is suspicious for AI API calls
                return False
            
            # Verify cost breakdown exists
            for record in cost_records.data:
                if not record.get("cost_breakdown") or not record.get("api_response_metadata"):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying actual costs: {str(e)}")
            return False
    
    async def _verify_authentic_vulnerabilities(self, scan_id: str) -> bool:
        """Verify that vulnerabilities are authentic and not fabricated"""
        
        try:
            vulnerabilities = await self.supabase.table("vulnerabilities").select("*").eq("scan_id", scan_id).execute()
            
            if not vulnerabilities.data:
                return True  # No vulnerabilities is valid
            
            for vuln in vulnerabilities.data:
                # Check for AI analysis authenticity
                if not vuln.get("ai_generated_remediation") or not vuln.get("attack_conversation"):
                    return False
                
                # Verify confidence scores are realistic
                confidence = vuln.get("ai_confidence_score", 0)
                if confidence == 0 or confidence == 1:  # Perfect scores are suspicious
                    return False
                
                # Check for evidence in attack conversation
                conversation = vuln.get("attack_conversation", {})
                if not conversation.get("attack_prompt") or not conversation.get("target_response"):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying authentic vulnerabilities: {str(e)}")
            return False
    
    async def _verify_real_progress(self, scan_id: str) -> bool:
        """Verify that progress tracking was based on real execution"""
        
        try:
            progress_records = await self.supabase.table("real_progress_tracking").select("*").eq("scan_id", scan_id).execute()
            
            if not progress_records.data:
                return False
            
            # Verify progress increments are realistic
            progress_values = [record["genuine_progress_percentage"] for record in progress_records.data]
            progress_values.sort()
            
            # Check for unrealistic progress jumps
            for i in range(1, len(progress_values)):
                jump = progress_values[i] - progress_values[i-1]
                if jump > 50:  # More than 50% jump is suspicious
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying real progress: {str(e)}")
            return False
    
    async def _verify_ai_analysis(self, scan_id: str) -> bool:
        """Verify that AI analysis was genuine and not rule-based"""
        
        try:
            analyses = await self.supabase.table("real_vulnerability_analyses").select("*").eq("scan_id", scan_id).execute()
            
            if not analyses.data:
                return False
            
            for analysis in analyses.data:
                # Check for AI reasoning quality
                reasoning = analysis.get("ai_reasoning", "")
                if len(reasoning) < 50:  # Too short for genuine AI analysis
                    return False
                
                # Check for verification data
                verification_data = analysis.get("verification_data", {})
                if not verification_data.get("authenticity_score"):
                    return False
                
                # Verify evidence matches response
                evidence = analysis.get("evidence", [])
                if not evidence:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying AI analysis: {str(e)}")
            return False
    
    async def _verify_no_hardcoded_responses(self, scan_id: str) -> bool:
        """Verify no hardcoded or templated responses were used"""
        
        try:
            # Check for common hardcoded response patterns
            api_logs = await self.supabase.table("api_communication_logs").select("response_content").eq("scan_id", scan_id).execute()
            
            hardcoded_patterns = [
                "f\"Target model response to: {prompt",
                "Simulated response",
                "Mock response",
                "Test response for",
                "Placeholder",
                "Lorem ipsum",
                "Example response"
            ]
            
            for log in api_logs.data:
                response = log.get("response_content", "")
                if any(pattern in response for pattern in hardcoded_patterns):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying no hardcoded responses: {str(e)}")
            return False
    
    async def _verify_timestamp_consistency(self, scan_id: str) -> bool:
        """Verify that timestamps are consistent and realistic"""
        
        try:
            # Get all timestamps from various tables
            scan_data = await self.supabase.table("llm_scans").select("created_at").eq("id", scan_id).single().execute()
            api_logs = await self.supabase.table("api_communication_logs").select("timestamp").eq("scan_id", scan_id).execute()
            
            scan_start = datetime.fromisoformat(scan_data.data["created_at"].replace('Z', '+00:00'))
            
            # Verify API call timestamps are after scan start
            for log in api_logs.data:
                api_time = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
                if api_time < scan_start:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying timestamp consistency: {str(e)}")
            return False
    
    async def _verify_cost_response_correlation(self, scan_id: str) -> bool:
        """Verify that costs correlate with response lengths and complexity"""
        
        try:
            cost_records = await self.supabase.table("real_cost_tracking").select("*").eq("scan_id", scan_id).execute()
            api_logs = await self.supabase.table("api_communication_logs").select("*").eq("scan_id", scan_id).execute()
            
            if not cost_records.data or not api_logs.data:
                return False
            
            # Create correlation map
            for cost_record in cost_records.data:
                tokens_used = cost_record["total_tokens"]
                cost = cost_record["actual_cost_usd"]
                
                # Verify cost/token ratio is realistic
                cost_per_token = cost / tokens_used if tokens_used > 0 else 0
                if cost_per_token < 0.000001 or cost_per_token > 0.001:  # Outside realistic range
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying cost-response correlation: {str(e)}")
            return False
    
    async def _identify_fake_indicators(self, scan_id: str, verification_checks: Dict[str, bool]) -> List[str]:
        """Identify specific indicators that suggest fake results"""
        
        fake_indicators = []
        
        if not verification_checks["real_api_calls_made"]:
            fake_indicators.append("No evidence of real API calls to external services")
        
        if not verification_checks["genuine_responses_received"]:
            fake_indicators.append("Responses appear templated or duplicated")
        
        if not verification_checks["actual_costs_incurred"]:
            fake_indicators.append("No realistic costs tracked for API usage")
        
        if not verification_checks["no_hardcoded_responses"]:
            fake_indicators.append("Hardcoded response patterns detected")
        
        return fake_indicators
    
    async def _identify_real_indicators(self, scan_id: str, verification_checks: Dict[str, bool]) -> List[str]:
        """Identify specific indicators that suggest real results"""
        
        real_indicators = []
        
        if verification_checks["real_api_calls_made"]:
            real_indicators.append("Genuine API calls with realistic response times")
        
        if verification_checks["actual_costs_incurred"]:
            real_indicators.append("Actual costs tracked with detailed breakdowns")
        
        if verification_checks["ai_analysis_authenticity"]:
            real_indicators.append("AI analysis shows genuine reasoning and evidence")
        
        if verification_checks["timestamp_consistency"]:
            real_indicators.append("Consistent and realistic timestamp progression")
        
        return real_indicators
    
    def _determine_confidence_level(self, authenticity_score: float, verification_checks: Dict[str, bool]) -> str:
        """Determine confidence level in authenticity assessment"""
        
        if authenticity_score > 0.9:
            return "very_high"
        elif authenticity_score > 0.8:
            return "high"
        elif authenticity_score > 0.6:
            return "medium"
        elif authenticity_score > 0.4:
            return "low"
        else:
            return "very_low"
    
    async def _store_authenticity_report(self, report: AuthenticityReport):
        """Store authenticity verification report"""
        
        try:
            report_record = {
                "scan_id": report.scan_id,
                "authenticity_score": report.authenticity_score,
                "is_authentic": report.is_authentic,
                "verification_checks": report.verification_checks,
                "fake_indicators": report.fake_indicators,
                "real_indicators": report.real_indicators,
                "confidence_level": report.confidence_level,
                "verification_timestamp": report.verification_timestamp.isoformat()
            }
            
            await self.supabase.table("authenticity_reports").insert(report_record).execute()
            
        except Exception as e:
            logger.error(f"Error storing authenticity report: {str(e)}")
