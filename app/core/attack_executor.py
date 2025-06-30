"""
PRODUCTION Attack Executor - REAL Attack Execution Only
Executes real attacks against real LLMs with real AI analysis
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict
from app.core.universal_llm_client import UniversalLLMClient
from app.core.vulnerability_analyzer import RealAIVulnerabilityAnalyzer, RealVulnerability

logger = logging.getLogger(__name__)

@dataclass
class RealAttackResult:
    """Real attack result with authentic data"""
    attack_id: str
    attack_type: str
    attack_prompt: str
    target_response: str
    vulnerabilities_found: List[RealVulnerability]
    success: bool
    confidence_score: float
    execution_time: float
    tokens_used: int
    target_model_info: Dict[str, Any]
    attacker_model_info: Dict[str, Any]
    timestamp: str
    real_api_calls_made: int
    
class RealAttackExecutor:
    """REAL attack executor - makes actual API calls"""
    
    def __init__(self, vulnerability_analyzer: RealAIVulnerabilityAnalyzer):
        self.vulnerability_analyzer = vulnerability_analyzer
        self.attack_count = 0
        self.execution_log = []
        
    async def execute_real_attack(self,
                                attack_prompt: str,
                                target_api_key: str,
                                target_provider: str,
                                target_model: str,
                                target_endpoint: str = None,
                                attack_type: str = "prompt_injection") -> RealAttackResult:
        """Execute REAL attack against REAL target LLM"""
        
        attack_id = f"real_attack_{self.attack_count + 1}_{int(datetime.utcnow().timestamp())}"
        start_time = datetime.utcnow()
        
        logger.info(f"🎯 EXECUTING REAL ATTACK: {attack_id}")
        logger.info(f"Target: {target_provider}:{target_model}")
        logger.info(f"Attack Type: {attack_type}")
        
        try:
            self.attack_count += 1
            
            # Prepare target model info
            target_model_info = {
                "provider": target_provider,
                "model": target_model,
                "endpoint": target_endpoint,
                "api_key_last_4": target_api_key[-4:] if len(target_api_key) > 4 else "****"
            }
            
            # Execute REAL attack against target LLM
            async with UniversalLLMClient() as client:
                logger.info(f"📡 Making REAL API call to target LLM...")
                
                target_response = await client.call_llm(
                    provider=target_provider,
                    model=target_model,
                    api_key=target_api_key,
                    prompt=attack_prompt,
                    endpoint=target_endpoint,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                if not target_response.success:
                    logger.error(f"❌ Target LLM API call failed: {target_response.error}")
                    
                    return RealAttackResult(
                        attack_id=attack_id,
                        attack_type=attack_type,
                        attack_prompt=attack_prompt,
                        target_response=f"API Error: {target_response.error}",
                        vulnerabilities_found=[],
                        success=False,
                        confidence_score=0.0,
                        execution_time=0.0,
                        tokens_used=0,
                        target_model_info=target_model_info,
                        attacker_model_info={"error": "Target API call failed"},
                        timestamp=datetime.utcnow().isoformat(),
                        real_api_calls_made=1
                    )
                
                logger.info(f"✅ Target LLM responded - {target_response.tokens_used} tokens")
                
                # Analyze response for REAL vulnerabilities using AI
                logger.info(f"🔍 Analyzing response for vulnerabilities using REAL AI...")
                
                vulnerabilities = await self.vulnerability_analyzer.analyze_response_for_vulnerabilities(
                    attack_prompt=attack_prompt,
                    target_response=target_response.content,
                    target_model_info=target_model_info
                )
                
                # Calculate real success metrics
                success = len(vulnerabilities) > 0
                confidence_score = self._calculate_real_confidence_score(vulnerabilities)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Log the real attack execution
                self.execution_log.append({
                    "attack_id": attack_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": success,
                    "vulnerabilities_count": len(vulnerabilities),
                    "tokens_used": target_response.tokens_used,
                    "execution_time": execution_time
                })
                
                logger.info(f"🎯 REAL ATTACK COMPLETED: {attack_id}")
                logger.info(f"Success: {success}, Vulnerabilities: {len(vulnerabilities)}, Confidence: {confidence_score:.2f}")
                
                return RealAttackResult(
                    attack_id=attack_id,
                    attack_type=attack_type,
                    attack_prompt=attack_prompt,
                    target_response=target_response.content,
                    vulnerabilities_found=vulnerabilities,
                    success=success,
                    confidence_score=confidence_score,
                    execution_time=execution_time,
                    tokens_used=target_response.tokens_used,
                    target_model_info=target_model_info,
                    attacker_model_info={
                        "analyzer_provider": self.vulnerability_analyzer.analyzer_provider,
                        "analyzer_model": self.vulnerability_analyzer.analyzer_model,
                        "analysis_tokens": sum(v.ai_analysis.get("tokens_used", 0) for v in vulnerabilities)
                    },
                    timestamp=datetime.utcnow().isoformat(),
                    real_api_calls_made=2  # Target call + analysis call
                )
                
        except Exception as e:
            logger.error(f"❌ REAL ATTACK EXECUTION FAILED: {attack_id} - {str(e)}")
            
            return RealAttackResult(
                attack_id=attack_id,
                attack_type=attack_type,
                attack_prompt=attack_prompt,
                target_response=f"Execution Error: {str(e)}",
                vulnerabilities_found=[],
                success=False,
                confidence_score=0.0,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                tokens_used=0,
                target_model_info=target_model_info,
                attacker_model_info={"error": str(e)},
                timestamp=datetime.utcnow().isoformat(),
                real_api_calls_made=0
            )
    
    def _calculate_real_confidence_score(self, vulnerabilities: List[RealVulnerability]) -> float:
        """Calculate real confidence score based on AI analysis"""
        
        if not vulnerabilities:
            return 0.0
        
        # Weight by severity and confidence
        severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for vuln in vulnerabilities:
            severity_weight = severity_weights.get(vuln.severity, 0.4)
            weighted_confidence = vuln.confidence_score * severity_weight
            total_score += weighted_confidence
            total_weight += severity_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
