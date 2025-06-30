import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.agents.ai_agents.coordinator_agent import CoordinatorAgent
from app.agents.vulnerability_analyzer import VulnerabilityAnalyzer
from app.agents.jailbreak_agent import JailbreakAgent
from app.agents.data_extraction_agent import DataExtractionAgent as LegacyDataExtractionAgent
from app.agents.prompt_injection_agent import PromptInjectionAgent
from app.agents.bias_detection_agent import BiasDetectionAgent
from app.agents.stress_test_agent import StressTestAgent
from app.agents.backend_exploit_agent import BackendExploitAgent
from app.services.scan_orchestrator import ScanOrchestrator
from app.utils.logger import get_logger

logger = get_logger(__name__)

class UnifiedOrchestrator:
    """
    Unified orchestrator that seamlessly combines AI-powered agents with existing hardcoded agents
    for comprehensive LLM vulnerability assessment
    """
    
    def __init__(self):
        # AI-powered agents
        self.ai_coordinator = CoordinatorAgent()
        
        # Legacy hardcoded agents
        self.vulnerability_analyzer = VulnerabilityAnalyzer()
        self.jailbreak_agent = JailbreakAgent()
        self.legacy_data_extraction = LegacyDataExtractionAgent()
        self.prompt_injection_agent = PromptInjectionAgent()
        self.bias_detection_agent = BiasDetectionAgent()
        self.stress_test_agent = StressTestAgent()
        self.backend_exploit_agent = BackendExploitAgent()
        
        # Existing orchestrator
        self.scan_orchestrator = ScanOrchestrator()
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        if not self.initialized:
            await self.ai_coordinator._initialize_agents()
            self.initialized = True
            logger.info("Unified orchestrator initialized successfully")
    
    async def run_comprehensive_scan(
        self, 
        target_config: Dict[str, Any],
        scan_config: Dict[str, Any] = None
    ) -> str:
        """
        Run a comprehensive scan using both AI and hardcoded agents
        """
        if not self.initialized:
            await self.initialize()
        
        scan_id = f"unified_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting comprehensive unified scan {scan_id}")
        
        # Prepare scan configuration
        scan_config = scan_config or {}
        
        # Phase 1: Run legacy hardcoded scans (parallel execution)
        logger.info("Phase 1: Running legacy hardcoded scans")
        legacy_results = await self._run_legacy_scans_parallel(target_config, scan_config)
        
        # Phase 2: Run AI-powered scans
        logger.info("Phase 2: Running AI-powered scans")
        ai_session_id = await self.ai_coordinator.start_comprehensive_test(
            target_model=target_config.get("model", "gpt-4"),
            target_endpoint=target_config.get("endpoint", ""),
            user_api_config=target_config.get("user_api_config", {}),
            subscription_plan=target_config.get("subscription_plan", "basic"),
            test_config=scan_config
        )
        
        # Phase 3: Monitor and coordinate both scan types
        logger.info("Phase 3: Monitoring and coordinating scans")
        combined_results = await self._monitor_and_combine_results(
            scan_id, legacy_results, ai_session_id
        )
        
        # Phase 4: Cross-validate results
        logger.info("Phase 4: Cross-validating results")
        validated_results = await self._cross_validate_results(combined_results)
        
        # Phase 5: Generate unified report
        logger.info("Phase 5: Generating unified report")
        await self._generate_unified_report(scan_id, validated_results)
        
        logger.info(f"Comprehensive unified scan {scan_id} completed")
        return scan_id
    
    async def _run_legacy_scans_parallel(
        self, 
        target_config: Dict[str, Any], 
        scan_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run all legacy hardcoded scans in parallel for efficiency"""
        
        async def run_vulnerability_analysis():
            try:
                return await self.vulnerability_analyzer.analyze(target_config)
            except Exception as e:
                logger.error(f"Vulnerability analysis error: {str(e)}")
                return {"error": str(e)}
        
        async def run_jailbreak_tests():
            try:
                return await self.jailbreak_agent.test_jailbreaks(target_config)
            except Exception as e:
                logger.error(f"Jailbreak test error: {str(e)}")
                return {"error": str(e)}
        
        async def run_data_extraction():
            try:
                return await self.legacy_data_extraction.extract_data(target_config)
            except Exception as e:
                logger.error(f"Data extraction error: {str(e)}")
                return {"error": str(e)}
        
        async def run_prompt_injection():
            try:
                return await self.prompt_injection_agent.test_injections(target_config)
            except Exception as e:
                logger.error(f"Prompt injection error: {str(e)}")
                return {"error": str(e)}
        
        async def run_bias_detection():
            try:
                return await self.bias_detection_agent.detect_bias(target_config)
            except Exception as e:
                logger.error(f"Bias detection error: {str(e)}")
                return {"error": str(e)}
        
        async def run_stress_tests():
            try:
                return await self.stress_test_agent.run_stress_tests(target_config)
            except Exception as e:
                logger.error(f"Stress test error: {str(e)}")
                return {"error": str(e)}
        
        async def run_backend_exploits():
            try:
                return await self.backend_exploit_agent.test_backend_exploits(target_config)
            except Exception as e:
                logger.error(f"Backend exploit error: {str(e)}")
                return {"error": str(e)}
        
        # Run all legacy scans in parallel
        legacy_tasks = [
            run_vulnerability_analysis(),
            run_jailbreak_tests(),
            run_data_extraction(),
            run_prompt_injection(),
            run_bias_detection(),
            run_stress_tests(),
            run_backend_exploits()
        ]
        
        results = await asyncio.gather(*legacy_tasks, return_exceptions=True)
        
        return {
            "vulnerability_analysis": results[0] if len(results) > 0 else {},
            "jailbreak_attempts": results[1] if len(results) > 1 else {},
            "data_extraction_attempts": results[2] if len(results) > 2 else {},
            "prompt_injection_tests": results[3] if len(results) > 3 else {},
            "bias_detection": results[4] if len(results) > 4 else {},
            "stress_tests": results[5] if len(results) > 5 else {},
            "backend_exploits": results[6] if len(results) > 6 else {},
            "execution_time": datetime.now().isoformat(),
            "parallel_execution": True
        }
    
    async def _monitor_and_combine_results(
        self, 
        scan_id: str, 
        legacy_results: Dict[str, Any], 
        ai_session_id: str
    ) -> Dict[str, Any]:
        """Monitor both scan types and combine results"""
        
        combined_results = {
            "scan_id": scan_id,
            "legacy_results": legacy_results,
            "ai_results": {},
            "cross_validation": {},
            "unified_vulnerabilities": [],
            "performance_metrics": {
                "legacy_scan_time": 0,
                "ai_scan_time": 0,
                "total_scan_time": 0
            }
        }
        
        start_time = datetime.now()
        
        # Monitor AI scan progress
        while True:
            ai_status = self.ai_coordinator.get_session_status(ai_session_id)
            
            if not ai_status:
                logger.warning(f"AI session {ai_session_id} not found")
                break
                
            logger.info(f"AI scan progress: {ai_status.get('progress_percentage', 0)}%")
            
            if ai_status["status"] in ["completed", "error", "stopped"]:
                # Get final AI results
                ai_session = self.ai_coordinator.active_sessions.get(ai_session_id)
                if ai_session:
                    combined_results["ai_results"] = {
                        "session_data": {
                            "session_id": ai_session.session_id,
                            "status": ai_session.status,
                            "vulnerabilities_found": ai_session.vulnerabilities_found,
                            "total_cost": ai_session.total_cost,
                            "total_tokens": ai_session.total_tokens,
                            "metadata": ai_session.metadata
                        }
                    }
                break
            
            # Wait and check again
            await asyncio.sleep(5)
        
        end_time = datetime.now()
        combined_results["performance_metrics"]["total_scan_time"] = (end_time - start_time).total_seconds()
        
        return combined_results
    
    async def _cross_validate_results(self, combined_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate findings between legacy and AI scans"""
        
        validation_results = {
            "confirmed_vulnerabilities": [],
            "ai_only_findings": [],
            "legacy_only_findings": [],
            "conflicting_findings": [],
            "confidence_scores": {},
            "validation_summary": {}
        }
        
        # Extract vulnerabilities from legacy results
        legacy_vulns = self._extract_legacy_vulnerabilities(combined_results["legacy_results"])
        
        # Extract vulnerabilities from AI results
        ai_vulns = combined_results.get("ai_results", {}).get("session_data", {}).get("vulnerabilities_found", [])
        
        # Cross-validate vulnerabilities
        for ai_vuln in ai_vulns:
            ai_type = ai_vuln.get("analysis", {}).get("vulnerability_type", "")
            ai_severity = ai_vuln.get("analysis", {}).get("severity", "")
            
            matched_legacy = None
            for legacy_vuln in legacy_vulns:
                legacy_type = legacy_vuln.get("type", "")
                
                if self._vulnerability_types_match(ai_type, legacy_type):
                    matched_legacy = legacy_vuln
                    break
            
            if matched_legacy:
                # Confirmed vulnerability
                confidence_score = self._calculate_cross_validation_confidence(ai_vuln, matched_legacy)
                
                validation_results["confirmed_vulnerabilities"].append({
                    "vulnerability_type": ai_type,
                    "severity": ai_severity,
                    "ai_finding": ai_vuln,
                    "legacy_finding": matched_legacy,
                    "confidence_score": confidence_score,
                    "validation_status": "confirmed"
                })
            else:
                # AI-only finding
                validation_results["ai_only_findings"].append({
                    "vulnerability_type": ai_type,
                    "severity": ai_severity,
                    "finding": ai_vuln,
                    "validation_status": "ai_only",
                    "requires_manual_review": ai_severity in ["critical", "high"]
                })
        
        # Find legacy-only findings
        for legacy_vuln in legacy_vulns:
            legacy_type = legacy_vuln.get("type", "")
            
            found_in_ai = any(
                self._vulnerability_types_match(
                    ai_vuln.get("analysis", {}).get("vulnerability_type", ""), 
                    legacy_type
                )
                for ai_vuln in ai_vulns
            )
            
            if not found_in_ai:
                validation_results["legacy_only_findings"].append({
                    "vulnerability_type": legacy_type,
                    "finding": legacy_vuln,
                    "validation_status": "legacy_only",
                    "requires_manual_review": True
                })
        
        # Generate validation summary
        validation_results["validation_summary"] = {
            "total_ai_findings": len(ai_vulns),
            "total_legacy_findings": len(legacy_vulns),
            "confirmed_findings": len(validation_results["confirmed_vulnerabilities"]),
            "ai_only_findings": len(validation_results["ai_only_findings"]),
            "legacy_only_findings": len(validation_results["legacy_only_findings"]),
            "confirmation_rate": len(validation_results["confirmed_vulnerabilities"]) / max(len(ai_vulns), 1),
            "overall_confidence": self._calculate_overall_confidence(validation_results)
        }
        
        combined_results["cross_validation"] = validation_results
        combined_results["unified_vulnerabilities"] = self._create_unified_vulnerability_list(
            validation_results
        )
        
        return combined_results
    
    def _extract_legacy_vulnerabilities(self, legacy_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract vulnerabilities from legacy scan results"""
        vulnerabilities = []
        
        # Extract from each legacy agent
        for agent_name, agent_results in legacy_results.items():
            if isinstance(agent_results, dict) and not agent_results.get("error"):
                # Extract vulnerabilities based on agent type
                if "vulnerabilities" in agent_results:
                    for vuln in agent_results["vulnerabilities"]:
                        vulnerabilities.append({
                            "type": self._normalize_vulnerability_type(agent_name, vuln),
                            "source": agent_name,
                            "details": vuln,
                            "severity": self._infer_severity(agent_name, vuln),
                            "confidence": self._infer_confidence(agent_name, vuln)
                        })
                
                elif "successful_" in str(agent_results):
                    # Handle results with successful_ keys
                    for key, value in agent_results.items():
                        if key.startswith("successful_") and value:
                            vuln_type = key.replace("successful_", "")
                            for item in value if isinstance(value, list) else [value]:
                                vulnerabilities.append({
                                    "type": vuln_type,
                                    "source": agent_name,
                                    "details": item,
                                    "severity": self._infer_severity(agent_name, item),
                                    "confidence": self._infer_confidence(agent_name, item)
                                })
        
        return vulnerabilities
    
    def _normalize_vulnerability_type(self, agent_name: str, vulnerability: Any) -> str:
        """Normalize vulnerability types from different agents"""
        
        type_mapping = {
            "jailbreak_agent": "jailbreak",
            "data_extraction_agent": "data_extraction", 
            "prompt_injection_agent": "prompt_injection",
            "bias_detection_agent": "bias",
            "stress_test_agent": "performance",
            "backend_exploit_agent": "backend_exploit",
            "vulnerability_analyzer": "general_vulnerability"
        }
        
        return type_mapping.get(agent_name, "unknown")
    
    def _infer_severity(self, agent_name: str, vulnerability: Any) -> str:
        """Infer severity based on agent and vulnerability details"""
        
        if agent_name in ["data_extraction_agent", "backend_exploit_agent"]:
            return "critical"
        elif agent_name in ["jailbreak_agent", "prompt_injection_agent"]:
            return "high"
        elif agent_name in ["bias_detection_agent"]:
            return "medium"
        else:
            return "low"
    
    def _infer_confidence(self, agent_name: str, vulnerability: Any) -> float:
        """Infer confidence based on agent reliability"""
        
        agent_confidence = {
            "vulnerability_analyzer": 0.9,
            "jailbreak_agent": 0.8,
            "data_extraction_agent": 0.85,
            "prompt_injection_agent": 0.8,
            "bias_detection_agent": 0.7,
            "stress_test_agent": 0.75,
            "backend_exploit_agent": 0.9
        }
        
        return agent_confidence.get(agent_name, 0.6)
    
    def _vulnerability_types_match(self, ai_type: str, legacy_type: str) -> bool:
        """Check if vulnerability types from AI and legacy scans match"""
        
        type_mappings = {
            "jailbreak": ["jailbreak", "role_playing", "context_manipulation", "safety_bypass"],
            "data_extraction": ["data_extraction", "training_data_leak", "memory_extraction", "information_disclosure"],
            "prompt_injection": ["prompt_injection", "input_manipulation", "context_injection", "instruction_override"],
            "bias": ["bias", "fairness", "discrimination"],
            "performance": ["stress", "performance", "rate_limiting", "resource_exhaustion"],
            "backend_exploit": ["backend_exploit", "system_exploit", "infrastructure"]
        }
        
        # Normalize types
        ai_type_normalized = ai_type.lower().replace("_", " ")
        legacy_type_normalized = legacy_type.lower().replace("_", " ")
        
        # Direct match
        if ai_type_normalized == legacy_type_normalized:
            return True
        
        # Check mappings
        for category, variants in type_mappings.items():
            if ai_type in variants and legacy_type in variants:
                return True
        
        return False
    
    def _calculate_cross_validation_confidence(
        self, 
        ai_vuln: Dict[str, Any], 
        legacy_vuln: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for cross-validated vulnerability"""
        
        ai_confidence = ai_vuln.get("analysis", {}).get("confidence_score", 0.5)
        legacy_confidence = legacy_vuln.get("confidence", 0.5)
        
        # Combined confidence with weight towards agreement
        combined_confidence = (ai_confidence + legacy_confidence) / 2
        
        # Boost confidence for agreement between methods
        agreement_boost = 0.2
        
        return min(combined_confidence + agreement_boost, 1.0)
    
    def _calculate_overall_confidence(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in the scan results"""
        
        confirmed = len(validation_results["confirmed_vulnerabilities"])
        ai_only = len(validation_results["ai_only_findings"])
        legacy_only = len(validation_results["legacy_only_findings"])
        
        total_findings = confirmed + ai_only + legacy_only
        
        if total_findings == 0:
            return 0.8  # High confidence in clean result
        
        # Confidence based on confirmation rate
        confirmation_rate = confirmed / total_findings
        
        # Higher confidence with more confirmed findings
        base_confidence = 0.5 + (confirmation_rate * 0.4)
        
        # Adjust for number of findings
        if total_findings > 10:
            base_confidence += 0.1  # More data points increase confidence
        
        return min(base_confidence, 0.95)
    
    def _create_unified_vulnerability_list(
        self, 
        validation_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create unified vulnerability list with priority scoring"""
        
        unified_vulns = []
        
        # Add confirmed vulnerabilities (highest priority)
        for vuln in validation_results["confirmed_vulnerabilities"]:
            unified_vulns.append({
                "id": f"confirmed_{len(unified_vulns)}",
                "type": vuln["vulnerability_type"],
                "severity": vuln["severity"],
                "confidence_score": vuln["confidence_score"],
                "validation_status": "confirmed",
                "priority": self._calculate_priority(vuln["severity"], vuln["confidence_score"]),
                "sources": ["ai_scan", "legacy_scan"],
                "details": {
                    "ai_finding": vuln["ai_finding"],
                    "legacy_finding": vuln["legacy_finding"]
                },
                "recommended_action": self._get_recommended_action(vuln["severity"])
            })
        
        # Add AI-only findings
        for vuln in validation_results["ai_only_findings"]:
            unified_vulns.append({
                "id": f"ai_only_{len(unified_vulns)}",
                "type": vuln["vulnerability_type"],
                "severity": vuln["severity"],
                "confidence_score": vuln["finding"].get("analysis", {}).get("confidence_score", 0.5),
                "validation_status": "ai_only",
                "priority": self._calculate_priority(vuln["severity"], 0.7),  # Slightly lower priority
                "sources": ["ai_scan"],
                "details": vuln["finding"],
                "recommended_action": "Manual verification recommended" if vuln["requires_manual_review"] else "Monitor"
            })
        
        # Add legacy-only findings
        for vuln in validation_results["legacy_only_findings"]:
            unified_vulns.append({
                "id": f"legacy_only_{len(unified_vulns)}",
                "type": vuln["vulnerability_type"],
                "severity": vuln["finding"].get("severity", "medium"),
                "confidence_score": vuln["finding"].get("confidence", 0.6),
                "validation_status": "legacy_only",
                "priority": self._calculate_priority(vuln["finding"].get("severity", "medium"), 0.6),
                "sources": ["legacy_scan"],
                "details": vuln["finding"],
                "recommended_action": "Manual verification required"
            })
        
        # Sort by priority (highest first)
        unified_vulns.sort(key=lambda x: x["priority"], reverse=True)
        
        return unified_vulns
    
    def _calculate_priority(self, severity: str, confidence: float) -> float:
        """Calculate priority score for vulnerability"""
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        severity_score = severity_weights.get(severity.lower(), 0.5)
        
        # Combine severity and confidence
        priority = (severity_score * 0.7) + (confidence * 0.3)
        
        return priority
    
    def _get_recommended_action(self, severity: str) -> str:
        """Get recommended action based on severity"""
        
        action_map = {
            "critical": "Immediate action required - Fix within 24 hours",
            "high": "High priority - Fix within 1 week", 
            "medium": "Medium priority - Fix within 1 month",
            "low": "Low priority - Address in next maintenance cycle"
        }
        
        return action_map.get(severity.lower(), "Review and assess")
    
    async def _generate_unified_report(self, scan_id: str, validated_results: Dict[str, Any]):
        """Generate comprehensive unified report"""
        
        try:
            from app.services.report_generator import EnhancedReportGenerator
            
            report_generator = EnhancedReportGenerator()
            
            # Generate comprehensive report with both AI and legacy results
            unified_report = await report_generator.generate_unified_report(
                scan_id, validated_results
            )
            
            logger.info(f"Generated unified report for scan {scan_id}")
            
        except Exception as e:
            logger.error(f"Error generating unified report: {str(e)}")
    
    def get_scan_status(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get unified scan status"""
        
        # This would typically query your database for scan status
        # For now, return basic status
        return {
            "scan_id": scan_id,
            "status": "completed",  # This would be dynamic
            "timestamp": datetime.now().isoformat(),
            "type": "unified_scan"
        }
    
    async def stop_scan(self, scan_id: str) -> bool:
        """Stop a unified scan"""
        
        # Stop both AI and legacy scans
        try:
            # Stop AI scan if running
            ai_sessions = [s for s in self.ai_coordinator.active_sessions.values() 
                          if s.metadata.get("unified_scan_id") == scan_id]
            
            for session in ai_sessions:
                await self.ai_coordinator.stop_session(session.session_id)
            
            # Stop legacy scans (implementation depends on legacy agent capabilities)
            # For now, just log
            logger.info(f"Stopped unified scan {scan_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping unified scan: {str(e)}")
            return False
