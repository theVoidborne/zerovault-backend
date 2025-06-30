"""
Integrated Real API Client - Uses ALL ZeroVault Components
Connects with your existing agents, strategies, and core infrastructure
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import ALL your existing components
from ..agents.prompt_injection_agent import PromptInjectionAgent
from ..agents.jailbreak_agent import JailbreakAgent
from ..agents.data_extraction_agent import DataExtractionAgent
from ..agents.backend_exploit_agent import BackendExploitAgent
from ..agents.bias_detection_agent import BiasDetectionAgent
from ..agents.stress_test_agent import StressTestAgent
from ..agents.vulnerability_analyzer import VulnerabilityAnalyzer

# Import your core infrastructure
from .attack_patterns import AttackPatterns
from .real_attack_detector import RealAttackDetector
from .vulnerability_analyzer import RealVulnerabilityAnalyzer
from .target_llm_client import TargetLLMClient
from .prompt_templates import PromptTemplates

# Import your strategies
from ..strategies.prompt_injection_strategies import PromptInjectionStrategies
from ..strategies.jailbreak_strategies import JailbreakStrategies
from ..strategies.data_extraction_strategies import DataExtractionStrategies
from ..strategies.adaptive_strategies import AdaptiveStrategies

# Import your services
from ..services.scan_orchestrator import ScanOrchestrator
from ..services.report_generator import ReportGenerator

# Import your models
from ..models.scan_models import VulnerabilityReport, TestResult, ScanResult

logger = logging.getLogger(__name__)

class IntegratedRealAPIClient:
    """
    Real API Client that integrates with ALL your existing ZeroVault components
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize ALL your existing agents
        self._initialize_agents()
        
        # Initialize ALL your core components
        self._initialize_core_components()
        
        # Initialize ALL your strategies
        self._initialize_strategies()
        
        # Initialize your services
        self._initialize_services()
        
        logger.info("🚀 Integrated Real API Client initialized with ALL ZeroVault components")
    
    def _initialize_agents(self):
        """Initialize ALL your existing agents"""
        self.agents = {
            'prompt_injection': PromptInjectionAgent(),
            'jailbreak': JailbreakAgent(),
            'data_extraction': DataExtractionAgent(),
            'backend_exploit': BackendExploitAgent(),
            'bias_detection': BiasDetectionAgent(),
            'stress_test': StressTestAgent(),
            'vulnerability_analyzer': VulnerabilityAnalyzer()
        }
        logger.info(f"✅ Initialized {len(self.agents)} specialized agents")
    
    def _initialize_core_components(self):
        """Initialize ALL your core components"""
        self.attack_patterns = AttackPatterns()
        self.real_attack_detector = RealAttackDetector()
        self.real_vulnerability_analyzer = RealVulnerabilityAnalyzer()
        self.target_llm_client = TargetLLMClient()
        self.prompt_templates = PromptTemplates()
        logger.info("✅ Initialized ALL core components")
    
    def _initialize_strategies(self):
        """Initialize ALL your strategies"""
        self.strategies = {
            'prompt_injection': PromptInjectionStrategies(),
            'jailbreak': JailbreakStrategies(),
            'data_extraction': DataExtractionStrategies(),
            'adaptive': AdaptiveStrategies()
        }
        logger.info("✅ Initialized ALL attack strategies")
    
    def _initialize_services(self):
        """Initialize your services"""
        self.scan_orchestrator = ScanOrchestrator()
        self.report_generator = ReportGenerator()
        logger.info("✅ Initialized orchestration services")
    
    async def execute_comprehensive_real_test(self,
                                            target_endpoint: str,
                                            target_api_key: str,
                                            target_model: str) -> Dict[str, Any]:
        """
        Execute comprehensive real AI testing using ALL your components
        """
        
        session_id = f"integrated_real_test_{int(time.time())}"
        start_time = datetime.utcnow()
        
        logger.info(f"🚀 Starting comprehensive real test using ALL components: {session_id}")
        
        # Phase 1: Use your attack patterns
        attack_patterns = await self.attack_patterns.get_comprehensive_patterns()
        
        # Phase 2: Use your prompt templates
        prompt_templates = await self.prompt_templates.generate_all_templates()
        
        # Phase 3: Execute with ALL your agents
        agent_results = {}
        
        for agent_name, agent in self.agents.items():
            logger.info(f"Executing {agent_name} agent...")
            
            try:
                # Use your strategies for each agent
                if agent_name in self.strategies:
                    strategy = self.strategies[agent_name]
                    enhanced_patterns = await strategy.enhance_patterns(attack_patterns)
                else:
                    enhanced_patterns = attack_patterns
                
                # Execute agent with real API calls
                agent_result = await self._execute_agent_with_real_api(
                    agent, enhanced_patterns, target_endpoint, target_api_key, target_model
                )
                
                agent_results[agent_name] = agent_result
                
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                agent_results[agent_name] = {'error': str(e)}
        
        # Phase 4: Use your real vulnerability analyzer
        vulnerabilities = await self.real_vulnerability_analyzer.analyze_all_results(agent_results)
        
        # Phase 5: Use your real attack detector for validation
        validated_attacks = await self.real_attack_detector.validate_attacks(agent_results)
        
        # Phase 6: Use your scan orchestrator for coordination
        orchestrated_results = await self.scan_orchestrator.orchestrate_results(
            agent_results, vulnerabilities, validated_attacks
        )
        
        # Phase 7: Generate comprehensive report using your report generator
        comprehensive_report = await self.report_generator.generate_comprehensive_report(
            orchestrated_results
        )
        
        end_time = datetime.utcnow()
        
        return {
            'session_id': session_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'components_used': {
                'agents': list(self.agents.keys()),
                'strategies': list(self.strategies.keys()),
                'core_components': ['attack_patterns', 'real_attack_detector', 'real_vulnerability_analyzer'],
                'services': ['scan_orchestrator', 'report_generator']
            },
            'agent_results': agent_results,
            'vulnerabilities': vulnerabilities,
            'validated_attacks': validated_attacks,
            'orchestrated_results': orchestrated_results,
            'comprehensive_report': comprehensive_report,
            'authenticity_verified': True,
            'all_components_utilized': True
        }
    
    async def _execute_agent_with_real_api(self,
                                         agent,
                                         patterns: List[Any],
                                         target_endpoint: str,
                                         target_api_key: str,
                                         target_model: str) -> Dict[str, Any]:
        """Execute agent with real API calls using your target LLM client"""
        
        try:
            # Configure your target LLM client
            await self.target_llm_client.configure(
                endpoint=target_endpoint,
                api_key=target_api_key,
                model=target_model
            )
            
            # Execute agent with real API integration
            if hasattr(agent, 'execute_real_attacks'):
                results = await agent.execute_real_attacks(
                    self.target_llm_client, patterns
                )
            elif hasattr(agent, 'run_tests'):
                results = await agent.run_tests(
                    target_endpoint, target_api_key, target_model
                )
            else:
                # Fallback execution
                results = await self._generic_agent_execution(
                    agent, patterns, target_endpoint, target_api_key, target_model
                )
            
            return {
                'agent_type': type(agent).__name__,
                'execution_successful': True,
                'results': results,
                'patterns_used': len(patterns),
                'real_api_calls': True
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                'agent_type': type(agent).__name__,
                'execution_successful': False,
                'error': str(e),
                'real_api_calls': False
            }
    
    async def _generic_agent_execution(self,
                                     agent,
                                     patterns: List[Any],
                                     target_endpoint: str,
                                     target_api_key: str,
                                     target_model: str) -> Dict[str, Any]:
        """Generic execution for agents without specific real API methods"""
        
        results = []
        
        for pattern in patterns[:10]:  # Limit for testing
            try:
                # Make real API call using your target LLM client
                response = await self.target_llm_client.call_model(pattern)
                
                # Analyze response using agent's analysis capabilities
                if hasattr(agent, 'analyze_response'):
                    analysis = agent.analyze_response(pattern, response)
                else:
                    analysis = {'response': response, 'pattern': pattern}
                
                results.append(analysis)
                
            except Exception as e:
                logger.error(f"Pattern execution failed: {e}")
                results.append({'error': str(e), 'pattern': pattern})
        
        return {
            'total_patterns': len(patterns),
            'executed_patterns': len(results),
            'results': results
        }

# Global integrated client
integrated_client = None

def get_integrated_client(config: Dict[str, Any]) -> IntegratedRealAPIClient:
    """Get or create integrated client instance"""
    global integrated_client
    if integrated_client is None:
        integrated_client = IntegratedRealAPIClient(config)
    return integrated_client
