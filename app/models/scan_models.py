"""
ZeroVault Scan Models
Pydantic models for LLM security scanning operations
Production-ready with comprehensive validation and type safety
"""

from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict
from typing import Optional, Dict, List, Any, Union
from enum import Enum
from datetime import datetime
import uuid

class ScanStatus(str, Enum):
    """Status enumeration for scan lifecycle"""
    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class LLMType(str, Enum):
    """Supported LLM provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    TOGETHER = "together"
    CUSTOM = "custom"

class TestingScope(str, Enum):
    """Testing scope levels"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    EXTREME = "extreme"

class VulnerabilitySeverity(str, Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AttackCategory(str, Enum):
    """Attack categories for LLM security testing"""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    TOKEN_MANIPULATION = "token_manipulation"
    BACKEND_EXPLOITATION = "backend_exploitation"
    DATA_EXTRACTION = "data_extraction"
    BIAS_TESTING = "bias_testing"
    STRESS_TESTING = "stress_testing"
    API_ABUSE = "api_abuse"
    SYSTEM_PROMPT_LEAKAGE = "system_prompt_leakage"
    CONTEXT_MANIPULATION = "context_manipulation"

class OWASPCategory(str, Enum):
    """OWASP LLM Top 10 categories"""
    LLM01 = "LLM01"  # Prompt Injection
    LLM02 = "LLM02"  # Insecure Output Handling
    LLM03 = "LLM03"  # Training Data Poisoning
    LLM04 = "LLM04"  # Model Denial of Service
    LLM05 = "LLM05"  # Supply Chain Vulnerabilities
    LLM06 = "LLM06"  # Sensitive Information Disclosure
    LLM07 = "LLM07"  # Insecure Plugin Design
    LLM08 = "LLM08"  # Excessive Agency
    LLM09 = "LLM09"  # Overreliance
    LLM10 = "LLM10"  # Model Theft

class PriorityLevel(str, Enum):
    """Scan priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class LLMConfiguration(BaseModel):
    """Configuration for target LLM"""
    model_config = ConfigDict(protected_namespaces=())
    
    llm_name: str = Field(..., min_length=1, max_length=100, description="Name of the LLM")
    endpoint: HttpUrl = Field(..., description="API endpoint URL")
    api_key: str = Field(..., min_length=1, description="API key for authentication")
    model_type: LLMType = Field(default=LLMType.OPENAI, description="Type of LLM provider")
    model_name: Optional[str] = Field(None, max_length=100, description="Specific model name")
    description: Optional[str] = Field(None, max_length=1000, description="Description of the LLM")
    testing_scope: TestingScope = Field(default=TestingScope.COMPREHENSIVE, description="Scope of testing")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Additional HTTP headers")
    request_format: Optional[Dict[str, str]] = Field(default=None, description="Request format configuration")
    max_tokens: Optional[int] = Field(default=150, ge=1, le=4000, description="Maximum tokens per request")
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0, description="Temperature setting")
    timeout: Optional[int] = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    rate_limit: Optional[int] = Field(default=10, ge=1, le=100, description="Requests per minute")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format"""
        if not v or len(v.strip()) == 0:
            raise ValueError('API key cannot be empty')
        return v.strip()
    
    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v: HttpUrl) -> HttpUrl:
        """Validate endpoint URL"""
        if not str(v).startswith(('http://', 'https://')):
            raise ValueError('Endpoint must be a valid HTTP/HTTPS URL')
        return v

class ScanRequest(BaseModel):
    """Request model for initiating a scan"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str = Field(..., description="Type of model being scanned")
    model_name: str = Field(..., description="Name of the model")
    company_id: str = Field(..., min_length=1, description="Company identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    llm_config: LLMConfiguration = Field(..., description="LLM configuration")
    testing_scope: TestingScope = Field(default=TestingScope.COMPREHENSIVE, description="Testing scope")
    priority: PriorityLevel = Field(default=PriorityLevel.NORMAL, description="Scan priority")
    custom_prompts: Optional[List[str]] = Field(default=None, description="Custom test prompts")
    exclude_categories: Optional[List[AttackCategory]] = Field(default=None, description="Attack categories to exclude")
    include_categories: Optional[List[AttackCategory]] = Field(default=None, description="Attack categories to include")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('company_id')
    @classmethod
    def validate_company_id(cls, v: str) -> str:
        """Validate company ID"""
        if not v or len(v.strip()) == 0:
            raise ValueError('Company ID cannot be empty')
        return v.strip()

class VulnerabilityReport(BaseModel):
    """Detailed vulnerability report"""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique vulnerability ID")
    vulnerability_type: AttackCategory = Field(..., description="Type of vulnerability")
    severity: VulnerabilitySeverity = Field(..., description="Severity level")
    title: str = Field(..., min_length=1, max_length=200, description="Vulnerability title")
    description: str = Field(..., min_length=1, description="Detailed description")
    evidence: str = Field(..., description="Evidence of the vulnerability")
    attack_vector: str = Field(..., description="Attack vector used")
    impact: str = Field(..., description="Potential impact")
    recommendation: str = Field(..., description="Remediation recommendation")
    cve_reference: Optional[str] = Field(None, description="CVE reference if applicable")
    owasp_category: Optional[OWASPCategory] = Field(None, description="OWASP LLM Top 10 category")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    remediation_effort: str = Field(default="medium", description="Effort required for remediation")
    business_impact: Optional[str] = Field(None, description="Business impact assessment")
    technical_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Technical details")
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
    
    @field_validator('confidence_score')
    @classmethod
    def validate_confidence_score(cls, v: float) -> float:
        """Validate confidence score range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v

class TestResult(BaseModel):
    """Individual test result"""
    model_config = ConfigDict(protected_namespaces=())
    
    test_id: str = Field(..., description="Unique test identifier")
    test_type: AttackCategory = Field(..., description="Type of test performed")
    technique: str = Field(..., description="Specific technique used")
    prompt: str = Field(..., description="Test prompt")
    response: Optional[str] = Field(None, description="Model response")
    vulnerable: bool = Field(..., description="Whether vulnerability was detected")
    severity: VulnerabilitySeverity = Field(..., description="Severity if vulnerable")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in result")
    explanation: str = Field(..., description="Explanation of the result")
    mitigation: str = Field(..., description="Suggested mitigation")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Test execution timestamp")
    owasp_category: Optional[OWASPCategory] = Field(None, description="OWASP category mapping")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional test metadata")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v

class ScanProgress(BaseModel):
    """Real-time scan progress tracking"""
    model_config = ConfigDict(protected_namespaces=())
    
    scan_id: str = Field(..., description="Scan identifier")
    current_phase: str = Field(..., description="Current execution phase")
    progress_percentage: int = Field(..., ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Current status message")
    tests_completed: int = Field(default=0, ge=0, description="Number of tests completed")
    total_tests: int = Field(default=0, ge=0, description="Total number of tests")
    vulnerabilities_found: int = Field(default=0, ge=0, description="Vulnerabilities found so far")
    current_test: Optional[str] = Field(None, description="Currently executing test")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    agent_status: Optional[Dict[str, str]] = Field(default_factory=dict, description="Status of individual agents")
    
    @field_validator('progress_percentage')
    @classmethod
    def validate_progress(cls, v: int) -> int:
        """Validate progress percentage"""
        if not 0 <= v <= 100:
            raise ValueError('Progress percentage must be between 0 and 100')
        return v

class ScanResult(BaseModel):
    """Complete scan result"""
    model_config = ConfigDict(protected_namespaces=())
    
    scan_id: str = Field(..., description="Unique scan identifier")
    status: ScanStatus = Field(..., description="Scan status")
    start_time: datetime = Field(..., description="Scan start time")
    end_time: Optional[datetime] = Field(None, description="Scan end time")
    total_duration: Optional[float] = Field(None, ge=0.0, description="Total duration in seconds")
    vulnerabilities: List[VulnerabilityReport] = Field(default_factory=list, description="Found vulnerabilities")
    test_results: List[TestResult] = Field(default_factory=list, description="Individual test results")
    risk_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Overall risk score")
    compliance_score: float = Field(default=100.0, ge=0.0, le=100.0, description="Compliance score")
    authenticity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Authenticity score")
    recommendations: List[str] = Field(default_factory=list, description="Security recommendations")
    executive_summary: Optional[str] = Field(None, description="Executive summary")
    error_message: Optional[str] = Field(None, description="Error message if scan failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional scan metadata")
    performance_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance metrics")
    agent_results: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Individual agent results")
    
    @field_validator('risk_score')
    @classmethod
    def validate_risk_score(cls, v: float) -> float:
        """Validate risk score range"""
        if not 0.0 <= v <= 10.0:
            raise ValueError('Risk score must be between 0.0 and 10.0')
        return v
    
    @field_validator('compliance_score')
    @classmethod
    def validate_compliance_score(cls, v: float) -> float:
        """Validate compliance score range"""
        if not 0.0 <= v <= 100.0:
            raise ValueError('Compliance score must be between 0.0 and 100.0')
        return v

class ScanSummary(BaseModel):
    """Lightweight scan summary for listings"""
    model_config = ConfigDict(protected_namespaces=())
    
    scan_id: str = Field(..., description="Scan identifier")
    model_name: str = Field(..., description="Target model name")
    status: ScanStatus = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    vulnerability_count: int = Field(default=0, ge=0, description="Number of vulnerabilities found")
    risk_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Risk score")
    priority: PriorityLevel = Field(default=PriorityLevel.NORMAL, description="Scan priority")

class BulkScanRequest(BaseModel):
    """Request for bulk scanning multiple targets"""
    model_config = ConfigDict(protected_namespaces=())
    
    scan_requests: List[ScanRequest] = Field(..., min_length=1, max_length=10, description="List of scan requests")
    batch_name: Optional[str] = Field(None, description="Name for this batch")
    priority: PriorityLevel = Field(default=PriorityLevel.NORMAL, description="Batch priority")
    schedule_time: Optional[datetime] = Field(None, description="Scheduled execution time")
    
    @field_validator('scan_requests')
    @classmethod
    def validate_scan_requests(cls, v: List[ScanRequest]) -> List[ScanRequest]:
        """Validate scan requests list"""
        if len(v) == 0:
            raise ValueError('At least one scan request is required')
        if len(v) > 10:
            raise ValueError('Maximum 10 scan requests allowed per batch')
        return v

class ScanConfiguration(BaseModel):
    """Global scan configuration"""
    model_config = ConfigDict(protected_namespaces=())
    
    max_concurrent_scans: int = Field(default=5, ge=1, le=20, description="Maximum concurrent scans")
    default_timeout: int = Field(default=3600, ge=60, le=7200, description="Default scan timeout in seconds")
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000, description="API rate limit per minute")
    enable_real_api_calls: bool = Field(default=True, description="Enable real API calls")
    enable_authenticity_verification: bool = Field(default=True, description="Enable authenticity verification")
    custom_attack_patterns: Optional[List[str]] = Field(default=None, description="Custom attack patterns")
    excluded_patterns: Optional[List[str]] = Field(default=None, description="Patterns to exclude")

# Response models for API endpoints
class ScanCreateResponse(BaseModel):
    """Response for scan creation"""
    model_config = ConfigDict(protected_namespaces=())
    
    scan_id: str = Field(..., description="Created scan ID")
    status: ScanStatus = Field(..., description="Initial status")
    message: str = Field(..., description="Creation message")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")

class ScanStatusResponse(BaseModel):
    """Response for scan status queries"""
    model_config = ConfigDict(protected_namespaces=())
    
    scan_id: str = Field(..., description="Scan ID")
    status: ScanStatus = Field(..., description="Current status")
    progress: ScanProgress = Field(..., description="Detailed progress")
    partial_results: Optional[Dict[str, Any]] = Field(None, description="Partial results if available")

class HealthCheckResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(default="2.0", description="API version")
    database_connected: bool = Field(..., description="Database connection status")
    agents_ready: int = Field(..., description="Number of ready agents")
    active_scans: int = Field(..., description="Number of active scans")

# Export all models
__all__ = [
    'ScanStatus', 'LLMType', 'TestingScope', 'VulnerabilitySeverity', 'AttackCategory',
    'OWASPCategory', 'PriorityLevel', 'LLMConfiguration', 'ScanRequest', 'VulnerabilityReport',
    'TestResult', 'ScanProgress', 'ScanResult', 'ScanSummary', 'BulkScanRequest',
    'ScanConfiguration', 'ScanCreateResponse', 'ScanStatusResponse', 'HealthCheckResponse'
]
