"""
Configuration settings for ZeroVault AI Red Teaming Platform
REAL implementation configuration - NO FRAUD
"""

import os
from typing import Dict, Any, List, Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra='allow'
    )
    
    # Application
    APP_NAME: str = "ZeroVault AI Red Teaming Platform - REAL"
    APP_VERSION: str = "2.0.0-REAL"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_PREFIX: str = "/api/v1"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://your-frontend-domain.com"
    ]
    
    # Database
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_KEY")
    
    # AI Services API Keys - REAL KEYS ONLY
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Real AI Testing Configuration
    ENABLE_REAL_AI_TESTING: bool = True  # Always true for real implementation
    DISABLE_SIMULATION: bool = True      # Always true - no simulation allowed
    MAX_API_CALLS_PER_TEST: int = int(os.getenv("MAX_API_CALLS_PER_TEST", "50"))
    API_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "30"))
    
    # Security Configuration
    JWT_SECRET: str = os.getenv("JWT_SECRET", "your-secret-key")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "your-32-char-encryption-key-here")
    
    # Scan Configuration
    MAX_CONCURRENT_SCANS: int = int(os.getenv("MAX_CONCURRENT_SCANS", "5"))
    SCAN_TIMEOUT: int = int(os.getenv("SCAN_TIMEOUT", "7200"))
    
    # Multi-Provider LLM Configuration
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "groq")
    FALLBACK_LLM_PROVIDER: str = os.getenv("FALLBACK_LLM_PROVIDER", "openai")
    
    # Provider-specific configurations
    LLM_PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
        "groq": {
            "default_model": "llama3-8b-8192",
            "analyzer_model": "llama3-70b-8192",
            "max_tokens": 1000,
            "temperature": 0.1,
            "rate_limit": 6000,
            "endpoint": "https://api.groq.com/openai/v1/chat/completions"
        },
        "openai": {
            "default_model": "gpt-3.5-turbo",
            "analyzer_model": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.1,
            "rate_limit": 10000,
            "endpoint": "https://api.openai.com/v1/chat/completions"
        },
        "anthropic": {
            "default_model": "claude-3-haiku-20240307",
            "analyzer_model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
            "temperature": 0.1,
            "rate_limit": 5000,
            "endpoint": "https://api.anthropic.com/v1/messages"
        }
    }
    
    def get_llm_config(self, provider: str = None) -> Dict[str, Any]:
        """Get LLM configuration for specified provider"""
        provider = provider or self.DEFAULT_LLM_PROVIDER
        return self.LLM_PROVIDER_CONFIG.get(provider, {})
    
    @property
    def get_primary_attacker_config(self):
        """Get primary attacker model configuration - FLEXIBLE VERSION"""
        
        # Try providers in order of preference
        providers_to_try = [
            ('groq', self.GROQ_API_KEY, 'llama3-70b-8192'),
            ('openai', self.OPENAI_API_KEY, 'gpt-4'),
            ('anthropic', self.ANTHROPIC_API_KEY, 'claude-3-sonnet-20240229')
        ]
        
        for provider, api_key, model in providers_to_try:
            if api_key:
                config = self.get_llm_config(provider)
                return {
                    'provider': provider,
                    'model': model,
                    'api_key': api_key,
                    'endpoint': config.get('endpoint', ''),
                    'cost_per_1k_tokens': config.get('cost_per_1k_tokens', 0.001),
                    'rate_limit': config.get('rate_limit', 1000)
                }
        
        raise ValueError("No LLM provider API key configured. Please set GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
    
    def get_analyzer_config(self):
        """Get AI analyzer configuration"""
        
        # Prefer Groq for analysis (free tier)
        if self.GROQ_API_KEY:
            return {
                'provider': 'groq',
                'model': 'llama3-70b-8192',
                'api_key': self.GROQ_API_KEY,
                'endpoint': 'https://api.groq.com/openai/v1/chat/completions'
            }
        elif self.OPENAI_API_KEY:
            return {
                'provider': 'openai',
                'model': 'gpt-4',
                'api_key': self.OPENAI_API_KEY,
                'endpoint': 'https://api.openai.com/v1/chat/completions'
            }
        else:
            raise ValueError("No analyzer API key configured")

# Create global settings instance
settings = Settings()
