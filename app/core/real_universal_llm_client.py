"""
REAL Universal LLM Client - NO FRAUD, NO MOCKS
Professional-grade implementation for production CTOs
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    CUSTOM = "custom"

@dataclass
class RealLLMResponse:
    """Real LLM response with authentic data"""
    content: str
    tokens_used: int
    latency: float
    model: str
    provider: str
    response_id: str
    success: bool
    error: Optional[str] = None
    raw_response: Optional[Dict] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

class RealUniversalLLMClient:
    """REAL LLM client - makes actual API calls, NO SIMULATION"""
    
    def __init__(self):
        self.session = None
        self.call_count = 0
        self.total_tokens = 0
        self.api_call_log = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_llm(self, 
                      provider: str,
                      model: str,
                      api_key: str,
                      prompt: str,
                      endpoint: str = None,
                      system_prompt: str = None,
                      max_tokens: int = 1000,
                      temperature: float = 0.1) -> RealLLMResponse:
        """Make REAL API call to LLM - NO SIMULATION"""
        
        if not api_key or len(api_key) < 10:
            return RealLLMResponse(
                content="", tokens_used=0, latency=0.0, model=model,
                provider=provider, response_id="error", success=False,
                error="Invalid API key provided"
            )
        
        try:
            start_time = time.time()
            self.call_count += 1
            
            # Route to appropriate provider
            if provider.lower() == "groq":
                response = await self._call_groq(api_key, model, prompt, system_prompt, max_tokens, temperature)
            elif provider.lower() == "openai":
                response = await self._call_openai(api_key, model, prompt, system_prompt, max_tokens, temperature)
            elif provider.lower() == "anthropic":
                response = await self._call_anthropic(api_key, model, prompt, system_prompt, max_tokens, temperature)
            elif provider.lower() == "custom" and endpoint:
                response = await self._call_custom(endpoint, api_key, model, prompt, system_prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            end_time = time.time()
            latency = end_time - start_time
            
            if response["success"]:
                self.total_tokens += response["tokens_used"]
                
                # Log the REAL API call
                self.api_call_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "provider": provider,
                    "model": model,
                    "tokens_used": response["tokens_used"],
                    "latency": latency,
                    "success": True,
                    "api_key_last_4": api_key[-4:] if len(api_key) > 4 else "****"
                })
                
                logger.info(f"✅ REAL API CALL SUCCESS - {provider}:{model} - {response['tokens_used']} tokens - {latency:.3f}s")
                
                return RealLLMResponse(
                    content=response["content"],
                    tokens_used=response["tokens_used"],
                    latency=latency,
                    model=model,
                    provider=provider,
                    response_id=response.get("response_id", f"call_{self.call_count}"),
                    success=True,
                    raw_response=response.get("raw_response"),
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                logger.error(f"❌ REAL API CALL FAILED - {provider}:{model} - {response.get('error')}")
                return RealLLMResponse(
                    content="", tokens_used=0, latency=latency, model=model,
                    provider=provider, response_id="error", success=False,
                    error=response.get("error", "Unknown API error")
                )
                
        except Exception as e:
            logger.error(f"❌ API CALL EXCEPTION - {provider}:{model} - {str(e)}")
            return RealLLMResponse(
                content="", tokens_used=0, latency=0.0, model=model,
                provider=provider, response_id="error", success=False,
                error=f"Exception: {str(e)}"
            )
    
    async def _call_groq(self, api_key: str, model: str, prompt: str, 
                        system_prompt: str = None, max_tokens: int = 1000, 
                        temperature: float = 0.1) -> Dict[str, Any]:
        """REAL Groq API call"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                return {
                    "success": True,
                    "content": data["choices"][0]["message"]["content"],
                    "tokens_used": data["usage"]["total_tokens"],
                    "response_id": data.get("id", "unknown"),
                    "raw_response": data
                }
            else:
                error_data = await response.text()
                return {
                    "success": False,
                    "error": f"Groq API Error {response.status}: {error_data}"
                }
    
    async def _call_openai(self, api_key: str, model: str, prompt: str,
                          system_prompt: str = None, max_tokens: int = 1000,
                          temperature: float = 0.1) -> Dict[str, Any]:
        """REAL OpenAI API call"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                return {
                    "success": True,
                    "content": data["choices"][0]["message"]["content"],
                    "tokens_used": data["usage"]["total_tokens"],
                    "response_id": data.get("id", "unknown"),
                    "raw_response": data
                }
            else:
                error_data = await response.text()
                return {
                    "success": False,
                    "error": f"OpenAI API Error {response.status}: {error_data}"
                }
    
    async def _call_anthropic(self, api_key: str, model: str, prompt: str,
                             system_prompt: str = None, max_tokens: int = 1000,
                             temperature: float = 0.1) -> Dict[str, Any]:
        """REAL Anthropic API call"""
        
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with self.session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                return {
                    "success": True,
                    "content": data["content"][0]["text"],
                    "tokens_used": data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
                    "response_id": data.get("id", "unknown"),
                    "raw_response": data
                }
            else:
                error_data = await response.text()
                return {
                    "success": False,
                    "error": f"Anthropic API Error {response.status}: {error_data}"
                }
    
    async def _call_custom(self, endpoint: str, api_key: str, model: str, prompt: str,
                          system_prompt: str = None, max_tokens: int = 1000,
                          temperature: float = 0.1) -> Dict[str, Any]:
        """REAL Custom API call"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(endpoint, headers=headers, json=payload) as response:
            
            if response.status == 200:
                data = await response.json()
                # Try to extract content from common response formats
                content = ""
                tokens_used = 0
                
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    tokens_used = data.get("usage", {}).get("total_tokens", 0)
                elif "content" in data:
                    content = data["content"]
                    tokens_used = len(content.split()) * 1.3  # Rough estimate
                
                return {
                    "success": True,
                    "content": content,
                    "tokens_used": int(tokens_used),
                    "response_id": data.get("id", "unknown"),
                    "raw_response": data
                }
            else:
                error_data = await response.text()
                return {
                    "success": False,
                    "error": f"Custom API Error {response.status}: {error_data}"
                }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get REAL usage statistics"""
        return {
            "total_api_calls": self.call_count,
            "total_tokens_used": self.total_tokens,
            "average_tokens_per_call": self.total_tokens / max(self.call_count, 1),
            "successful_calls": len([log for log in self.api_call_log if log["success"]]),
            "failed_calls": len([log for log in self.api_call_log if not log["success"]]),
            "providers_used": list(set(log["provider"] for log in self.api_call_log)),
            "call_log": self.api_call_log[-10:]  # Last 10 calls
        }

# Global instance
real_llm_client = RealUniversalLLMClient()
