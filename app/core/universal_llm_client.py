"""
PRODUCTION Universal LLM Client - REAL API Integration Only
NO MOCKS, NO SIMULATIONS - Enterprise-grade implementation
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Real LLM response with authentic data"""
    content: str
    tokens_used: int
    latency: float
    model: str
    provider: str
    response_id: str
    success: bool
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

class UniversalLLMClient:
    """PRODUCTION LLM client - makes actual API calls only"""
    
    def __init__(self):
        self.session = None
        self.call_count = 0
        self.total_tokens = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_llm(self, provider: str, model: str, api_key: str, 
                      prompt: str, endpoint: str = None, 
                      max_tokens: int = 1000, temperature: float = 0.1) -> LLMResponse:
        """Make REAL API call to LLM - NO SIMULATION"""
        
        if not api_key or len(api_key) < 10:
            return LLMResponse(
                content="", tokens_used=0, latency=0.0, model=model,
                provider=provider, response_id="error", success=False,
                error="Invalid API key provided"
            )
        
        try:
            start_time = time.time()
            self.call_count += 1
            
            if provider.lower() == "groq":
                response = await self._call_groq(api_key, model, prompt, max_tokens, temperature)
            elif provider.lower() == "openai":
                response = await self._call_openai(api_key, model, prompt, max_tokens, temperature)
            elif provider.lower() == "anthropic":
                response = await self._call_anthropic(api_key, model, prompt, max_tokens, temperature)
            elif provider.lower() == "custom" and endpoint:
                response = await self._call_custom(endpoint, api_key, model, prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            end_time = time.time()
            latency = end_time - start_time
            
            if response["success"]:
                self.total_tokens += response["tokens_used"]
                
                logger.info(f"✅ REAL API CALL SUCCESS - {provider}:{model} - {response['tokens_used']} tokens - {latency:.3f}s")
                
                return LLMResponse(
                    content=response["content"],
                    tokens_used=response["tokens_used"],
                    latency=latency,
                    model=model,
                    provider=provider,
                    response_id=response.get("response_id", f"call_{self.call_count}"),
                    success=True,
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                logger.error(f"❌ REAL API CALL FAILED - {provider}:{model} - {response.get('error')}")
                return LLMResponse(
                    content="", tokens_used=0, latency=latency, model=model,
                    provider=provider, response_id="error", success=False,
                    error=response.get("error", "Unknown API error")
                )
                
        except Exception as e:
            logger.error(f"❌ API CALL EXCEPTION - {provider}:{model} - {str(e)}")
            return LLMResponse(
                content="", tokens_used=0, latency=0.0, model=model,
                provider=provider, response_id="error", success=False,
                error=f"Exception: {str(e)}"
            )
    
    async def _call_groq(self, api_key: str, model: str, prompt: str, 
                        max_tokens: int, temperature: float) -> Dict[str, Any]:
        """REAL Groq API call"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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
                    "response_id": data.get("id", "unknown")
                }
            else:
                error_data = await response.text()
                return {
                    "success": False,
                    "error": f"Groq API Error {response.status}: {error_data}"
                }
    
    async def _call_openai(self, api_key: str, model: str, prompt: str,
                          max_tokens: int, temperature: float) -> Dict[str, Any]:
        """REAL OpenAI API call"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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
                    "response_id": data.get("id", "unknown")
                }
            else:
                error_data = await response.text()
                return {
                    "success": False,
                    "error": f"OpenAI API Error {response.status}: {error_data}"
                }
    
    async def _call_anthropic(self, api_key: str, model: str, prompt: str,
                             max_tokens: int, temperature: float) -> Dict[str, Any]:
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
                    "response_id": data.get("id", "unknown")
                }
            else:
                error_data = await response.text()
                return {
                    "success": False,
                    "error": f"Anthropic API Error {response.status}: {error_data}"
                }
    
    async def _call_custom(self, endpoint: str, api_key: str, model: str, prompt: str,
                          max_tokens: int, temperature: float) -> Dict[str, Any]:
        """REAL Custom API call"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(endpoint, headers=headers, json=payload) as response:
            
            if response.status == 200:
                data = await response.json()
                content = ""
                tokens_used = 0
                
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    tokens_used = data.get("usage", {}).get("total_tokens", 0)
                elif "content" in data:
                    content = data["content"]
                    tokens_used = len(content.split()) * 1.3
                
                return {
                    "success": True,
                    "content": content,
                    "tokens_used": int(tokens_used),
                    "response_id": data.get("id", "unknown")
                }
            else:
                error_data = await response.text()
                return {
                    "success": False,
                    "error": f"Custom API Error {response.status}: {error_data}"
                }
