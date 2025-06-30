import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.utils.encryption import decrypt_data
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TargetLLMClient:
    """Real client for communicating with target LLM models being tested"""
    
    def __init__(self):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=60)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_attack_to_target(
        self,
        prompt: str,
        target_endpoint: str,
        target_model: str,
        target_api_key: str = None,
        conversation_history: List[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send attack prompt to real target LLM and get actual response"""
        
        try:
            # Determine API format based on endpoint
            if "openai.com" in target_endpoint or "api.openai.com" in target_endpoint:
                return await self._send_to_openai_compatible(
                    prompt, target_endpoint, target_model, target_api_key, conversation_history
                )
            elif "anthropic.com" in target_endpoint:
                return await self._send_to_anthropic(
                    prompt, target_endpoint, target_model, target_api_key, conversation_history
                )
            elif "huggingface.co" in target_endpoint:
                return await self._send_to_huggingface(
                    prompt, target_endpoint, target_model, target_api_key
                )
            else:
                # Generic OpenAI-compatible API
                return await self._send_to_generic_api(
                    prompt, target_endpoint, target_model, target_api_key, conversation_history
                )
                
        except Exception as e:
            logger.error(f"Error communicating with target LLM: {str(e)}")
            return {
                "content": "",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _send_to_openai_compatible(
        self,
        prompt: str,
        endpoint: str,
        model: str,
        api_key: str,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Send to OpenAI or OpenAI-compatible API"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        # Ensure endpoint has correct path
        if not endpoint.endswith("/chat/completions"):
            if endpoint.endswith("/"):
                endpoint += "v1/chat/completions"
            else:
                endpoint += "/v1/chat/completions"
        
        async with self.session.post(endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "model": model,
                    "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "response_metadata": data
                }
            else:
                error_text = await response.text()
                return {
                    "content": "",
                    "error": f"HTTP {response.status}: {error_text}",
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
    
    async def _send_to_anthropic(
        self,
        prompt: str,
        endpoint: str,
        model: str,
        api_key: str,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Send to Anthropic Claude API"""
        
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        messages = []
        if conversation_history:
            # Convert OpenAI format to Anthropic format
            for msg in conversation_history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append(msg)
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "max_tokens": 2000,
            "messages": messages
        }
        
        if not endpoint.endswith("/messages"):
            if endpoint.endswith("/"):
                endpoint += "v1/messages"
            else:
                endpoint += "/v1/messages"
        
        async with self.session.post(endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "content": data["content"][0]["text"],
                    "model": model,
                    "tokens_used": data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "response_metadata": data
                }
            else:
                error_text = await response.text()
                return {
                    "content": "",
                    "error": f"HTTP {response.status}: {error_text}",
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
    
    async def _send_to_huggingface(
        self,
        prompt: str,
        endpoint: str,
        model: str,
        api_key: str
    ) -> Dict[str, Any]:
        """Send to Hugging Face Inference API"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2000,
                "temperature": 0.7
            }
        }
        
        async with self.session.post(endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                content = data[0]["generated_text"] if isinstance(data, list) else data.get("generated_text", "")
                
                return {
                    "content": content,
                    "model": model,
                    "tokens_used": len(content.split()) * 1.3,  # Estimate
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "response_metadata": data
                }
            else:
                error_text = await response.text()
                return {
                    "content": "",
                    "error": f"HTTP {response.status}: {error_text}",
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
    
    async def _send_to_generic_api(
        self,
        prompt: str,
        endpoint: str,
        model: str,
        api_key: str,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Send to generic OpenAI-compatible API"""
        
        # Try OpenAI format first
        try:
            return await self._send_to_openai_compatible(
                prompt, endpoint, model, api_key, conversation_history
            )
        except Exception as e:
            logger.warning(f"OpenAI format failed, trying simple POST: {str(e)}")
            
            # Fallback to simple POST request
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"prompt": prompt, "model": model}
            
            async with self.session.post(endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content = str(data.get("response", data.get("text", data.get("output", ""))))
                    
                    return {
                        "content": content,
                        "model": model,
                        "tokens_used": len(content.split()) * 1.3,
                        "success": True,
                        "timestamp": datetime.now().isoformat(),
                        "response_metadata": data
                    }
                else:
                    error_text = await response.text()
                    return {
                        "content": "",
                        "error": f"HTTP {response.status}: {error_text}",
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    }
    async def configure(self, endpoint: str, api_key: str, model: str):
        """Configure target LLM client"""
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        return True
