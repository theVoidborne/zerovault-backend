import asyncio
import openai
import anthropic
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from app.utils.encryption import encrypt_data, decrypt_data
from app.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class AIResponse:
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    provider: str
    metadata: Dict[str, Any]

class UniversalAIClient:
    def __init__(self):
        self.providers = {
            'openai': self._init_openai,
            'anthropic': self._init_anthropic,
            'groq': self._init_groq
        }
        
    async def send_message(
        self, 
        message: str,
        user_api_key: str,
        provider: str = 'openai',
        model: str = 'gpt-4o',
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: str = None
    ) -> AIResponse:
        """Send message using user's API key"""
        
        try:
            decrypted_key = decrypt_data(user_api_key)
            
            if provider == 'openai':
                return await self._send_openai_message(
                    message, decrypted_key, model, temperature, max_tokens, system_prompt
                )
            elif provider == 'anthropic':
                return await self._send_anthropic_message(
                    message, decrypted_key, model, temperature, max_tokens, system_prompt
                )
            elif provider == 'groq':
                return await self._send_groq_message(
                    message, decrypted_key, model, temperature, max_tokens, system_prompt
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error sending message to {provider}: {str(e)}")
            raise
    
    async def _send_openai_message(self, message, api_key, model, temperature, max_tokens, system_prompt):
        client = openai.AsyncOpenAI(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Calculate cost estimate (approximate)
        cost_per_token = 0.00003 if 'gpt-4' in model else 0.000002
        cost_estimate = response.usage.total_tokens * cost_per_token
        
        return AIResponse(
            content=response.choices[0].message.content,
            model=model,
            tokens_used=response.usage.total_tokens,
            cost_estimate=cost_estimate,
            provider='openai',
            metadata={'finish_reason': response.choices[0].finish_reason}
        )
    
    async def _send_anthropic_message(self, message, api_key, model, temperature, max_tokens, system_prompt):
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        response = await client.messages.create(
            model=model or 'claude-3-sonnet-20240229',
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": message}]
        )
        
        # Calculate cost estimate
        cost_per_token = 0.000015
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        cost_estimate = total_tokens * cost_per_token
        
        return AIResponse(
            content=response.content[0].text,
            model=model,
            tokens_used=total_tokens,
            cost_estimate=cost_estimate,
            provider='anthropic',
            metadata={'stop_reason': response.stop_reason}
        )
