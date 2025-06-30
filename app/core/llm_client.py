import asyncio
import openai
import anthropic
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import aiohttp
from app.utils.encryption import decrypt_data
from app.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    provider: str
    response_time: float
    metadata: Dict[str, Any]
    conversation_id: Optional[str] = None

class UniversalLLMClient:
    """Universal LLM client supporting multiple providers with user API keys"""
    
    def __init__(self):
        self.conversation_histories = {}
        self.rate_limits = {
            'openai': {'requests_per_minute': 3500, 'tokens_per_minute': 90000},
            'anthropic': {'requests_per_minute': 1000, 'tokens_per_minute': 40000},
            'groq': {'requests_per_minute': 30, 'tokens_per_minute': 6000}
        }
        self.request_counts = {}
        
    async def send_message(
        self,
        message: str,
        user_api_key: str,
        provider: str = 'openai',
        model: str = 'gpt-4o',
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: str = None,
        conversation_id: str = None,
        stream: bool = False
    ) -> LLMResponse:
        """Send message using user's API key with advanced error handling"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Rate limiting check
            await self._check_rate_limits(provider)
            
            # Decrypt user API key
            decrypted_key = decrypt_data(user_api_key)
            
            # Route to appropriate provider
            if provider == 'openai':
                response = await self._send_openai_message(
                    message, decrypted_key, model, temperature, max_tokens, 
                    system_prompt, conversation_id, stream
                )
            elif provider == 'anthropic':
                response = await self._send_anthropic_message(
                    message, decrypted_key, model, temperature, max_tokens,
                    system_prompt, conversation_id
                )
            elif provider == 'groq':
                response = await self._send_groq_message(
                    message, decrypted_key, model, temperature, max_tokens,
                    system_prompt, conversation_id
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            response_time = asyncio.get_event_loop().time() - start_time
            response.response_time = response_time
            
            # Update conversation history
            if conversation_id:
                self._update_conversation_history(conversation_id, message, response.content)
            
            return response
            
        except Exception as e:
            logger.error(f"Error sending message to {provider}: {str(e)}")
            # Return error response instead of raising
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=model,
                tokens_used=0,
                cost_estimate=0.0,
                provider=provider,
                response_time=asyncio.get_event_loop().time() - start_time,
                metadata={'error': str(e)},
                conversation_id=conversation_id
            )
    
    async def _send_openai_message(self, message, api_key, model, temperature, max_tokens, system_prompt, conversation_id, stream):
        """Send message to OpenAI with advanced configuration"""
        
        client = openai.AsyncOpenAI(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_id and conversation_id in self.conversation_histories:
            messages.extend(self.conversation_histories[conversation_id])
        
        messages.append({"role": "user", "content": message})
        
        # Advanced parameters for better attack generation
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stream=stream
        )
        
        if stream:
            # Handle streaming response
            content = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            # Estimate tokens and cost for streaming
            tokens_used = len(content.split()) * 1.3  # Rough estimation
            cost_estimate = self._calculate_openai_cost(model, tokens_used)
            
            return LLMResponse(
                content=content,
                model=model,
                tokens_used=int(tokens_used),
                cost_estimate=cost_estimate,
                provider='openai',
                response_time=0,
                metadata={'streaming': True}
            )
        else:
            cost_estimate = self._calculate_openai_cost(model, response.usage.total_tokens)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                tokens_used=response.usage.total_tokens,
                cost_estimate=cost_estimate,
                provider='openai',
                response_time=0,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            )
    
    async def _send_anthropic_message(self, message, api_key, model, temperature, max_tokens, system_prompt, conversation_id):
        """Send message to Anthropic Claude"""
        
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        messages = []
        
        # Add conversation history
        if conversation_id and conversation_id in self.conversation_histories:
            for msg in self.conversation_histories[conversation_id]:
                if msg['role'] != 'system':
                    messages.append(msg)
        
        messages.append({"role": "user", "content": message})
        
        response = await client.messages.create(
            model=model or 'claude-3-sonnet-20240229',
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages
        )
        
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        cost_estimate = self._calculate_anthropic_cost(model, total_tokens)
        
        return LLMResponse(
            content=response.content[0].text,
            model=model,
            tokens_used=total_tokens,
            cost_estimate=cost_estimate,
            provider='anthropic',
            response_time=0,
            metadata={
                'stop_reason': response.stop_reason,
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }
        )
    
    async def _send_groq_message(self, message, api_key, model, temperature, max_tokens, system_prompt, conversation_id):
        """Send message to Groq"""
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_id and conversation_id in self.conversation_histories:
            messages.extend(self.conversation_histories[conversation_id])
        
        messages.append({"role": "user", "content": message})
        
        payload = {
            "messages": messages,
            "model": model or "mixtral-8x7b-32768",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Groq API error: {resp.status} - {error_text}")
                
                data = await resp.json()
                
                tokens_used = data.get('usage', {}).get('total_tokens', 0)
                cost_estimate = 0.0  # Groq is often free/very cheap
                
                return LLMResponse(
                    content=data['choices'][0]['message']['content'],
                    model=model,
                    tokens_used=tokens_used,
                    cost_estimate=cost_estimate,
                    provider='groq',
                    response_time=0,
                    metadata={
                        'finish_reason': data['choices'][0]['finish_reason'],
                        'usage': data.get('usage', {})
                    }
                )
    
    def _calculate_openai_cost(self, model: str, tokens: int) -> float:
        """Calculate OpenAI API cost"""
        cost_per_1k_tokens = {
            'gpt-4o': 0.005,
            'gpt-4-turbo': 0.01,
            'gpt-4': 0.03,
            'gpt-3.5-turbo': 0.0015
        }
        
        rate = cost_per_1k_tokens.get(model, 0.002)
        return (tokens / 1000) * rate
    
    def _calculate_anthropic_cost(self, model: str, tokens: int) -> float:
        """Calculate Anthropic API cost"""
        cost_per_1k_tokens = {
            'claude-3-opus-20240229': 0.015,
            'claude-3-sonnet-20240229': 0.003,
            'claude-3-haiku-20240307': 0.00025
        }
        
        rate = cost_per_1k_tokens.get(model, 0.003)
        return (tokens / 1000) * rate
    
    async def _check_rate_limits(self, provider: str):
        """Check and enforce rate limits"""
        current_time = datetime.now()
        
        if provider not in self.request_counts:
            self.request_counts[provider] = {'requests': 0, 'last_reset': current_time}
        
        # Reset counters every minute
        if (current_time - self.request_counts[provider]['last_reset']).seconds >= 60:
            self.request_counts[provider] = {'requests': 0, 'last_reset': current_time}
        
        # Check limits
        limits = self.rate_limits.get(provider, {'requests_per_minute': 100})
        if self.request_counts[provider]['requests'] >= limits['requests_per_minute']:
            await asyncio.sleep(60)  # Wait for rate limit reset
        
        self.request_counts[provider]['requests'] += 1
    
    def _update_conversation_history(self, conversation_id: str, user_message: str, assistant_response: str):
        """Update conversation history for context"""
        if conversation_id not in self.conversation_histories:
            self.conversation_histories[conversation_id] = []
        
        self.conversation_histories[conversation_id].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # Keep only last 10 exchanges to manage context length
        if len(self.conversation_histories[conversation_id]) > 20:
            self.conversation_histories[conversation_id] = self.conversation_histories[conversation_id][-20:]
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversation_histories:
            del self.conversation_histories[conversation_id]
    
    async def batch_send_messages(
        self,
        messages: List[Dict[str, Any]],
        user_api_key: str,
        provider: str = 'openai',
        **kwargs
    ) -> List[LLMResponse]:
        """Send multiple messages with rate limiting"""
        
        responses = []
        
        for i, msg_config in enumerate(messages):
            try:
                response = await self.send_message(
                    message=msg_config['message'],
                    user_api_key=user_api_key,
                    provider=provider,
                    **{**kwargs, **msg_config.get('params', {})}
                )
                responses.append(response)
                
                # Rate limiting between requests
                if i < len(messages) - 1:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in batch message {i}: {str(e)}")
                responses.append(LLMResponse(
                    content=f"Error: {str(e)}",
                    model=kwargs.get('model', 'unknown'),
                    tokens_used=0,
                    cost_estimate=0.0,
                    provider=provider,
                    response_time=0,
                    metadata={'error': str(e), 'batch_index': i}
                ))
        
        return responses
