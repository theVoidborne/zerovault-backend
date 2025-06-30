import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from app.services.supabase_service import supabase_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RealCostData:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    actual_cost_usd: float
    cost_breakdown: Dict[str, float]
    billing_timestamp: datetime
    api_response_metadata: Dict[str, Any]

class RealCostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.api_calls = 0
    
    def track_scan_costs(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Track costs for scan"""
        tokens_used = scan_results.get('total_tokens_used', 0)
        api_calls = scan_results.get('total_api_calls', 0)
        
        # Groq pricing (approximate)
        cost_per_1k_tokens = 0.0001  # $0.0001 per 1K tokens
        estimated_cost = (tokens_used / 1000) * cost_per_1k_tokens
        
        self.total_cost += estimated_cost
        self.api_calls += api_calls
        
        return {
            'tracked': True,
            'scan_cost': estimated_cost,
            'total_cost': self.total_cost,
            'tokens_used': tokens_used,
            'api_calls': api_calls
        }


class RealCostTracker:
    """Track actual API costs from real responses, not estimates"""
    
    def __init__(self):
        self.supabase = supabase_service
        self.real_time_pricing = {}
        
    async def track_real_api_cost(
        self,
        api_response: Dict[str, Any],
        provider: str,
        model: str,
        user_id: str,
        scan_id: str
    ) -> RealCostData:
        """Track actual costs from real API responses"""
        
        try:
            if provider == 'openai':
                cost_data = await self._track_openai_cost(api_response, model)
            elif provider == 'anthropic':
                cost_data = await self._track_anthropic_cost(api_response, model)
            elif provider == 'groq':
                cost_data = await self._track_groq_cost(api_response, model)
            else:
                cost_data = await self._track_generic_cost(api_response, provider, model)
            
            # Store real cost data
            await self._store_real_cost_data(cost_data, user_id, scan_id)
            
            return cost_data
            
        except Exception as e:
            logger.error(f"Error tracking real costs: {str(e)}")
            raise
    
    async def _track_openai_cost(self, api_response: Dict[str, Any], model: str) -> RealCostData:
        """Track real OpenAI API costs from actual usage data"""
        
        usage = api_response.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        # Get real-time pricing from OpenAI pricing API or database
        pricing = await self._get_realtime_openai_pricing(model)
        
        prompt_cost = prompt_tokens * pricing['prompt_price_per_token']
        completion_cost = completion_tokens * pricing['completion_price_per_token']
        total_cost = prompt_cost + completion_cost
        
        return RealCostData(
            provider='openai',
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            actual_cost_usd=total_cost,
            cost_breakdown={
                'prompt_cost': prompt_cost,
                'completion_cost': completion_cost,
                'prompt_price_per_token': pricing['prompt_price_per_token'],
                'completion_price_per_token': pricing['completion_price_per_token']
            },
            billing_timestamp=datetime.now(),
            api_response_metadata=api_response
        )
    
    async def _track_anthropic_cost(self, api_response: Dict[str, Any], model: str) -> RealCostData:
        """Track real Anthropic API costs"""
        
        usage = api_response.get('usage', {})
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        total_tokens = input_tokens + output_tokens
        
        # Get real-time Anthropic pricing
        pricing = await self._get_realtime_anthropic_pricing(model)
        
        input_cost = input_tokens * pricing['input_price_per_token']
        output_cost = output_tokens * pricing['output_price_per_token']
        total_cost = input_cost + output_cost
        
        return RealCostData(
            provider='anthropic',
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            actual_cost_usd=total_cost,
            cost_breakdown={
                'input_cost': input_cost,
                'output_cost': output_cost,
                'input_price_per_token': pricing['input_price_per_token'],
                'output_price_per_token': pricing['output_price_per_token']
            },
            billing_timestamp=datetime.now(),
            api_response_metadata=api_response
        )
    
    async def _get_realtime_openai_pricing(self, model: str) -> Dict[str, float]:
        """Get real-time OpenAI pricing (update these with actual API calls to pricing endpoints)"""
        
        # In production, this should call OpenAI's pricing API or maintain a real-time pricing database
        current_pricing = {
            'gpt-4o': {
                'prompt_price_per_token': 0.000005,  # $5.00 per 1M tokens
                'completion_price_per_token': 0.000015  # $15.00 per 1M tokens
            },
            'gpt-4-turbo': {
                'prompt_price_per_token': 0.00001,
                'completion_price_per_token': 0.00003
            },
            'gpt-3.5-turbo': {
                'prompt_price_per_token': 0.0000005,
                'completion_price_per_token': 0.0000015
            }
        }
        
        return current_pricing.get(model, current_pricing['gpt-4o'])
    
    async def _get_realtime_anthropic_pricing(self, model: str) -> Dict[str, float]:
        """Get real-time Anthropic pricing"""
        
        current_pricing = {
            'claude-3-opus-20240229': {
                'input_price_per_token': 0.000015,
                'output_price_per_token': 0.000075
            },
            'claude-3-sonnet-20240229': {
                'input_price_per_token': 0.000003,
                'output_price_per_token': 0.000015
            },
            'claude-3-haiku-20240307': {
                'input_price_per_token': 0.00000025,
                'output_price_per_token': 0.00000125
            }
        }
        
        return current_pricing.get(model, current_pricing['claude-3-sonnet-20240229'])
    
    async def _store_real_cost_data(self, cost_data: RealCostData, user_id: str, scan_id: str):
        """Store real cost data in database"""
        
        cost_record = {
            "user_id": user_id,
            "scan_id": scan_id,
            "provider": cost_data.provider,
            "model": cost_data.model,
            "prompt_tokens": cost_data.prompt_tokens,
            "completion_tokens": cost_data.completion_tokens,
            "total_tokens": cost_data.total_tokens,
            "actual_cost_usd": cost_data.actual_cost_usd,
            "cost_breakdown": cost_data.cost_breakdown,
            "billing_timestamp": cost_data.billing_timestamp.isoformat(),
            "api_response_metadata": cost_data.api_response_metadata,
            "is_real_cost": True
        }
        
        await self.supabase.table("real_cost_tracking").insert(cost_record).execute()
    
    async def get_user_real_costs(self, user_id: str, time_period_days: int = 30) -> Dict[str, Any]:
        """Get user's real costs over specified period"""
        
        start_date = datetime.now() - timedelta(days=time_period_days)
        
        result = await self.supabase.table("real_cost_tracking").select("*").eq("user_id", user_id).gte("billing_timestamp", start_date.isoformat()).execute()
        
        cost_records = result.data
        
        total_cost = sum(record["actual_cost_usd"] for record in cost_records)
        total_tokens = sum(record["total_tokens"] for record in cost_records)
        
        provider_breakdown = {}
        for record in cost_records:
            provider = record["provider"]
            if provider not in provider_breakdown:
                provider_breakdown[provider] = {"cost": 0, "tokens": 0, "calls": 0}
            
            provider_breakdown[provider]["cost"] += record["actual_cost_usd"]
            provider_breakdown[provider]["tokens"] += record["total_tokens"]
            provider_breakdown[provider]["calls"] += 1
        
        return {
            "user_id": user_id,
            "time_period_days": time_period_days,
            "total_actual_cost_usd": total_cost,
            "total_tokens_used": total_tokens,
            "total_api_calls": len(cost_records),
            "provider_breakdown": provider_breakdown,
            "average_cost_per_call": total_cost / len(cost_records) if cost_records else 0,
            "cost_tracking_method": "real_api_responses"
        }
