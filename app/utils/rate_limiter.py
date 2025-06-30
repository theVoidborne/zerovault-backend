import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
        self.locks = defaultdict(asyncio.Lock)
    
    async def is_allowed(self, identifier: str, max_requests: int, 
                        time_window: int) -> bool:
        """Check if request is allowed based on rate limits"""
        async with self.locks[identifier]:
            now = time.time()
            request_times = self.requests[identifier]
            
            # Remove old requests outside the time window
            while request_times and request_times[0] <= now - time_window:
                request_times.popleft()
            
            # Check if under the limit
            if len(request_times) < max_requests:
                request_times.append(now)
                return True
            
            return False
    
    async def wait_for_slot(self, identifier: str, max_requests: int, 
                           time_window: int) -> None:
        """Wait until a slot is available"""
        while not await self.is_allowed(identifier, max_requests, time_window):
            await asyncio.sleep(0.1)
    
    def get_remaining_requests(self, identifier: str, max_requests: int, 
                              time_window: int) -> int:
        """Get remaining requests in current window"""
        now = time.time()
        request_times = self.requests[identifier]
        
        # Remove old requests
        while request_times and request_times[0] <= now - time_window:
            request_times.popleft()
        
        return max(0, max_requests - len(request_times))
    
    def reset_limits(self, identifier: str) -> None:
        """Reset rate limits for identifier"""
        if identifier in self.requests:
            self.requests[identifier].clear()

# Global rate limiter instance
rate_limiter = RateLimiter()
