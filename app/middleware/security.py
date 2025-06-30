"""
Enhanced security middleware for production
"""

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import logging

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Add security headers
        response = await call_next(request)
        
        # Security headers for production
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.clients = {
            ip: times for ip, times in self.clients.items()
            if any(t > current_time - self.period for t in times)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            self.clients[client_ip] = [
                t for t in self.clients[client_ip]
                if t > current_time - self.period
            ]
            
            if len(self.clients[client_ip]) >= self.calls:
                response = Response(
                    content="Rate limit exceeded",
                    status_code=429
                )
                return response
        else:
            self.clients[client_ip] = []
        
        self.clients[client_ip].append(current_time)
        
        return await call_next(request)
