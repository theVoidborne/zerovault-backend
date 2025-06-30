"""
Complete API routes initialization
"""

from fastapi import APIRouter
from . import scans, health, auth, reports

# Create main router
api_router = APIRouter()

# Include all route modules
api_router.include_router(scans.router, prefix="/scans", tags=["scans"])
api_router.include_router(health.router, prefix="/health", tags=["health"])

# Add auth routes if they exist
try:
    api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
except AttributeError:
    pass  # Auth routes optional

# Add reports routes if they exist
try:
    api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
except AttributeError:
    pass  # Reports routes optional

__all__ = ["api_router"]
