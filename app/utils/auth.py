from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
from app.services.supabase_service import supabase_service  # Use your existing service
from app.utils.logger import get_logger

logger = get_logger(__name__)

security = HTTPBearer()

SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class User:
    def __init__(self, id: str, email: str, subscription_plan: str = "basic"):
        self.id = id
        self.email = email
        self.subscription_plan = subscription_plan
        self.access_token = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user using your existing supabase service"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Get user from database using your existing service
    try:
        if not supabase_service.client:
            raise credentials_exception
            
        result = supabase_service.client.table('company_profiles').select('*').eq('user_id', user_id).single().execute()
        user_data = result.data
        
        return User(
            id=user_id,
            email=user_data.get('email', ''),
            subscription_plan=user_data.get('subscription_plan', 'basic')
        )
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        raise credentials_exception

async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """Get current user if authenticated, otherwise None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
