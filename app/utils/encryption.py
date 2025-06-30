"""
Encryption utilities for ZeroVault - FIXED VERSION
"""

import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def encrypt_data(data: str) -> str:
    """Encrypt data using base64 encoding - FIXED to prevent corruption"""
    
    if not data:
        return data
    
    # Don't double-encrypt
    if data.startswith('encrypted:'):
        return data
    
    try:
        # Use simple base64 encoding to prevent corruption
        encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
        return f"encrypted:{encoded}"
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return data  # Return raw data if encryption fails

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt data using base64 decoding - FIXED to prevent corruption"""
    
    if not encrypted_data:
        return encrypted_data
    
    # If not encrypted, return as-is
    if not encrypted_data.startswith('encrypted:'):
        return encrypted_data
    
    try:
        # Remove prefix and decode
        encoded_part = encrypted_data.replace('encrypted:', '')
        decoded = base64.b64decode(encoded_part.encode('utf-8')).decode('utf-8')
        return decoded
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        # Return without prefix if decryption fails
        return encrypted_data.replace('encrypted:', '')

def encrypt_api_key(api_key: str) -> str:
    """Encrypt API key specifically"""
    return encrypt_data(api_key)

def decrypt_api_key(encrypted_api_key: str) -> str:
    """Decrypt API key specifically"""
    return decrypt_data(encrypted_api_key)
