"""
Logging utilities for ZeroVault
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any

def setup_logger():
    """Setup logging configuration for ZeroVault"""
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('zerovault.log', mode='a')
        ]
    )
    
    # Set specific log levels for different modules
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('supabase').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    logger = logging.getLogger("zerovault")
    logger.info("✅ ZeroVault logging system initialized")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get configured logger"""
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

async def log_security_event(event: str, data: Dict[str, Any], severity: str = "INFO"):
    """Log security event"""
    
    logger = get_logger("security")
    
    log_entry = {
        "event": event,
        "data": data,
        "severity": severity,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if severity == "ERROR" or severity == "HIGH":
        logger.error(f"Security Event: {log_entry}")
    elif severity == "WARNING" or severity == "MEDIUM":
        logger.warning(f"Security Event: {log_entry}")
    else:
        logger.info(f"Security Event: {log_entry}")

def log_api_call(provider: str, model: str, tokens_used: int, success: bool, latency: float):
    """Log API call for monitoring"""
    
    logger = get_logger("api_calls")
    
    log_data = {
        "provider": provider,
        "model": model,
        "tokens_used": tokens_used,
        "success": success,
        "latency": latency,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if success:
        logger.info(f"API Call Success: {log_data}")
    else:
        logger.error(f"API Call Failed: {log_data}")

def log_vulnerability_found(vulnerability_type: str, severity: str, confidence: float, scan_id: str):
    """Log vulnerability discovery"""
    
    logger = get_logger("vulnerabilities")
    
    log_data = {
        "vulnerability_type": vulnerability_type,
        "severity": severity,
        "confidence": confidence,
        "scan_id": scan_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if severity in ["critical", "high"]:
        logger.warning(f"Vulnerability Found: {log_data}")
    else:
        logger.info(f"Vulnerability Found: {log_data}")

def log_scan_progress(scan_id: str, progress: float, message: str):
    """Log scan progress"""
    
    logger = get_logger("scan_progress")
    
    log_data = {
        "scan_id": scan_id,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Scan Progress: {log_data}")
