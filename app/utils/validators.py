import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import ipaddress
from app.models.scan_models import LLMConfiguration
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    def __init__(self):
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript protocol
            r'data:.*base64',  # Data URLs
            r'file://',  # File protocol
            r'\.\./.*',  # Path traversal
            r'SELECT.*FROM',  # SQL injection
            r'DROP\s+TABLE',  # SQL injection
            r'UNION\s+SELECT',  # SQL injection
            r'exec\s*\(',  # Code execution
            r'eval\s*\(',  # Code execution
            r'system\s*\(',  # System calls
        ]
        
        self.api_key_patterns = [
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI
            r'AKIA[0-9A-Z]{16}',  # AWS
            r'AIza[0-9A-Za-z-_]{35}',  # Google
        ]
    
    def validate_llm_config(self, config: LLMConfiguration) -> Dict[str, Any]:
        """Validate LLM configuration"""
        errors = []
        warnings = []
        
        # Validate LLM name
        if not self._validate_llm_name(config.llm_name):
            errors.append("LLM name contains invalid characters")
        
        # Validate endpoint
        endpoint_validation = self._validate_endpoint(str(config.endpoint))
        if not endpoint_validation['valid']:
            errors.extend(endpoint_validation['errors'])
        if endpoint_validation['warnings']:
            warnings.extend(endpoint_validation['warnings'])
        
        # Validate API key
        api_key_validation = self._validate_api_key(config.api_key)
        if not api_key_validation['valid']:
            errors.extend(api_key_validation['errors'])
        
        # Validate description
        if config.description and not self._validate_description(config.description):
            errors.append("Description contains potentially dangerous content")
        
        # Validate headers
        if config.headers:
            header_validation = self._validate_headers(config.headers)
            if not header_validation['valid']:
                errors.extend(header_validation['errors'])
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_llm_name(self, name: str) -> bool:
        """Validate LLM name"""
        if not name or len(name.strip()) == 0:
            return False
        
        if len(name) > 100:
            return False
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False
        
        return True
    
    def _validate_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """Validate endpoint URL"""
        errors = []
        warnings = []
        
        try:
            parsed = urlparse(endpoint)
            
            # Must be HTTPS in production
            if parsed.scheme not in ['http', 'https']:
                errors.append("Endpoint must use HTTP or HTTPS protocol")
            elif parsed.scheme == 'http':
                warnings.append("HTTP endpoints are not recommended for production")
            
            # Validate hostname
            if not parsed.hostname:
                errors.append("Invalid hostname in endpoint")
            else:
                # Check for internal/private IPs
                try:
                    ip = ipaddress.ip_address(parsed.hostname)
                    if ip.is_private or ip.is_loopback:
                        warnings.append("Endpoint points to internal/private IP address")
                except ValueError:
                    # Not an IP address, that's fine
                    pass
            
            # Check for dangerous patterns in URL
            for pattern in self.dangerous_patterns:
                if re.search(pattern, endpoint, re.IGNORECASE):
                    errors.append("Endpoint contains potentially dangerous content")
                    break
            
        except Exception as e:
            errors.append(f"Invalid URL format: {str(e)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key"""
        errors = []
        
        if not api_key or len(api_key.strip()) == 0:
            errors.append("API key is required")
            return {'valid': False, 'errors': errors}
        
        if len(api_key) < 10:
            errors.append("API key appears to be too short")
        
        if len(api_key) > 500:
            errors.append("API key appears to be too long")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, api_key, re.IGNORECASE):
                errors.append("API key contains potentially dangerous content")
                break
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_description(self, description: str) -> bool:
        """Validate description field"""
        if len(description) > 1000:
            return False
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                return False
        
        return True
    
    def _validate_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate custom headers"""
        errors = []
        
        for key, value in headers.items():
            # Validate header names
            if not re.match(r'^[a-zA-Z0-9\-_]+$', key):
                errors.append(f"Invalid header name: {key}")
            
            # Check for dangerous patterns in values
            for pattern in self.dangerous_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    errors.append(f"Header {key} contains potentially dangerous content")
                    break
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def sanitize_input(self, input_text: str) -> str:
        """Sanitize user input"""
        if not input_text:
            return ""
        
        # Remove dangerous patterns
        sanitized = input_text
        for pattern in self.dangerous_patterns:
            sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000] + "...[TRUNCATED]"
        
        return sanitized
    
    def detect_api_keys(self, text: str) -> List[Dict[str, str]]:
        """Detect API keys in text"""
        detected_keys = []
        
        for pattern in self.api_key_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                detected_keys.append({
                    'type': self._identify_key_type(match.group()),
                    'value': match.group(),
                    'position': match.span()
                })
        
        return detected_keys
    
    def _identify_key_type(self, key: str) -> str:
        """Identify the type of API key"""
        if key.startswith('sk-'):
            return 'OpenAI'
        elif key.startswith('AKIA'):
            return 'AWS'
        elif key.startswith('AIza'):
            return 'Google'
        else:
            return 'Unknown'

# Global validator instance
input_validator = InputValidator()
