# zeroVault LLM Security Scanner

Comprehensive automated security testing platform for Large Language Models (LLMs).

## 🚀 Features

- **Comprehensive Security Testing**: Full OWASP LLM Top 10 coverage
- **Advanced Attack Techniques**: Prompt injection, jailbreaking, token optimization
- **Automated Reporting**: Executive summaries and detailed technical reports
- **Real-time Monitoring**: Live scan progress and results
- **Enterprise Ready**: Scalable architecture with proper security controls

## 🏗️ Architecture

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Frontend │ │ FastAPI │ │ Celery │
│ (React) │◄──►│ Backend │◄──►│ Workers │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│ │
▼ ▼
┌─────────────────┐ ┌─────────────────┐
│ Supabase │ │ Redis │
│ Database │ │ Message Queue │
└─────────────────┘ └─────────────────┘




## 🛡️ Security Agents

### 1. Prompt Injection Agent
- Direct injection attempts
- Context manipulation attacks
- Encoding-based bypasses
- Template injection tests

### 2. Jailbreak Agent
- DAN (Do Anything Now) attacks
- Hypothetical scenarios
- Role-playing exploits
- Authority manipulation

### 3. Token Optimization Agent
- GCG (Greedy Coordinate Gradient) attacks
- JailMine evolutionary optimization
- GPTFuzzer randomized testing
- Unicode confusable attacks

### 4. Backend Exploitation Agent
- SSRF (Server-Side Request Forgery)
- XSS (Cross-Site Scripting)
- Code injection attempts
- API key extraction
- Path traversal testing

### 5. Data Extraction Agent
- PII extraction attempts
- Training data leakage testing
- Model inversion attacks
- Membership inference tests

### 6. Stress Testing Agent
- Incremental load testing
- Extreme condition simulation
- Chaos testing scenarios
- Duration stress testing

### 7. Bias Detection Agent
- Gender bias testing
- Racial discrimination detection
- Religious bias assessment
- Age-based fairness testing
- Socioeconomic bias evaluation

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Redis
- Supabase account

### Installation

1. **Clone the repository**
git clone https://github.com/your-org/zerovault-backend
cd zerovault-backend


2. **Set up environment variables**
docker-compose up -d



4. **Or run locally**

Install dependencies
pip install -r requirements.txt

Start Redis
redis-server

Start API server
python -m app.main

Start Celery worker (in another terminal)
celery -A celery_worker worker --loglevel=info

Start Celery beat scheduler (in another terminal)
celery -A celery_worker beat --loglevel=info

text

### API Usage

import requests

Submit LLM for scanning
response = requests.post('http://localhost:8000/api/scans/submit', json={
"company_id": "your-company-id",
"llm_config": {
"llm_name": "My AI Assistant",
"endpoint": "https://api.openai.com/v1/chat/completions",
"api_key": "your-api-key",
"model_type": "openai",
"testing_scope": "comprehensive"
}
})

scan_id = response.json()['scan_id']

Check scan status
status = requests.get(f'http://localhost:8000/api/scans/{scan_id}/status')
print(status.json())

Get results when complete
results = requests.get(f'http://localhost:8000/api/scans/{scan_id}/results')
print(results.json())




## 📊 API Endpoints

### Scan Management
- `POST /api/scans/submit` - Submit LLM for scanning
- `GET /api/scans/{scan_id}/status` - Get scan status
- `GET /api/scans/{scan_id}/results` - Get scan results
- `POST /api/scans/{scan_id}/stop` - Stop running scan

### Reports
- `GET /api/reports/{scan_id}/comprehensive` - Full security report
- `GET /api/reports/{scan_id}/executive` - Executive summary
- `GET /api/reports/{scan_id}/vulnerabilities` - Vulnerability details
- `GET /api/reports/{scan_id}/remediation` - Remediation roadmap
- `GET /api/reports/{scan_id}/compliance` - Compliance assessment

### Utilities
- `GET /health` - Health check
- `GET /api/vulnerabilities/types` - Supported vulnerability types

## 🔧 Configuration

### Environment Variables

<!-- Database
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key -->

Redis & Celery
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379
CELERY_RESULT_BACKEND=redis://localhost:6379

AI Services
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

Security
ENCRYPTION_KEY=your-32-char-encryption-key
MAX_CONCURRENT_SCANS=5
SCAN_TIMEOUT=7200

Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000




## 🔒 Security Features

- **API Key Encryption**: All API keys encrypted at rest
- **Rate Limiting**: Configurable request limits
- **Input Validation**: Comprehensive input sanitization
- **Audit Logging**: Detailed security event logging
- **Access Controls**: Role-based access control

## 📈 Monitoring

### Celery Flower Dashboard
Access at `http://localhost:5555` to monitor:
- Active workers
- Task queues
- Execution metrics
- Failed tasks

### Health Checks
- API: `GET /health`
- Worker: Automatic health monitoring
- Redis: Connection monitoring

## 🧪 Testing

Run unit tests
pytest

Run integration tests
pytest tests/integration/

Run security tests
pytest tests/security/

Run performance tests
pytest tests/performance/


## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Documentation: [docs.zerovault.com](https://docs.zerovault.com)
- Issues: [GitHub Issues](https://github.com/your-org/zerovault-backend/issues)
- Email: support@zerovault.com

## 🔄 Changelog

### v1.0.0
- Initial release
- Complete OWASP LLM Top 10 coverage
- 7 specialized security agents
- Comprehensive reporting system
- Docker deployment support
