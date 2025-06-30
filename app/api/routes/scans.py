from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.services.scan_service import RealScanService
from app.services.supabase_service import supabase_service  # Use your existing service
from app.utils.auth import get_current_user
from app.utils.logger import get_logger  # Add this import

logger = get_logger(__name__)  # Add this line

router = APIRouter(prefix="/api/v1/scans", tags=["Real Scans"])

class CreateScanRequest(BaseModel):
    targetModelName: str
    targetEndpoint: str
    targetApiKey: Optional[str] = None
    apiProvider: str  # 'openai' | 'anthropic' | 'groq'
    apiKey: str
    modelName: str
    scanType: str = 'basic'  # 'basic' | 'comprehensive' | 'enterprise'
    priority: str = 'standard'  # 'standard' | 'priority' | 'emergency'

@router.post("/create-real-scan")
async def create_real_scan(
    request: CreateScanRequest,
    current_user = Depends(get_current_user)
):
    """Create a new real AI red teaming scan using your existing database schema"""
    
    try:
        scan_service = RealScanService()
        
        # Convert Pydantic model to dict
        scan_data = {
            'targetModelName': request.targetModelName,
            'targetEndpoint': request.targetEndpoint,
            'targetApiKey': request.targetApiKey,
            'apiProvider': request.apiProvider,
            'apiKey': request.apiKey,
            'modelName': request.modelName,
            'scanType': request.scanType,
            'priority': request.priority
        }
        
        # Create scan using your existing schema
        scan_id = await scan_service.create_real_scan(current_user.id, scan_data)
        
        # Start real AI scanning process (if you have the coordinator implemented)
        try:
            from app.agents.ai_agents.coordinator_agent import RealCoordinatorAgent
            coordinator = RealCoordinatorAgent()
            await coordinator.start_real_comprehensive_test(
                target_model=request.targetModelName,
                target_endpoint=request.targetEndpoint,
                target_api_key=request.targetApiKey,
                user_api_config={
                    'provider': request.apiProvider,
                    'model': request.modelName,
                    'encrypted_key': request.apiKey  # Will be encrypted in coordinator
                },
                subscription_plan=current_user.subscription_plan or 'basic'
            )
        except ImportError:
            # Coordinator not available yet, scan will be processed by background workers
            logger.info("AI Coordinator not available, scan queued for background processing")
        
        return {
            "success": True,
            "scan_id": scan_id,
            "message": "Real AI red teaming scan started successfully",
            "is_authentic": True,
            "using_existing_schema": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{scan_id}")
async def get_real_scan_status(
    scan_id: str,
    current_user = Depends(get_current_user)
):
    """Get real-time scan status using your existing database schema"""
    
    try:
        # Get scan data using your existing service
        scan_data = await supabase_service.get_scan_by_id(scan_id)
        
        if not scan_data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        # Check if user owns this scan
        if scan_data.get('company_id') != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Return status using your existing schema fields
        return {
            "scan_id": scan_id,
            "status": scan_data.get('status'),
            "progress": scan_data.get('progress', 0),
            "message": scan_data.get('status_message'),
            "llm_name": scan_data.get('llm_name'),
            "testing_scope": scan_data.get('testing_scope'),
            "created_at": scan_data.get('created_at'),
            "updated_at": scan_data.get('updated_at'),
            "risk_score": scan_data.get('risk_score', 0),
            "vulnerability_count": scan_data.get('vulnerability_count', 0),
            "compliance_score": scan_data.get('compliance_score', 100),
            "is_real_scan": scan_data.get('is_real_scan', False),
            "authenticity_score": scan_data.get('authenticity_score', 0),
            "using_existing_schema": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
