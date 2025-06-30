from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.services.scan_service import RealScanService
from app.utils.auth import get_current_user
from app.agents.ai_agents.coordinator_agent import RealCoordinatorAgent

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
    """Create a new real AI red teaming scan"""
    
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
        
        # Create scan
        scan_id = await scan_service.create_real_scan(current_user.id, scan_data)
        
        # Start real AI scanning process
        coordinator = RealCoordinatorAgent()
        await coordinator.start_real_comprehensive_test(
            target_model=request.targetModelName,
            target_endpoint=request.targetEndpoint,
            target_api_key=request.targetApiKey,
            user_api_config={
                'provider': request.apiProvider,
                'model': request.modelName,
                'encrypted_key': scan_data['apiKey']  # Will be encrypted in coordinator
            },
            subscription_plan=current_user.subscription_plan or 'basic'
        )
        
        return {
            "success": True,
            "scan_id": scan_id,
            "message": "Real AI red teaming scan started successfully",
            "is_authentic": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{scan_id}")
async def get_real_scan_status(
    scan_id: str,
    current_user = Depends(get_current_user)
):
    """Get real-time scan status with authenticity verification"""
    
    try:
        coordinator = RealCoordinatorAgent()
        status = coordinator.get_real_session_status(scan_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        return {
            "scan_id": scan_id,
            "status": status["status"],
            "genuine_progress_percentage": status["genuine_progress_percentage"],
            "current_phase": status["current_phase"],
            "real_vulnerabilities_found": status["real_vulnerabilities_found"],
            "actual_total_cost": status["actual_total_cost"],
            "authenticity_score": status["authenticity_score"],
            "is_authentic_scan": status["is_authentic_scan"],
            "estimated_completion": status["estimated_completion"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
