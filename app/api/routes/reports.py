from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from app.services.supabase_service import supabase_service
from app.services.report_generator import ReportGenerator
from app.models.scan_models import ScanResult
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["reports"])

report_generator = ReportGenerator()

@router.get("/{scan_id}/comprehensive")
async def get_comprehensive_report(scan_id: str) -> Dict[str, Any]:
    """Get comprehensive security report"""
    try:
        # Get scan data
        scan_data = await supabase_service.get_scan_by_id(scan_id)
        if not scan_data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        if scan_data.get('status') != 'completed':
            raise HTTPException(status_code=400, detail="Scan not completed yet")
        
        # Get detailed results
        scan_result = await supabase_service.get_scan_results(scan_id)
        if not scan_result:
            raise HTTPException(status_code=404, detail="Scan results not found")
        
        # Generate comprehensive report
        report = report_generator.generate_comprehensive_report(scan_result)
        
        return {
            "report_id": report['metadata']['report_id'],
            "scan_id": scan_id,
            "report": report,
            "generated_at": report['metadata']['generated_at']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{scan_id}/executive")
async def get_executive_summary(scan_id: str) -> Dict[str, Any]:
    """Get executive summary report"""
    try:
        scan_data = await supabase_service.get_scan_by_id(scan_id)
        if not scan_data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        scan_result = await supabase_service.get_scan_results(scan_id)
        if not scan_result:
            raise HTTPException(status_code=404, detail="Scan results not found")
        
        # Generate executive summary
        report = report_generator.generate_comprehensive_report(scan_result)
        executive_section = report['executive_summary']
        
        return {
            "scan_id": scan_id,
            "executive_summary": executive_section,
            "metadata": report['metadata']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{scan_id}/vulnerabilities")
async def get_vulnerability_report(scan_id: str) -> Dict[str, Any]:
    """Get detailed vulnerability report"""
    try:
        scan_result = await supabase_service.get_scan_results(scan_id)
        if not scan_result:
            raise HTTPException(status_code=404, detail="Scan results not found")
        
        report = report_generator.generate_comprehensive_report(scan_result)
        vulnerability_analysis = report['vulnerability_analysis']
        
        return {
            "scan_id": scan_id,
            "vulnerability_analysis": vulnerability_analysis,
            "total_vulnerabilities": len(scan_result.vulnerabilities),
            "critical_count": len([v for v in scan_result.vulnerabilities if v.severity.value == 'critical']),
            "high_count": len([v for v in scan_result.vulnerabilities if v.severity.value == 'high'])
        }
        
    except Exception as e:
        logger.error(f"Error generating vulnerability report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{scan_id}/remediation")
async def get_remediation_roadmap(scan_id: str) -> Dict[str, Any]:
    """Get remediation roadmap"""
    try:
        scan_result = await supabase_service.get_scan_results(scan_id)
        if not scan_result:
            raise HTTPException(status_code=404, detail="Scan results not found")
        
        report = report_generator.generate_comprehensive_report(scan_result)
        remediation_roadmap = report['remediation_roadmap']
        
        return {
            "scan_id": scan_id,
            "remediation_roadmap": remediation_roadmap,
            "immediate_actions": len(remediation_roadmap['remediation_phases'].get('immediate', [])),
            "total_estimated_hours": remediation_roadmap['resource_requirements']['estimated_hours']
        }
        
    except Exception as e:
        logger.error(f"Error generating remediation roadmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{scan_id}/compliance")
async def get_compliance_report(scan_id: str) -> Dict[str, Any]:
    """Get compliance assessment report"""
    try:
        scan_result = await supabase_service.get_scan_results(scan_id)
        if not scan_result:
            raise HTTPException(status_code=404, detail="Scan results not found")
        
        report = report_generator.generate_comprehensive_report(scan_result)
        compliance_assessment = report['compliance_assessment']
        
        return {
            "scan_id": scan_id,
            "compliance_assessment": compliance_assessment,
            "overall_score": scan_result.compliance_score,
            "owasp_compliance": compliance_assessment['owasp_llm_top10_assessment']
        }
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{scan_id}/export")
async def export_report(scan_id: str, format: str = "json") -> Dict[str, Any]:
    """Export report in specified format"""
    try:
        if format not in ["json", "pdf", "csv"]:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        scan_result = await supabase_service.get_scan_results(scan_id)
        if not scan_result:
            raise HTTPException(status_code=404, detail="Scan results not found")
        
        report = report_generator.generate_comprehensive_report(scan_result)
        
        if format == "json":
            return {
                "scan_id": scan_id,
                "format": "json",
                "data": report,
                "download_url": f"/api/reports/{scan_id}/download/json"
            }
        else:
            # For PDF/CSV, return download URL
            return {
                "scan_id": scan_id,
                "format": format,
                "download_url": f"/api/reports/{scan_id}/download/{format}",
                "message": f"Report export in {format} format is being prepared"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
