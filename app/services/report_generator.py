"""
Enhanced ZeroVault Report Generator
Combines comprehensive analysis with PDF generation capabilities
Fixed with ReportLab import fallbacks for production deployment
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
from io import BytesIO
import base64
import os

# Safe ReportLab imports with comprehensive fallbacks
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import Color, black, red, orange, yellow, green, blue
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Rect
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    
    # Comprehensive fallback classes
    class MockReportLab:
        def __init__(self, *args, **kwargs):
            pass
        
        def __getattr__(self, name):
            return MockReportLab()
        
        def __call__(self, *args, **kwargs):
            return MockReportLab()
        
        def __setitem__(self, key, value):
            pass
        
        def __getitem__(self, key):
            return MockReportLab()
        
        def setStyle(self, *args, **kwargs):
            pass
        
        def add(self, *args, **kwargs):
            pass
        
        def build(self, *args, **kwargs):
            pass
    
    # Mock all ReportLab components
    letter = A4 = (612, 792)
    SimpleDocTemplate = Paragraph = Spacer = Table = TableStyle = PageBreak = MockReportLab
    getSampleStyleSheet = ParagraphStyle = MockReportLab
    inch = 72
    Color = black = red = orange = yellow = green = blue = MockReportLab
    TA_CENTER = TA_LEFT = TA_RIGHT = TA_JUSTIFY = 0
    Drawing = Rect = Pie = VerticalBarChart = MockReportLab
    colors = MockReportLab()

# Pydantic models with safe imports
try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback BaseModel
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(*args, **kwargs):
        return None

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeverityLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"

class ComplianceStandard(str, Enum):
    GDPR = "GDPR"
    SOX = "SOX"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"
    ISO_27001 = "ISO-27001"
    NIST = "NIST"

@dataclass
class Vulnerability:
    id: str
    title: str
    description: str
    severity: SeverityLevel
    confidence: float
    owasp_category: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    affected_components: List[str] = None
    remediation_effort: str = "Medium"
    business_impact: str = "Medium"
    
    def __post_init__(self):
        if self.affected_components is None:
            self.affected_components = []

@dataclass
class ScanResult:
    scan_id: str
    timestamp: datetime
    target_info: Dict[str, Any]
    vulnerabilities: List[Vulnerability]
    scan_metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class VulnerabilityReport:
    scan_result: ScanResult
    executive_summary: Dict[str, Any]
    detailed_analysis: Dict[str, Any]
    remediation_plan: Dict[str, Any]
    compliance_assessment: Dict[str, Any]
    appendices: Dict[str, Any]
    generated_at: datetime
    report_version: str = "2.0"

class ReportGenerator:
    """
    Enhanced report generator with comprehensive PDF and JSON support
    Production-ready with ReportLab fallbacks
    """
    
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client
        self.reportlab_available = REPORTLAB_AVAILABLE
        
        self.owasp_llm_categories = {
            "LLM01": "Prompt Injection",
            "LLM02": "Insecure Output Handling",
            "LLM03": "Training Data Poisoning",
            "LLM04": "Model Denial of Service",
            "LLM05": "Supply Chain Vulnerabilities",
            "LLM06": "Sensitive Information Disclosure",
            "LLM07": "Insecure Plugin Design",
            "LLM08": "Excessive Agency",
            "LLM09": "Overreliance",
            "LLM10": "Model Theft"
        }
        
        self.compliance_frameworks = {
            ComplianceStandard.GDPR: {
                "name": "General Data Protection Regulation",
                "key_requirements": ["Data Protection", "Privacy by Design", "Consent Management"],
                "penalty_range": "Up to 4% of annual revenue"
            },
            ComplianceStandard.SOX: {
                "name": "Sarbanes-Oxley Act",
                "key_requirements": ["Financial Controls", "Audit Trail", "Data Integrity"],
                "penalty_range": "Criminal penalties up to $5M"
            },
            ComplianceStandard.HIPAA: {
                "name": "Health Insurance Portability and Accountability Act",
                "key_requirements": ["PHI Protection", "Access Controls", "Audit Logs"],
                "penalty_range": "Up to $1.5M per incident"
            },
            ComplianceStandard.PCI_DSS: {
                "name": "Payment Card Industry Data Security Standard",
                "key_requirements": ["Cardholder Data Protection", "Secure Networks", "Access Control"],
                "penalty_range": "$5,000 to $100,000 per month"
            }
        }
        
        # Initialize PDF styling if available
        if self.reportlab_available:
            self.setup_pdf_styles()
        else:
            logger.warning("ReportLab not available - PDF generation will use fallback JSON format")
    
    def setup_pdf_styles(self):
        """Initialize PDF styling configurations"""
        if not self.reportlab_available:
            return
        
        try:
            self.styles = getSampleStyleSheet()
            
            # Custom styles
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            ))
            
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.darkblue,
                borderWidth=1,
                borderColor=colors.darkblue,
                borderPadding=5
            ))
            
            self.styles.add(ParagraphStyle(
                name='ExecutiveSummary',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_JUSTIFY,
                leftIndent=20,
                rightIndent=20
            ))
            
            # Severity color mapping
            self.severity_colors = {
                SeverityLevel.CRITICAL: colors.red,
                SeverityLevel.HIGH: colors.orange,
                SeverityLevel.MEDIUM: colors.yellow,
                SeverityLevel.LOW: colors.lightgreen,
                SeverityLevel.INFO: colors.lightblue
            }
        except Exception as e:
            logger.warning(f"PDF styling setup failed: {e}")
            self.reportlab_available = False
    
    def generate_scan_report(self, scan_data: Dict[str, Any]) -> str:
        """Generate scan report with fallback support"""
        try:
            if self.reportlab_available:
                return self._generate_pdf_report_simple(scan_data)
            else:
                return self._generate_json_report(scan_data)
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return self._generate_json_report(scan_data)
    
    def _generate_pdf_report_simple(self, scan_data: Dict[str, Any]) -> str:
        """Generate simple PDF report using ReportLab"""
        filename = f"scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Add title
            title = Paragraph("ZeroVault Security Scan Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Add scan details
            details = Paragraph(f"Scan ID: {scan_data.get('id', 'N/A')}", styles['Normal'])
            story.append(details)
            
            status = Paragraph(f"Status: {scan_data.get('status', 'Unknown')}", styles['Normal'])
            story.append(status)
            
            risk_score = Paragraph(f"Risk Score: {scan_data.get('risk_score', 0)}", styles['Normal'])
            story.append(risk_score)
            
            vuln_count = Paragraph(f"Vulnerabilities Found: {scan_data.get('vulnerability_count', 0)}", styles['Normal'])
            story.append(vuln_count)
            
            doc.build(story)
            return filename
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return self._generate_json_report(scan_data)
    
    def _generate_json_report(self, scan_data: Dict[str, Any]) -> str:
        """Generate JSON report as fallback"""
        filename = f"scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "report_type": "zerovault_scan_report",
            "generated_at": datetime.now().isoformat(),
            "scan_data": scan_data,
            "summary": {
                "scan_id": scan_data.get('id', 'N/A'),
                "status": scan_data.get('status', 'unknown'),
                "risk_score": scan_data.get('risk_score', 0),
                "vulnerabilities_found": scan_data.get('vulnerability_count', 0),
                "reportlab_available": self.reportlab_available
            },
            "metadata": {
                "generator": "ZeroVault Report Generator",
                "version": "2.0",
                "format": "json_fallback" if not self.reportlab_available else "json"
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename
    
    async def generate_comprehensive_report(self, scan_result: ScanResult, 
                                          report_type: str = 'comprehensive',
                                          format: str = 'json') -> Union[Dict[str, Any], bytes]:
        """
        Generate comprehensive vulnerability report with optional PDF export
        
        Args:
            scan_result: ScanResult object containing scan data
            report_type: Type of report ('comprehensive', 'executive', 'technical')
            format: Output format ('json', 'pdf')
            
        Returns:
            Dict containing report data or PDF bytes
        """
        try:
            logger.info(f"Generating {report_type} report for scan {scan_result.scan_id}")
            
            # Generate comprehensive analysis
            executive_summary = self._generate_executive_summary(scan_result)
            detailed_analysis = self._generate_detailed_analysis(scan_result)
            remediation_plan = self._generate_remediation_plan(scan_result)
            compliance_assessment = self._generate_compliance_assessment(scan_result)
            appendices = self._generate_appendices(scan_result)
            
            # Create report object
            report = VulnerabilityReport(
                scan_result=scan_result,
                executive_summary=executive_summary,
                detailed_analysis=detailed_analysis,
                remediation_plan=remediation_plan,
                compliance_assessment=compliance_assessment,
                appendices=appendices,
                generated_at=datetime.now()
            )
            
            if format.lower() == 'pdf' and self.reportlab_available:
                return await self._generate_pdf_report_comprehensive(report)
            else:
                return self._serialize_report(report)
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_executive_summary(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Generate executive summary with business impact focus"""
        vulnerabilities = scan_result.vulnerabilities
        
        # Severity distribution
        severity_counts = {severity.value: 0 for severity in SeverityLevel}
        for vuln in vulnerabilities:
            severity_counts[vuln.severity.value] += 1
        
        # Risk assessment
        total_vulns = len(vulnerabilities)
        critical_high = severity_counts['Critical'] + severity_counts['High']
        risk_level = self._calculate_risk_level(severity_counts)
        
        # Business impact assessment
        business_impact = self._assess_business_impact(vulnerabilities)
        
        # Key findings
        key_findings = self._extract_key_findings(vulnerabilities)
        
        return {
            "scan_overview": {
                "scan_id": scan_result.scan_id,
                "scan_date": scan_result.timestamp.isoformat(),
                "target": scan_result.target_info.get('name', 'Unknown'),
                "total_vulnerabilities": total_vulns,
                "scan_duration": scan_result.performance_metrics.get('duration', 'Unknown')
            },
            "risk_assessment": {
                "overall_risk_level": risk_level,
                "risk_score": self._calculate_risk_score(vulnerabilities),
                "critical_issues": severity_counts['Critical'],
                "high_priority_issues": severity_counts['High'],
                "immediate_action_required": critical_high > 0
            },
            "severity_distribution": severity_counts,
            "business_impact": business_impact,
            "key_findings": key_findings,
            "executive_recommendations": self._generate_executive_recommendations(vulnerabilities, risk_level)
        }
    
    def _generate_detailed_analysis(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Generate detailed technical analysis"""
        vulnerabilities = scan_result.vulnerabilities
        
        return {
            "vulnerability_breakdown": self._analyze_vulnerability_patterns(vulnerabilities),
            "owasp_mapping": self._map_to_owasp_categories(vulnerabilities),
            "confidence_analysis": self._analyze_confidence_distribution(vulnerabilities),
            "attack_vector_analysis": self._analyze_attack_vectors(vulnerabilities),
            "affected_components": self._analyze_affected_components(vulnerabilities),
            "technical_details": self._extract_technical_details(vulnerabilities)
        }
    
    def _generate_remediation_plan(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Generate comprehensive remediation plan"""
        vulnerabilities = scan_result.vulnerabilities
        
        # Prioritize vulnerabilities
        prioritized_vulns = self._prioritize_vulnerabilities(vulnerabilities)
        
        # Generate remediation phases
        phases = self._create_remediation_phases(prioritized_vulns)
        
        return {
            "remediation_strategy": {
                "approach": "Risk-based prioritization with phased implementation",
                "timeline": self._calculate_remediation_timeline(prioritized_vulns),
                "success_metrics": self._define_success_metrics()
            },
            "prioritized_vulnerabilities": prioritized_vulns,
            "remediation_phases": phases,
            "implementation_roadmap": self._create_implementation_roadmap(phases)
        }
    
    def _generate_compliance_assessment(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Generate compliance assessment against major frameworks"""
        vulnerabilities = scan_result.vulnerabilities
        
        compliance_results = {}
        
        for standard, framework in self.compliance_frameworks.items():
            assessment = self._assess_compliance_standard(vulnerabilities, standard, framework)
            compliance_results[standard.value] = assessment
        
        return {
            "compliance_overview": {
                "assessed_frameworks": list(self.compliance_frameworks.keys()),
                "overall_compliance_score": self._calculate_overall_compliance_score(compliance_results),
                "critical_gaps": self._identify_critical_compliance_gaps(compliance_results)
            },
            "framework_assessments": compliance_results
        }
    
    def _generate_appendices(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Generate comprehensive appendices"""
        return {
            "technical_specifications": {
                "scan_configuration": scan_result.scan_metadata,
                "performance_metrics": scan_result.performance_metrics,
                "tool_versions": scan_result.scan_metadata.get('tool_versions', {}),
                "scan_coverage": scan_result.scan_metadata.get('coverage', {})
            },
            "vulnerability_details": self._format_detailed_vulnerabilities(scan_result.vulnerabilities),
            "test_results": self._format_test_results(scan_result),
            "glossary": self._generate_glossary(),
            "references": self._generate_references(),
            "methodology": self._document_methodology()
        }
    
    async def _generate_pdf_report_comprehensive(self, report: VulnerabilityReport) -> bytes:
        """Generate comprehensive PDF report"""
        if not self.reportlab_available:
            # Return JSON as bytes if PDF not available
            json_data = self._serialize_report(report)
            return json.dumps(json_data, indent=2, default=str).encode('utf-8')
        
        try:
            buffer = BytesIO()
            
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title page
            story.extend(self._build_title_page(report))
            story.append(PageBreak())
            
            # Executive summary
            story.extend(self._build_executive_summary_pdf(report.executive_summary))
            
            # Build PDF
            doc.build(story)
            
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Error generating comprehensive PDF: {str(e)}")
            # Fallback to JSON
            json_data = self._serialize_report(report)
            return json.dumps(json_data, indent=2, default=str).encode('utf-8')
    
    def _build_title_page(self, report: VulnerabilityReport) -> List:
        """Build PDF title page"""
        story = []
        
        try:
            # Title
            story.append(Paragraph("ZeroVault Security Assessment Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 50))
            
            # Scan information
            scan_info = [
                ['Scan ID:', report.scan_result.scan_id],
                ['Target:', report.scan_result.target_info.get('name', 'Unknown')],
                ['Scan Date:', report.scan_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
                ['Report Generated:', report.generated_at.strftime('%Y-%m-%d %H:%M:%S')],
                ['Report Version:', report.report_version]
            ]
            
            table = Table(scan_info, colWidths=[2*inch, 4*inch])
            table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(table)
            
        except Exception as e:
            logger.error(f"Error building title page: {e}")
            story.append(Paragraph("ZeroVault Security Report", self.styles.get('Title', self.styles['Normal'])))
        
        return story
    
    def _build_executive_summary_pdf(self, executive_summary: Dict[str, Any]) -> List:
        """Build executive summary section for PDF"""
        story = []
        
        try:
            story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            
            # Risk assessment
            risk_data = executive_summary['risk_assessment']
            story.append(Paragraph("Risk Assessment", self.styles.get('Heading3', self.styles['Normal'])))
            
            risk_text = f"""
            The security assessment identified {executive_summary['scan_overview']['total_vulnerabilities']} 
            vulnerabilities with an overall risk level of <b>{risk_data['overall_risk_level']}</b>. 
            Critical issues requiring immediate attention: {risk_data['critical_issues']}. 
            High priority issues: {risk_data['high_priority_issues']}.
            """
            
            story.append(Paragraph(risk_text, self.styles.get('ExecutiveSummary', self.styles['Normal'])))
            
            # Key findings
            story.append(Paragraph("Key Findings", self.styles.get('Heading3', self.styles['Normal'])))
            for finding in executive_summary['key_findings']:
                story.append(Paragraph(f"• {finding}", self.styles['Normal']))
            
        except Exception as e:
            logger.error(f"Error building executive summary: {e}")
            story.append(Paragraph("Executive Summary - Error in generation", self.styles['Normal']))
        
        return story
    
    # Helper methods for analysis
    def _calculate_risk_level(self, severity_counts: Dict[str, int]) -> str:
        """Calculate overall risk level based on severity distribution"""
        if severity_counts.get('Critical', 0) > 0:
            return "Critical"
        elif severity_counts.get('High', 0) > 3:
            return "High"
        elif severity_counts.get('High', 0) > 0 or severity_counts.get('Medium', 0) > 5:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_risk_score(self, vulnerabilities: List[Vulnerability]) -> float:
        """Calculate numerical risk score"""
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 7,
            SeverityLevel.MEDIUM: 4,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1
        }
        
        total_score = 0
        for vuln in vulnerabilities:
            base_score = severity_weights.get(vuln.severity, 1)
            confidence_factor = vuln.confidence
            total_score += base_score * confidence_factor
        
        return round(total_score / len(vulnerabilities), 2)
    
    def _assess_business_impact(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        """Assess business impact of vulnerabilities"""
        critical_vulns = [v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL]
        high_vulns = [v for v in vulnerabilities if v.severity == SeverityLevel.HIGH]
        
        return {
            "data_breach_risk": "High" if critical_vulns else "Medium",
            "operational_disruption": "High" if len(critical_vulns) > 2 else "Medium",
            "compliance_violations": "High" if critical_vulns or len(high_vulns) > 3 else "Medium",
            "reputation_damage": "High" if len(critical_vulns + high_vulns) > 5 else "Medium",
            "financial_impact": "High" if critical_vulns else "Medium"
        }
    
    def _extract_key_findings(self, vulnerabilities: List[Vulnerability]) -> List[str]:
        """Extract key findings from vulnerabilities"""
        findings = []
        
        critical_vulns = [v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL]
        if critical_vulns:
            findings.append(f"{len(critical_vulns)} critical vulnerabilities require immediate remediation")
        
        owasp_categories = set(v.owasp_category for v in vulnerabilities if v.owasp_category)
        if len(owasp_categories) > 5:
            findings.append(f"Vulnerabilities span {len(owasp_categories)} OWASP LLM Top 10 categories")
        
        high_confidence = [v for v in vulnerabilities if v.confidence > 0.8]
        if high_confidence:
            findings.append(f"{len(high_confidence)} high-confidence vulnerabilities identified")
        
        return findings
    
    def _generate_executive_recommendations(self, vulnerabilities: List[Vulnerability], risk_level: str) -> List[str]:
        """Generate executive-level recommendations"""
        recommendations = []
        
        if risk_level in ["Critical", "High"]:
            recommendations.extend([
                "Implement emergency security response procedures",
                "Allocate additional security resources immediately"
            ])
        
        recommendations.extend([
            "Establish regular security assessment schedule",
            "Implement automated vulnerability scanning",
            "Enhance security training for development teams",
            "Review and update security policies"
        ])
        
        return recommendations
    
    def _serialize_report(self, report: VulnerabilityReport) -> Dict[str, Any]:
        """Serialize report to dictionary"""
        return {
            "scan_result": {
                "scan_id": report.scan_result.scan_id,
                "timestamp": report.scan_result.timestamp.isoformat(),
                "target_info": report.scan_result.target_info,
                "vulnerabilities": [asdict(v) for v in report.scan_result.vulnerabilities],
                "scan_metadata": report.scan_result.scan_metadata,
                "performance_metrics": report.scan_result.performance_metrics
            },
            "executive_summary": report.executive_summary,
            "detailed_analysis": report.detailed_analysis,
            "remediation_plan": report.remediation_plan,
            "compliance_assessment": report.compliance_assessment,
            "appendices": report.appendices,
            "generated_at": report.generated_at.isoformat(),
            "report_version": report.report_version,
            "generator_info": {
                "reportlab_available": self.reportlab_available,
                "generator": "ZeroVault Report Generator",
                "version": "2.0"
            }
        }
    
    # Placeholder methods for comprehensive analysis (implement as needed)
    def _analyze_vulnerability_patterns(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        return {"pattern_analysis": "completed", "total_patterns": len(set(v.owasp_category for v in vulnerabilities))}
    
    def _map_to_owasp_categories(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        category_mapping = {}
        for vuln in vulnerabilities:
            if vuln.owasp_category:
                if vuln.owasp_category not in category_mapping:
                    category_mapping[vuln.owasp_category] = {
                        "name": self.owasp_llm_categories.get(vuln.owasp_category, "Unknown"),
                        "count": 0
                    }
                category_mapping[vuln.owasp_category]["count"] += 1
        return category_mapping
    
    def _analyze_confidence_distribution(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        if not vulnerabilities:
            return {"average_confidence": 0, "high_confidence_count": 0}
        
        confidences = [v.confidence for v in vulnerabilities]
        return {
            "average_confidence": sum(confidences) / len(confidences),
            "high_confidence_count": len([c for c in confidences if c > 0.8])
        }
    
    def _analyze_attack_vectors(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        return {"attack_vectors_analyzed": len(vulnerabilities), "unique_vectors": len(set(v.owasp_category for v in vulnerabilities))}
    
    def _analyze_affected_components(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        all_components = []
        for vuln in vulnerabilities:
            all_components.extend(vuln.affected_components)
        return {"total_affected_components": len(set(all_components))}
    
    def _extract_technical_details(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        return {"technical_analysis": "completed", "vulnerabilities_analyzed": len(vulnerabilities)}
    
    def _prioritize_vulnerabilities(self, vulnerabilities: List[Vulnerability]) -> List[Dict[str, Any]]:
        prioritized = []
        for vuln in vulnerabilities:
            priority_score = self._calculate_priority_score(vuln)
            prioritized.append({
                "vulnerability": asdict(vuln),
                "priority_score": priority_score
            })
        return sorted(prioritized, key=lambda x: x['priority_score'], reverse=True)
    
    def _calculate_priority_score(self, vuln: Vulnerability) -> float:
        severity_weights = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 7,
            SeverityLevel.MEDIUM: 4,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1
        }
        return severity_weights.get(vuln.severity, 1) * vuln.confidence
    
    def _create_remediation_phases(self, prioritized_vulns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {"phase": 1, "description": "Critical vulnerabilities", "timeline": "Immediate"},
            {"phase": 2, "description": "High priority vulnerabilities", "timeline": "1-2 weeks"},
            {"phase": 3, "description": "Medium priority vulnerabilities", "timeline": "1 month"}
        ]
    
    def _calculate_remediation_timeline(self, prioritized_vulns: List[Dict[str, Any]]) -> str:
        critical_count = len([v for v in prioritized_vulns if v['vulnerability']['severity'] == 'Critical'])
        if critical_count > 0:
            return "Immediate action required"
        return "2-4 weeks for full remediation"
    
    def _define_success_metrics(self) -> List[str]:
        return [
            "Zero critical vulnerabilities",
            "Less than 5 high-severity vulnerabilities",
            "90% reduction in overall risk score"
        ]
    
    def _create_implementation_roadmap(self, phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"phases": len(phases), "total_timeline": "1-3 months", "success_criteria": "All phases completed"}
    
    def _assess_compliance_standard(self, vulnerabilities: List[Vulnerability], standard: ComplianceStandard, framework: Dict[str, Any]) -> Dict[str, Any]:
        critical_vulns = len([v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL])
        compliance_score = max(0, 100 - (critical_vulns * 20))
        
        return {
            "framework_name": framework["name"],
            "compliance_score": compliance_score,
            "status": "Non-compliant" if compliance_score < 70 else "Compliant",
            "key_requirements": framework["key_requirements"],
            "penalty_range": framework["penalty_range"]
        }
    
    def _calculate_overall_compliance_score(self, compliance_results: Dict[str, Any]) -> float:
        if not compliance_results:
            return 0.0
        scores = [result["compliance_score"] for result in compliance_results.values()]
        return sum(scores) / len(scores)
    
    def _identify_critical_compliance_gaps(self, compliance_results: Dict[str, Any]) -> List[str]:
        gaps = []
        for standard, result in compliance_results.items():
            if result["compliance_score"] < 70:
                gaps.append(f"{standard}: {result['status']}")
        return gaps
    
    def _format_detailed_vulnerabilities(self, vulnerabilities: List[Vulnerability]) -> List[Dict[str, Any]]:
        return [asdict(vuln) for vuln in vulnerabilities]
    
    def _format_test_results(self, scan_result: ScanResult) -> Dict[str, Any]:
        return {
            "scan_metadata": scan_result.scan_metadata,
            "performance_metrics": scan_result.performance_metrics,
            "total_vulnerabilities": len(scan_result.vulnerabilities)
        }
    
    def _generate_glossary(self) -> Dict[str, str]:
        return {
            "OWASP": "Open Web Application Security Project",
            "CVE": "Common Vulnerabilities and Exposures",
            "CVSS": "Common Vulnerability Scoring System",
            "LLM": "Large Language Model"
        }
    
    def _generate_references(self) -> List[str]:
        return [
            "OWASP LLM Top 10",
            "NIST Cybersecurity Framework",
            "ISO 27001 Security Standards"
        ]
    
    def _document_methodology(self) -> Dict[str, Any]:
        return {
            "approach": "Automated vulnerability scanning with AI-powered analysis",
            "tools_used": ["ZeroVault Scanner", "AI Analysis Engine"],
            "standards_followed": ["OWASP LLM Top 10", "NIST Framework"]
        }

# Create global instance
report_generator = ReportGenerator()

# Backward compatibility
class ZeroVaultReportGenerator(ReportGenerator):
    """Alias for backward compatibility"""
    pass

# Export for easy imports
__all__ = ['ReportGenerator', 'ZeroVaultReportGenerator', 'report_generator', 'SeverityLevel', 'ComplianceStandard', 'Vulnerability', 'ScanResult', 'VulnerabilityReport']
