"""
Reports route — list, fetch, export, delete, compare reports + audit log.
"""
from fastapi import APIRouter, HTTPException, Response
from typing import List

from backend.services.report_service import report_service
from backend.services.audit_service import audit_service
from backend.api.schemas.schemas import (
    FullAnalysisReport, ComparisonResult, AuditEntry, AuditStats
)

router = APIRouter()


@router.get("/", response_model=List[FullAnalysisReport])
async def list_reports():
    return report_service.list_reports()


@router.get("/audit", response_model=List[AuditEntry])
async def get_audit_log(limit: int = 100):
    return audit_service.get_recent(limit=limit)


@router.get("/audit/stats", response_model=AuditStats)
async def get_audit_stats():
    stats = audit_service.get_stats()
    if not stats:
        raise HTTPException(404, "No audit data available")
    return stats


@router.get("/compare")
async def compare_reports(report_a: str, report_b: str) -> ComparisonResult:
    result = report_service.compare_reports(report_a.upper(), report_b.upper())
    if not result:
        raise HTTPException(404, "One or both reports not found or have no prediction")
    return result


@router.get("/{report_id}", response_model=FullAnalysisReport)
async def get_report(report_id: str):
    report = report_service.get_report(report_id.upper())
    if not report:
        raise HTTPException(404, f"Report not found: {report_id}")
    return report


@router.get("/{report_id}/pdf")
async def export_pdf(report_id: str):
    pdf_bytes = report_service.generate_pdf(report_id.upper())
    if pdf_bytes is None:
        raise HTTPException(404, "Report not found or PDF generation failed")
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=mri_report_{report_id}.pdf"},
    )


@router.delete("/{report_id}")
async def delete_report(report_id: str):
    if not report_service.delete_report(report_id.upper()):
        raise HTTPException(404, f"Report not found: {report_id}")
    return {"message": "Report deleted"}