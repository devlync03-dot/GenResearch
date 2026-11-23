# ============================================================
# GenResearch — Reports Router (C3)
# On-demand PDF generation from stored Supabase reports.
# ============================================================
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from database.supabase_client import get_supabase
from services.agents.citation_agent import format_bibtex
from services.pdf_renderer import render_markdown_to_pdf

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{session_id}/references.bib")
async def get_references_bibtex(session_id: str):
    """Download the report's source registry as a BibTeX file."""
    sb = get_supabase()
    result = await asyncio.to_thread(
        sb.table("research_reports")
        .select("citation_registry")
        .eq("session_id", session_id)
        .execute
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found for this session.")
    references = result.data[0].get("citation_registry") or []
    if not isinstance(references, list):
        raise HTTPException(status_code=500, detail="Report citation metadata is invalid.")
    bibtex = format_bibtex([
        reference for reference in references
        if isinstance(reference, dict) and reference.get("evidence_level") != "none"
    ])
    if not bibtex:
        raise HTTPException(status_code=404, detail="No references are available for this report.")
    return Response(
        content=bibtex,
        media_type="application/x-bibtex",
        headers={"Content-Disposition": 'attachment; filename="references.bib"'},
    )


@router.get("/{session_id}/draft.pdf")
async def get_draft_pdf(session_id: str):
    """Generates and returns the Draft PDF on-demand."""
    sb = get_supabase()
    res = await asyncio.to_thread(
        sb.table("research_reports")
        .select("topic, draft_text, citation_registry")
        .eq("session_id", session_id)
        .execute
    )
    
    if not res.data:
        raise HTTPException(status_code=404, detail="Report not found for this session.")
        
    report = res.data[0]
    topic = report.get("topic", "Research Draft")
    draft_text = report.get("draft_text", "")
    citation_registry = report.get("citation_registry") or []
    
    if not draft_text:
        raise HTTPException(status_code=404, detail="Draft text not available yet.")
        
    # Resolve internal citation tags and append the alphabetized references.
    pdf_bytes = await asyncio.to_thread(
        render_markdown_to_pdf,
        draft_text,
        title=topic,
        citation_registry=citation_registry,
    )
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="draft_{session_id}.pdf"'}
    )


@router.get("/{session_id}/guide.pdf")
async def get_guide_pdf(session_id: str):
    """Generates and returns the Completion Guide PDF on-demand."""
    sb = get_supabase()
    res = await asyncio.to_thread(
        sb.table("research_reports")
        .select("topic, completion_guide")
        .eq("session_id", session_id)
        .execute
    )
    
    if not res.data:
        raise HTTPException(status_code=404, detail="Report not found for this session.")
        
    report = res.data[0]
    topic = report.get("topic", "Research Draft")
    completion_guide = report.get("completion_guide", "")
    
    if not completion_guide:
        raise HTTPException(status_code=404, detail="Completion guide not available yet.")
        
    pdf_bytes = await asyncio.to_thread(
        render_markdown_to_pdf, completion_guide, title=f"Guide: {topic}"
    )
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="guide_{session_id}.pdf"'}
    )
