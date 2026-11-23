# ============================================================
# GenResearch — Agent Tasks Router
# API endpoints for all 4 research agents
# ============================================================
import asyncio
import json
import uuid
from io import BytesIO
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from services.agents.summarization_agent import run_summarization
from services.agents.literature_review_agent import run_literature_review
from services.agents.citation_agent import (
    find_evidence_for_claim,
    format_accepted_references,
    run_citation_workflow,
    run_citation_extraction,
)
from services.agents.proposal_graph import run_proposal_draft_stream
from services import request_context, tracking

router = APIRouter(prefix="/agents", tags=["agents"])
TASK_RESULT_TYPES = {"summarization": "summary", "literature_review": "review"}
_active_tasks: dict[str, tuple[str, asyncio.Task]] = {}


async def _run_tracked_agent(
    user_id: str,
    title: str,
    agent_type: str,
    paper_ids: list[str],
    run,
    persist_result: bool = True,
):
    """Run an agent while tracking task status and optionally persisting its result."""
    task_id = await asyncio.to_thread(
        tracking.create_task, user_id, title, agent_type, len(paper_ids)
    )
    current_task = asyncio.current_task()
    if task_id and current_task:
        _active_tasks[task_id] = (user_id, current_task)
    user_token = request_context.set_user_id(user_id)
    try:
        result = await run()
        summaries = result.get("summaries", []) if isinstance(result, dict) else []
        failed_paper = any(
            isinstance(summary, dict)
            and summary.get("status") == "error"
            for summary in summaries
        )
        failed_run = (
            failed_paper
            or isinstance(result, dict)
            and result.get("status") == "error"
        )
        await asyncio.to_thread(
            tracking.complete_task, task_id, "failed" if failed_run else "completed"
        )
        if persist_result:
            await asyncio.to_thread(
                tracking.save_task_result,
                task_id=task_id,
                user_id=user_id,
                type_=TASK_RESULT_TYPES.get(agent_type, agent_type),
                title=title,
                content=json.dumps(result, ensure_ascii=False, default=str),
            )
        await asyncio.to_thread(
            tracking.log_agent_event,
            "warning" if failed_run else "success",
            (
                f"{title} failed for one or more papers."
                if failed_paper
                else f"{title} failed."
                if failed_run
                else f"Completed {title} for {len(paper_ids)} paper(s)."
            ),
            agent=agent_type,
            user_id=user_id,
        )
        return result
    except Exception as error:
        await asyncio.to_thread(tracking.complete_task, task_id, "failed")
        await asyncio.to_thread(
            tracking.log_agent_event,
            "error", f"{title} failed: {str(error)[:1800]}", agent=agent_type, user_id=user_id
        )
        raise
    except asyncio.CancelledError:
        await asyncio.to_thread(tracking.complete_task, task_id, "failed")
        await asyncio.to_thread(
            tracking.log_agent_event,
            "warning", f"{title} was cancelled.", agent=agent_type, user_id=user_id
        )
        raise
    finally:
        if task_id and _active_tasks.get(task_id, (None, None))[1] is current_task:
            _active_tasks.pop(task_id, None)
        request_context.reset_user_id(user_token)


# ── Request Models ────────────────────────────────────────────

class AgentRequest(BaseModel):
    user_id: str
    paper_ids: list[str]


class SummarizationRequest(AgentRequest):
    detail: Literal["quick", "standard", "deep"] = "standard"


class LitReviewRequest(BaseModel):
    user_id: str
    paper_ids: list[str]
    focus_topic: str = ""


class CitationRequest(BaseModel):
    user_id: str
    paper_ids: list[str] = Field(default_factory=list)
    style: str = "apa"  # apa, mla, ieee, chicago
    draft_text: str | None = Field(default=None, max_length=30_000)


class CitationAuditRequest(BaseModel):
    user_id: str
    text: str = Field(min_length=1, max_length=500_000)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)


class EvidenceSearchRequest(BaseModel):
    user_id: str
    claim: str = Field(min_length=1, max_length=3_000)
    claim_id: str | None = Field(default=None, max_length=100)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)


class CitationWriteCheckRequest(BaseModel):
    user_id: str
    text: str = Field(min_length=1, max_length=10_000)
    citation: str | None = Field(default=None, max_length=3_000)
    evidence_text: str | None = Field(default=None, max_length=5_000)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)


class CitationDecisionSource(BaseModel):
    id: str = Field(max_length=100)
    title: str = Field(max_length=1_000)
    authors: str | None = Field(default=None, max_length=2_000)
    year: int | None = None
    doi: str | None = Field(default=None, max_length=300)
    url: str | None = Field(default=None, max_length=2_000)
    retraction_status: str | None = Field(default=None, max_length=50)


class CitationDecisionRequest(BaseModel):
    user_id: str
    claim_id: str = Field(min_length=1, max_length=100)
    claim: str = Field(min_length=1, max_length=3_000)
    action: Literal["accepted", "rejected", "edited"]
    source: CitationDecisionSource | None = None
    edited_claim: str | None = Field(default=None, max_length=3_000)

    @model_validator(mode="after")
    def require_decision_details(self) -> "CitationDecisionRequest":
        if self.action == "accepted" and self.source is None:
            raise ValueError("An accepted decision must identify a source.")
        if self.action == "edited" and not (self.edited_claim or "").strip():
            raise ValueError("An edited decision must include the edited claim.")
        return self


class CitationFormatReference(BaseModel):
    title: str = Field(min_length=1, max_length=1_000)
    authors: str | None = Field(default=None, max_length=2_000)
    year: int | None = None
    venue: str | None = Field(default=None, max_length=1_000)
    journal: str | None = Field(default=None, max_length=1_000)
    volume: str | None = Field(default=None, max_length=100)
    issue: str | None = Field(default=None, max_length=100)
    pages: str | None = Field(default=None, max_length=100)
    doi: str | None = Field(default=None, max_length=300)
    url: str | None = Field(default=None, max_length=2_000)


class CitationFormatRequest(BaseModel):
    references: list[CitationFormatReference] = Field(min_length=1, max_length=50)
    style: Literal["apa", "mla", "ieee", "chicago"] = "apa"


class ProposalRequest(BaseModel):
    user_id: str
    paper_ids: list[str]
    topic: str


# ── POST /agents/summarize ────────────────────────────────────

@router.post("/summarize")
async def summarize_papers(req: SummarizationRequest):
    """
    Summarize one or more papers.
    Returns structured summaries with key findings, methodology, and conclusions.
    """
    if not req.paper_ids:
        raise HTTPException(status_code=400, detail="At least one paper_id is required.")
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    try:
        result = await _run_tracked_agent(
            user_id=req.user_id,
            title="Paper summarization",
            agent_type="summarization",
            paper_ids=req.paper_ids,
            run=lambda: run_summarization(
                user_id=req.user_id,
                paper_ids=req.paper_ids,
                detail=req.detail,
            ),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")


@router.post("/summarize/stream")
async def summarize_papers_stream(req: SummarizationRequest):
    """Stream progress updates and the final structured summary as NDJSON."""
    if not req.paper_ids:
        raise HTTPException(status_code=400, detail="At least one paper_id is required.")
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    async def stream_events():
        events: asyncio.Queue[dict | None] = asyncio.Queue()

        async def report_progress(message: str) -> None:
            await events.put({"type": "progress", "message": message})

        async def run_agent() -> None:
            try:
                result = await _run_tracked_agent(
                    user_id=req.user_id,
                    title="Paper summarization",
                    agent_type="summarization",
                    paper_ids=req.paper_ids,
                    run=lambda: run_summarization(
                        user_id=req.user_id,
                        paper_ids=req.paper_ids,
                        progress_callback=report_progress,
                        detail=req.detail,
                    ),
                )
                await events.put({"type": "done", "result": result})
            except Exception as error:
                await events.put({"type": "error", "message": f"Summarization failed: {error}"})
            finally:
                await events.put(None)

        task = asyncio.create_task(run_agent())
        try:
            while True:
                event = await events.get()
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False, default=str) + "\n"
            await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        stream_events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── POST /agents/literature-review ────────────────────────────

@router.post("/literature-review")
async def literature_review(req: LitReviewRequest):
    """
    Generate a comparative literature review from multiple papers.
    Optionally focuses on a specific topic.
    """
    if not req.paper_ids:
        raise HTTPException(status_code=400, detail="At least one paper_id is required.")
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    try:
        result = await _run_tracked_agent(
            user_id=req.user_id,
            title=req.focus_topic or "Literature review",
            agent_type="literature_review",
            paper_ids=req.paper_ids,
            run=lambda: run_literature_review(
                user_id=req.user_id,
                paper_ids=req.paper_ids,
                focus_topic=req.focus_topic,
            ),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Literature review failed: {e}")


@router.post("/literature-review/stream")
async def literature_review_stream(req: LitReviewRequest):
    """Stream literature-review progress and the final structured result as NDJSON."""
    if not req.paper_ids:
        raise HTTPException(status_code=400, detail="At least one paper_id is required.")
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    async def stream_events():
        events: asyncio.Queue[dict | None] = asyncio.Queue()

        async def report_progress(message: str) -> None:
            await events.put({"type": "progress", "message": message})

        async def run_agent() -> None:
            try:
                result = await _run_tracked_agent(
                    user_id=req.user_id,
                    title=req.focus_topic or "Literature review",
                    agent_type="literature_review",
                    paper_ids=req.paper_ids,
                    run=lambda: run_literature_review(
                        user_id=req.user_id,
                        paper_ids=req.paper_ids,
                        focus_topic=req.focus_topic,
                        progress_callback=report_progress,
                    ),
                )
                await events.put({"type": "done", "result": result})
            except Exception as error:
                await events.put({
                    "type": "error",
                    "message": f"Literature review failed: {error}",
                })
            finally:
                await events.put(None)

        task = asyncio.create_task(run_agent())
        try:
            while True:
                event = await events.get()
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False, default=str) + "\n"
            await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        stream_events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── POST /agents/citations ───────────────────────────────────

@router.post("/citations")
async def extract_citations(req: CitationRequest):
    """
    Extract and format citations from papers.
    Supports APA, MLA, IEEE, and Chicago styles.
    """
    if not req.paper_ids and not (req.draft_text or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Select at least one paper or provide draft text to audit.",
        )
    if req.style.lower() not in ("apa", "mla", "ieee", "chicago"):
        raise HTTPException(status_code=400, detail="Style must be: apa, mla, ieee, or chicago.")

    try:
        result = await _run_tracked_agent(
            user_id=req.user_id,
            title=f"Citation extraction ({req.style.upper()})",
            agent_type="citation",
            paper_ids=req.paper_ids,
            run=lambda: run_citation_extraction(
                user_id=req.user_id,
                paper_ids=req.paper_ids,
                style=req.style,
                draft_text=req.draft_text,
            ),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Citation extraction failed: {e}")


MAX_AUDIT_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_AUDIT_TEXT_CHARS = 500_000


async def _read_audit_document(file: UploadFile) -> str:
    filename = file.filename or ""
    suffix = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
    if suffix not in {"pdf", "docx", "txt"}:
        raise HTTPException(status_code=415, detail="Upload a PDF, DOCX, or TXT document.")
    content = await file.read(MAX_AUDIT_UPLOAD_BYTES + 1)
    if len(content) > MAX_AUDIT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Document exceeds the 12 MB upload limit.")
    if suffix == "pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=415, detail="The uploaded file is not a valid PDF.")
        from services.pdf_extractor import extract_text_from_pdf

        text = await asyncio.to_thread(extract_text_from_pdf, content)
    elif suffix == "docx":
        from docx import Document

        def extract_docx() -> str:
            document = Document(BytesIO(content))
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            table_rows = [
                " | ".join(cell.text.strip() for cell in row.cells)
                for table in document.tables
                for row in table.rows
            ]
            return "\n".join([*paragraphs, *table_rows])

        try:
            text = await asyncio.to_thread(extract_docx)
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"Could not read the DOCX document: {error}") from error
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise HTTPException(status_code=422, detail="TXT documents must use UTF-8 encoding.") from error

    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="No readable text was extracted. Scanned PDFs need OCR before audit.",
        )
    if len(text) > MAX_AUDIT_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Extracted document text exceeds the {MAX_AUDIT_TEXT_CHARS:,}-character audit limit.",
        )
    return text


async def _audit_document(user_id: str, text: str, paper_ids: list[str]) -> dict[str, Any]:
    return await _run_tracked_agent(
        user_id=user_id,
        title="Citation document audit",
        agent_type="citation",
        paper_ids=paper_ids,
        run=lambda: run_citation_workflow("audit", user_id, paper_ids, text),
        persist_result=False,
    )


@router.post("/citations/audit")
async def audit_citation_text(req: CitationAuditRequest):
    """Audit pasted text without automatically searching for replacement sources."""
    try:
        return await _audit_document(req.user_id, req.text, req.paper_ids)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Citation audit failed: {error}") from error


@router.post("/citations/audit-document")
async def audit_citation_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    paper_ids_json: str = Form(default="[]"),
):
    """Extract and audit a PDF, DOCX, or UTF-8 text document."""
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    try:
        paper_ids = json.loads(paper_ids_json)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="paper_ids_json must be a JSON string array.") from error
    if (
        not isinstance(paper_ids, list)
        or len(paper_ids) > 20
        or any(not isinstance(paper_id, str) or not paper_id for paper_id in paper_ids)
    ):
        raise HTTPException(status_code=400, detail="paper_ids_json must contain up to 20 paper IDs.")
    text = await _read_audit_document(file)
    try:
        result = await _audit_document(user_id, text, paper_ids)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Citation audit failed: {error}") from error
    result["document_name"] = file.filename
    return result


@router.post("/citations/find-evidence")
async def find_citation_evidence(req: EvidenceSearchRequest):
    """Search and rank sources only for the claim the user explicitly selected."""
    user_token = request_context.set_user_id(req.user_id)
    try:
        result = await find_evidence_for_claim(req.user_id, req.paper_ids, req.claim)
        result["claim_id"] = req.claim_id
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Evidence search failed: {error}") from error
    finally:
        request_context.reset_user_id(user_token)


@router.post("/citations/write-check")
async def check_citation_text(req: CitationWriteCheckRequest):
    """Check text against its citation or search for candidate evidence."""
    user_token = request_context.set_user_id(req.user_id)
    try:
        return await run_citation_workflow(
            "check",
            req.user_id,
            req.paper_ids,
            req.text,
            citation=req.citation,
            evidence_text=req.evidence_text,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Write-and-check failed: {error}") from error
    finally:
        request_context.reset_user_id(user_token)


@router.post("/citations/decision")
async def log_citation_decision(req: CitationDecisionRequest):
    """Record an explicit accept/reject/edit decision without changing document text."""
    source_title = req.source.title if req.source else ""
    source_id = req.source.id if req.source else "none"
    source_doi = req.source.doi if req.source else "none"
    tracking.log_agent_event(
        "success",
        (
            f"User {req.action} citation evidence for claim {req.claim_id}; "
            f"source_id={source_id}; title={source_title or 'none'}; DOI={source_doi}."
        ),
        agent="citation_decision",
        user_id=req.user_id,
    )
    tracking.log_audit_event("citation.user_decision")
    return {"recorded": True, "claim_id": req.claim_id, "action": req.action}


@router.post("/citations/format")
async def format_citation_decisions(req: CitationFormatRequest):
    """Format only references the user has accepted."""
    from services.agents.citation_agent import format_accepted_references

    try:
        return format_accepted_references(
            [reference.model_dump(exclude_none=True) for reference in req.references],
            req.style,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# ── POST /agents/proposal ────────────────────────────────────
#
# @router.post("/proposal")
# async def draft_proposal(req: ProposalRequest):
#     """
#     Draft a complete research proposal using LangGraph orchestration.
#     Invokes all sub-agents (summarization, literature review, citation)
#     and composes a publication-ready proposal.
#     """
#     if not req.paper_ids:
#         raise HTTPException(status_code=400, detail="At least one paper_id is required.")
#     if not req.topic.strip():
#         raise HTTPException(status_code=400, detail="Research topic is required.")
#
#     try:
#         # result = await run_proposal_draft(
#         #     user_id=req.user_id,
#         #     paper_ids=req.paper_ids,
#         #     topic=req.topic,
#         # )
#         # return result
#         raise NotImplementedError("run_proposal_draft is not implemented.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Proposal drafting failed: {e}")

# ── POST /agents/proposal-stream ────────────────────────────

@router.post("/proposal-stream")
async def draft_proposal_stream(req: ProposalRequest):
    """
    Draft a complete research proposal using LangGraph orchestration.
    Streams intermediate states via SSE.
    """
    if not req.paper_ids:
        raise HTTPException(status_code=400, detail="At least one paper_id is required.")
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Research topic is required.")

    async def tracked_proposal_stream():
        task_id = await asyncio.to_thread(
            tracking.create_task,
            req.user_id,
            req.topic,
            "proposal",
            paper_count=len(req.paper_ids),
        )
        current_task = asyncio.current_task()
        if task_id and current_task:
            _active_tasks[task_id] = (req.user_id, current_task)
        user_token = request_context.set_user_id(req.user_id)
        session_token = request_context.set_session_id(str(uuid.uuid4()))
        final_result = None
        try:
            async for line in run_proposal_draft_stream(
                user_id=req.user_id,
                paper_ids=req.paper_ids,
                topic=req.topic,
            ):
                try:
                    event = json.loads(line)
                    if event.get("type") == "done":
                        final_result = event.get("final_result")
                except (TypeError, json.JSONDecodeError):
                    pass
                yield line

            await asyncio.to_thread(tracking.complete_task, task_id, "completed")
            if final_result:
                await asyncio.to_thread(
                    tracking.save_task_result,
                    task_id=task_id,
                    user_id=req.user_id,
                    type_="proposal",
                    title=req.topic,
                    content=json.dumps(final_result, ensure_ascii=False, default=str),
                )
            await asyncio.to_thread(
                tracking.log_agent_event,
                "success",
                f"Completed proposal drafting for {len(req.paper_ids)} paper(s).",
                agent="proposal",
                user_id=req.user_id,
            )
        except Exception as error:
            await asyncio.to_thread(tracking.complete_task, task_id, "failed")
            await asyncio.to_thread(
                tracking.log_agent_event,
                "error", f"Proposal drafting failed: {str(error)[:1800]}",
                agent="proposal", user_id=req.user_id,
            )
            raise
        except asyncio.CancelledError:
            await asyncio.to_thread(tracking.complete_task, task_id, "failed")
            await asyncio.to_thread(
                tracking.log_agent_event,
                "warning",
                "Proposal drafting was cancelled.",
                agent="proposal",
                user_id=req.user_id,
            )
            raise
        finally:
            if task_id and _active_tasks.get(task_id, (None, None))[1] is current_task:
                _active_tasks.pop(task_id, None)
            request_context.reset_session_id(session_token)
            request_context.reset_user_id(user_token)

    return StreamingResponse(tracked_proposal_stream(), media_type="application/x-ndjson")


# ── POST /agents/cancel ──────────────────────────────────────

class CancelRequest(BaseModel):
    user_id: str
    task_id: str = ""


@router.post("/cancel")
async def cancel_agent_task(req: CancelRequest):
    """Request cancellation of a task active on this worker and owned by the user."""
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    if not req.task_id:
        raise HTTPException(status_code=400, detail="task_id is required.")

    active_task = _active_tasks.get(req.task_id)
    if active_task is None or active_task[0] != req.user_id:
        raise HTTPException(status_code=404, detail="Active task not found.")

    active_task[1].cancel()
    return {"status": "cancellation_requested", "task_id": req.task_id}
