# ============================================================
# GenResearch — Papers Router
# Upload PDFs → Extract → Chunk → Embed → Store in ChromaDB
# Sync metadata to Supabase papers table
# ============================================================
import asyncio
import uuid
import logging
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from config import settings
from database.chroma_client import get_user_collection
from database.supabase_client import get_supabase
from services.pdf_extractor import extract_text_from_pdf, get_pdf_page_count
from services.chunker import chunk_text
from services.chroma_service import store_chunks, store_structured_chunks, delete_paper_chunks
from services.embedder import OllamaEmbeddingError
from ingestion import chunk_document, parse_pdf
from ingestion.pdf_parser import ParsedBlock
from ingestion.table_extractor import extract_tables

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage" / "pdfs"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAPERS_PER_USER = 20

def _format_file_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _index_error_message(error: Exception) -> str:
    if isinstance(error, OllamaEmbeddingError):
        return (
            f"{error} The PDF is saved but was not indexed; check Ollama and retry."
        )
    return str(error)


def _check_user_quota(sb, user_id: str):
    """Enforce user quota: max 20 papers."""
    try:
        res = sb.table("papers").select("id").eq("user_id", user_id).execute()
        papers = res.data or []
        if len(papers) >= MAX_PAPERS_PER_USER:
            raise HTTPException(
                status_code=400,
                detail=f"User quota exceeded: Maximum {MAX_PAPERS_PER_USER} papers allowed per account.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to check user quota: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Database error during quota check: {exc}",
        )


async def _execute_query(query):
    return await asyncio.to_thread(query.execute)


def _save_pdf_file(user_id: str, paper_id: str, file_bytes: bytes) -> str:
    """Save raw PDF file to server storage."""
    user_dir = STORAGE_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / f"{paper_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return str(file_path)


async def _schedule_structure_extraction(user_id: str, paper_id: str) -> None:
    """Run document structure extraction after upload but never let it alter upload success."""
    if not settings.DOC_STRUCTURE_ON_UPLOAD:
        return
    try:
        from services.doc_structure.pipeline import extract_structure, persist_structure

        pdf_path = STORAGE_DIR / user_id / f"{paper_id}.pdf"
        if not pdf_path.exists():
            return
        with pdf_path.open("rb") as fh:
            pdf_bytes = fh.read()
        structure = await asyncio.to_thread(extract_structure, pdf_bytes)
        await persist_structure(user_id, paper_id, structure)
    except Exception as exc:
        logger.warning("post_upload_structure_extraction_failed", extra={"paper_id": paper_id, "error": str(exc)}, exc_info=True)


async def _index_pdf(
    user_id: str,
    paper_id: str,
    title: str,
    collection: str,
    file_bytes: bytes,
) -> tuple[int, int]:
    text = await extract_text_from_pdf(file_bytes)
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    pages = await get_pdf_page_count(file_bytes)
    parsed_blocks = await asyncio.to_thread(parse_pdf, file_bytes)
    parsed_blocks.extend(
        ParsedBlock(
            text=f"Table {table['table']} on page {table['page']}: {table['rows']}",
            page=table["page"],
            section_heading=f"Table {table['table']}",
            is_table=True,
            table_data={"rows": table["rows"]},
        )
        for table in await asyncio.to_thread(extract_tables, file_bytes)
    )
    structured_chunks = chunk_document(parsed_blocks, paper_id, title)
    if structured_chunks:
        num_chunks = await store_structured_chunks(
            user_id=user_id,
            chunks=structured_chunks,
            collection_name=collection,
        )
    else:
        chunks = chunk_text(text)
        num_chunks = await store_chunks(
            user_id=user_id,
            paper_id=paper_id,
            chunks=chunks,
            title=title,
            collection_name=collection,
        )

    if num_chunks == 0:
        raise ValueError("No searchable text chunks were created from the PDF.")
    return pages, num_chunks


# ── POST /papers/upload ──────────────────────────────────────
@router.post("/upload")
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    title: str = Form(""),
    authors: str = Form(""),
    year: int = Form(None),
    collection: str = Form(""),
    tags: str = Form(""),  # comma-separated
):
    """
    Upload a single PDF paper:
    1. Insert metadata into Supabase with status='processing'
    2. Extract text from PDF
    3. Chunk text
    4. Embed chunks and store in ChromaDB
    5. Update Supabase with status='indexed' and chunk count
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    file_size = _format_file_size(len(file_bytes))
    paper_id = str(uuid.uuid4())
    paper_title = title or file.filename.rsplit(".", 1)[0]
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    sb = get_supabase()
    await asyncio.to_thread(_check_user_quota, sb, user_id)

    collection_name = collection.strip() or "Library"
    storage_path = _save_pdf_file(user_id, paper_id, file_bytes)

    # Step 1: Insert into Supabase with status='processing'
    try:
        await _execute_query(sb.table("papers").insert({
            "id": paper_id,
            "user_id": user_id,
            "title": paper_title,
            "authors": authors,
            "year": year,
            "file_name": file.filename,
            "file_size": file_size,
            "tags": tag_list,
            "collection": collection_name,
            "status": "processing",
            "chunks": 0,
            "storage_path": storage_path,
        }))
    except Exception as e:
        try:
            os.remove(storage_path)
        except OSError:
            logger.exception("failed_to_remove_pdf_after_supabase_insert_failure", extra={"paper_id": paper_id})
        raise HTTPException(status_code=500, detail=f"Supabase insert failed: {e}")

    # Step 2–4: Extract → Chunk → Embed → Store
    try:
        pages, num_chunks = await _index_pdf(user_id, paper_id, paper_title, collection_name, file_bytes)

        # Step 5: Update Supabase with indexed status
        await _execute_query(sb.table("papers").update({
            "status": "indexed",
            "chunks": num_chunks,
            "pages": pages,
        }).eq("id", paper_id))

        if settings.DOC_STRUCTURE_ON_UPLOAD:
            background_tasks.add_task(_schedule_structure_extraction, user_id, paper_id)

        return {
            "paper_id": paper_id,
            "title": paper_title,
            "file_name": file.filename,
            "file_size": file_size,
            "pages": pages,
            "chunks": num_chunks,
            "status": "indexed",
        }

    except Exception as e:
        # On failure: mark paper as failed
        try:
            await asyncio.to_thread(delete_paper_chunks, user_id, paper_id)
        except Exception:
            logger.exception("failed_to_remove_partial_paper_chunks", extra={"paper_id": paper_id})
        try:
            await _execute_query(
                sb.table("papers")
                .update({"status": "failed", "chunks": 0})
                .eq("id", paper_id)
            )
        except Exception:
            logger.exception("failed_to_mark_paper_failed", extra={"paper_id": paper_id})
        raise HTTPException(
            status_code=503 if isinstance(e, OllamaEmbeddingError) else 500,
            detail=f"Processing failed: {_index_error_message(e)}",
        )


# ── POST /papers/upload-batch ────────────────────────────────
@router.post("/upload-batch")
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    user_id: str = Form(...),
    authors: str = Form(""),
    year: int = Form(None),
    collection: str = Form(""),
    tags: str = Form(""),
):
    """Upload multiple PDFs at once. Each file is processed independently."""
    sb = get_supabase()
    await asyncio.to_thread(_check_user_quota, sb, user_id)

    results = []
    for file in files:
        paper_id: str | None = None
        storage_path: str | None = None
        paper_record_created = False
        try:
            await asyncio.to_thread(_check_user_quota, sb, user_id)
            # Re-use the single upload logic inline
            file_bytes = await file.read()
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                results.append({
                    "file_name": file.filename,
                    "status": "failed",
                    "error": "Not a PDF file",
                })
                continue

            file_size = _format_file_size(len(file_bytes))
            paper_id = str(uuid.uuid4())
            paper_title = (file.filename or "Untitled").rsplit(".", 1)[0]
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            collection_name = collection.strip() or "Library"

            storage_path = _save_pdf_file(user_id, paper_id, file_bytes)

            await _execute_query(sb.table("papers").insert({
                "id": paper_id,
                "user_id": user_id,
                "title": paper_title,
                "authors": authors,
                "year": year,
                "file_name": file.filename,
                "file_size": file_size,
                "tags": tag_list,
                "collection": collection_name,
                "status": "processing",
                "chunks": 0,
                "storage_path": storage_path,
            }))
            paper_record_created = True

            pages, num_chunks = await _index_pdf(user_id, paper_id, paper_title, collection_name, file_bytes)

            await _execute_query(sb.table("papers").update({
                "status": "indexed",
                "chunks": num_chunks,
                "pages": pages,
            }).eq("id", paper_id))

            if settings.DOC_STRUCTURE_ON_UPLOAD:
                background_tasks.add_task(_schedule_structure_extraction, user_id, paper_id)

            results.append({
                "paper_id": paper_id,
                "file_name": file.filename,
                "file_size": file_size,
                "pages": pages,
                "chunks": num_chunks,
                "status": "indexed",
            })

        except Exception as e:
            logger.error(f"Error processing batch upload for {file.filename}: {e}", exc_info=True)
            if paper_id is not None:
                if paper_record_created:
                    try:
                        await asyncio.to_thread(delete_paper_chunks, user_id, paper_id)
                    except Exception:
                        logger.exception("failed_to_remove_partial_paper_chunks", extra={"paper_id": paper_id})
                    try:
                        await _execute_query(
                            sb.table("papers")
                            .update({"status": "failed", "chunks": 0})
                            .eq("id", paper_id)
                        )
                    except Exception:
                        logger.exception("failed_to_mark_paper_failed", extra={"paper_id": paper_id})
                elif storage_path:
                    try:
                        os.remove(storage_path)
                    except OSError:
                        logger.exception("failed_to_remove_pdf_after_supabase_insert_failure", extra={"paper_id": paper_id})
            
            results.append({
                "file_name": file.filename,
                "status": "failed",
                "error": _index_error_message(e),
            })

    return {"uploaded": len(results), "results": results}


# ── POST /papers/{paper_id}/reindex ────────────────────────────
@router.post("/{paper_id}/reindex")
async def retry_index_paper(paper_id: str, user_id: str):
    """Retry indexing a previously failed upload without uploading it again."""
    sb = get_supabase()
    result = await _execute_query(
        sb.table("papers")
        .select("*")
        .eq("id", paper_id)
        .eq("user_id", user_id)
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Paper not found.")

    paper = result.data[0]
    if paper.get("status") != "failed":
        raise HTTPException(status_code=409, detail="Only failed papers can be retried.")

    storage_path = paper.get("storage_path") or str(
        STORAGE_DIR / user_id / f"{paper_id}.pdf"
    )
    if not os.path.exists(storage_path):
        raise HTTPException(
            status_code=404,
            detail="The saved PDF is missing; upload the paper again to retry indexing.",
        )

    claim = await _execute_query(
        sb.table("papers")
        .update({"status": "processing"})
        .eq("id", paper_id)
        .eq("user_id", user_id)
        .eq("status", "failed")
        .select("id")
    )
    if not claim.data:
        raise HTTPException(status_code=409, detail="This paper is already being retried.")
    try:
        pages, num_chunks = await _index_pdf(
            user_id,
            paper_id,
            paper.get("title") or "Untitled",
            paper.get("collection") or "Library",
            await asyncio.to_thread(Path(storage_path).read_bytes),
        )
        await _execute_query(sb.table("papers").update({
            "status": "indexed",
            "chunks": num_chunks,
            "pages": pages,
        }).eq("id", paper_id))
        return {
            "paper_id": paper_id,
            "status": "indexed",
            "pages": pages,
            "chunks": num_chunks,
        }
    except Exception as error:
        try:
            await asyncio.to_thread(delete_paper_chunks, user_id, paper_id)
        except Exception:
            logger.exception(
                "failed_to_remove_partial_paper_chunks",
                extra={"paper_id": paper_id},
            )
        try:
            await _execute_query(
                sb.table("papers")
                .update({"status": "failed", "chunks": 0})
                .eq("id", paper_id)
            )
        except Exception:
            logger.exception(
                "failed_to_mark_paper_failed",
                extra={"paper_id": paper_id},
            )
        status_code = 503 if isinstance(error, OllamaEmbeddingError) else 500
        raise HTTPException(
            status_code=status_code,
            detail=f"Indexing retry failed: {_index_error_message(error)}",
        ) from error


# ── DELETE /papers/{paper_id} ────────────────────────────────
@router.delete("/{paper_id}")
async def delete_paper(paper_id: str, user_id: str):
    """
    Delete a paper from both ChromaDB and Supabase.
    """
    sb = get_supabase()

    # Verify paper exists and belongs to user
    result = await _execute_query(
        sb.table("papers").select("id, user_id, storage_path").eq("id", paper_id)
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Paper not found.")

    paper = result.data[0]
    if paper["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this paper.")

    # Remove stored PDF file if it exists
    storage_path = paper.get("storage_path") or str(STORAGE_DIR / user_id / f"{paper_id}.pdf")
    if os.path.exists(storage_path):
        try:
            os.remove(storage_path)
        except Exception as err:
            logger.warning(f"Could not remove PDF file {storage_path}: {err}")

    # Delete chunks from ChromaDB
    deleted_chunks = await asyncio.to_thread(delete_paper_chunks, user_id, paper_id)

    # Delete row from Supabase
    await _execute_query(sb.table("papers").delete().eq("id", paper_id))

    return {"deleted": True, "paper_id": paper_id, "chunks_removed": deleted_chunks}


# ── GET /papers/{paper_id}/download ──────────────────────────
@router.get("/{paper_id}/download")
async def download_paper(paper_id: str, user_id: str):
    """Serve the stored PDF file for download."""
    sb = get_supabase()
    res = await _execute_query(
        sb.table("papers")
        .select("id, user_id, file_name, storage_path")
        .eq("id", paper_id)
        .eq("user_id", user_id)
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Paper not found.")

    paper = res.data[0]
    file_path = paper.get("storage_path") or str(STORAGE_DIR / user_id / f"{paper_id}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server storage.")

    return FileResponse(path=file_path, filename=paper.get("file_name", "paper.pdf"), media_type="application/pdf")


# ── GET /papers/ ─────────────────────────────────────────────
@router.get("/")
async def list_papers(user_id: str):
    """Return Supabase paper records with live counts from the user's ChromaDB collection."""
    sb = get_supabase()
    result = await _execute_query(
        sb.table("papers")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
    )
    papers = result.data or []
    if not papers:
        return {"papers": []}

    vector_counts = await asyncio.to_thread(
        _get_vector_chunk_counts, user_id, [paper["id"] for paper in papers]
    )

    return {
        "papers": [
            {**paper, "vector_chunks": vector_counts.get(paper["id"], 0)}
            for paper in papers
        ]
    }


def _get_vector_chunk_counts(user_id: str, paper_ids: list[str]) -> dict[str, int]:
    collection = get_user_collection(user_id)
    if not collection.count():
        return {}
    vectors = collection.get(
        where={"paper_id": {"$in": paper_ids}},
        include=["metadatas"],
    )
    counts: dict[str, int] = {}
    for metadata in vectors.get("metadatas") or []:
        metadata = metadata or {}
        paper_id = metadata.get("paper_id")
        if paper_id:
            counts[paper_id] = counts.get(paper_id, 0) + 1
    return counts
