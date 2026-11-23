# ============================================================
# GenResearch — Ollama Embedder
# Async document and query embeddings for the existing Chroma RAG store
# ============================================================
import asyncio

import httpx

from config import settings

_client: httpx.AsyncClient | None = None
_BATCH_SIZE = 100
_MAX_EMBED_RETRIES = 2
_embedding_semaphore = asyncio.Semaphore(1)


class OllamaEmbeddingError(RuntimeError):
    """Raised when the configured Ollama server cannot produce embeddings."""


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        timeout = httpx.Timeout(
            connect=settings.OLLAMA_CONNECT_TIMEOUT,
            read=settings.OLLAMA_REQUEST_TIMEOUT,
            write=30.0,
            pool=30.0,
        )
        _client = httpx.AsyncClient(timeout=timeout)
    return _client


async def close_embedder_client() -> None:
    """Close the reusable Ollama embedding client during application shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _with_task_prefix(texts: list[str], task_type: str) -> list[str]:
    if not settings.OLLAMA_EMBED_PREFIXES:
        return texts
    prefix = "search_document: " if task_type == "RETRIEVAL_DOCUMENT" else "search_query: "
    return [f"{prefix}{text}" for text in texts]


async def _embed_batch(
    client: httpx.AsyncClient,
    texts: list[str],
    task_type: str,
) -> list[list[float]]:
    payload = {
        "model": settings.OLLAMA_EMBED_MODEL,
        "input": _with_task_prefix(texts, task_type),
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
    }
    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embed"

    for attempt in range(_MAX_EMBED_RETRIES + 1):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as error:
            if (
                error.response.status_code != 429
                and error.response.status_code < 500
            ) or attempt >= _MAX_EMBED_RETRIES:
                status_code = error.response.status_code
                hint = (
                    " Check that the embedding model is installed on the Ollama server."
                    if status_code == 404
                    else ""
                )
                raise OllamaEmbeddingError(
                    f"Ollama embedding request failed with HTTP {status_code}.{hint}"
                ) from error
            await asyncio.sleep(min(2 ** attempt, 8))
        except httpx.RequestError as error:
            if attempt >= _MAX_EMBED_RETRIES:
                raise OllamaEmbeddingError(
                    "Could not connect to the configured Ollama embedding server. "
                    "Check OLLAMA_BASE_URL and confirm the server is reachable."
                ) from error
            await asyncio.sleep(min(2 ** attempt, 8))

    try:
        body = response.json()
        embeddings = body.get("embeddings") if isinstance(body, dict) else None
    except ValueError as error:
        raise OllamaEmbeddingError("Ollama returned an invalid embedding response.") from error

    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        returned_count = len(embeddings) if isinstance(embeddings, list) else 0
        raise OllamaEmbeddingError(
            f"Ollama returned {returned_count} embeddings for a batch of {len(texts)} texts."
        )
    if any(
        not isinstance(vector, list)
        or not vector
        or any(not isinstance(value, (int, float)) for value in vector)
        for vector in embeddings
    ):
        raise OllamaEmbeddingError("Ollama returned a missing or invalid embedding vector.")
    if any(len(vector) != len(embeddings[0]) for vector in embeddings):
        raise OllamaEmbeddingError("Ollama returned embeddings with inconsistent dimensions.")
    return embeddings


async def _embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    if not texts:
        return []
    client = _get_client()
    batches = [
        texts[index:index + _BATCH_SIZE]
        for index in range(0, len(texts), _BATCH_SIZE)
    ]
    vectors = []
    for batch in batches:
        async with _embedding_semaphore:
            vectors.extend(await _embed_batch(client, batch, task_type))
    return vectors


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing in ChromaDB."""
    return await _embed_texts(texts, "RETRIEVAL_DOCUMENT")


async def embed_single(text: str) -> list[float]:
    """Embed a search query in the same vector space as indexed document chunks."""
    result = await _embed_texts([text], "RETRIEVAL_QUERY")
    return result[0]
