"""Ollama embedding batching, task prefixes, retries, and validation."""
import asyncio

import httpx
import pytest

from config import settings
from services import embedder


def _embedding_response(inputs):
    return {
        "embeddings": [
            [float(index), 0.0]
            for index, _ in enumerate(inputs)
        ]
    }


def test_embedder_batches_reuses_client_and_uses_task_prefixes(monkeypatch):
    requests = []
    clients_created = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        is_closed = False

        def __init__(self, *args, **kwargs):
            clients_created.append(self)

        async def post(self, url, *, json):
            requests.append((url, json))
            return FakeResponse(_embedding_response(json["input"]))

        async def aclose(self):
            self.is_closed = True

    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://ollama.test/")
    monkeypatch.setattr(settings, "OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr(settings, "OLLAMA_KEEP_ALIVE", "30m")
    monkeypatch.setattr(settings, "OLLAMA_EMBED_PREFIXES", True)
    monkeypatch.setattr(embedder.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(embedder, "_client", None)
    monkeypatch.setattr(embedder, "_embedding_semaphore", asyncio.Semaphore(1))

    async def run_calls():
        documents = await embedder.embed_texts(["paper text"])
        query = await embedder.embed_single("search text")
        await embedder.close_embedder_client()
        return documents, query

    documents, query = asyncio.run(run_calls())

    assert documents == [[0.0, 0.0]]
    assert query == [0.0, 0.0]
    assert len(clients_created) == 1
    assert requests == [
        (
            "http://ollama.test/api/embed",
            {
                "model": "nomic-embed-text",
                "input": ["search_document: paper text"],
                "keep_alive": "30m",
            },
        ),
        (
            "http://ollama.test/api/embed",
            {
                "model": "nomic-embed-text",
                "input": ["search_query: search text"],
                "keep_alive": "30m",
            },
        ),
    ]


def test_embedder_retries_transient_server_error(monkeypatch):
    attempts = 0

    class FakeClient:
        is_closed = False

        async def post(self, _url, *, json):
            nonlocal attempts
            attempts += 1
            request = httpx.Request("POST", "http://ollama.test/api/embed")
            if attempts == 1:
                return httpx.Response(503, request=request)
            return httpx.Response(200, json=_embedding_response(json["input"]), request=request)

    monkeypatch.setattr(settings, "OLLAMA_EMBED_PREFIXES", False)
    monkeypatch.setattr(embedder, "_client", FakeClient())
    monkeypatch.setattr(embedder, "_embedding_semaphore", asyncio.Semaphore(1))

    vectors = asyncio.run(embedder.embed_texts(["paper text"]))

    assert vectors == [[0.0, 0.0]]
    assert attempts == 2


def test_embedder_retries_transient_connection_timeout(monkeypatch):
    attempts = 0
    original_sleep = asyncio.sleep

    class FakeClient:
        is_closed = False

        async def post(self, url, *, json):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectTimeout("temporary network timeout")
            return httpx.Response(
                200,
                json=_embedding_response(json["input"]),
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(settings, "OLLAMA_EMBED_PREFIXES", False)
    monkeypatch.setattr(embedder, "_client", FakeClient())
    monkeypatch.setattr(embedder, "_embedding_semaphore", asyncio.Semaphore(1))

    async def skip_backoff(_seconds):
        await original_sleep(0)

    monkeypatch.setattr(embedder.asyncio, "sleep", skip_backoff)

    vectors = asyncio.run(embedder.embed_texts(["paper text"]))

    assert vectors == [[0.0, 0.0]]
    assert attempts == 2


def test_embedder_serializes_concurrent_batches(monkeypatch):
    active_requests = 0
    max_active_requests = 0

    class FakeClient:
        is_closed = False

        async def post(self, _url, *, json):
            nonlocal active_requests, max_active_requests
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0)
            active_requests -= 1
            return httpx.Response(
                200,
                json=_embedding_response(json["input"]),
                request=httpx.Request("POST", "http://ollama.test/api/embed"),
            )

    monkeypatch.setattr(settings, "OLLAMA_EMBED_PREFIXES", False)
    monkeypatch.setattr(embedder, "_client", FakeClient())
    monkeypatch.setattr(embedder, "_embedding_semaphore", asyncio.Semaphore(1))

    async def run_concurrent_batches():
        return await asyncio.gather(
            embedder.embed_texts(["first"]),
            embedder.embed_texts(["second"]),
        )

    results = asyncio.run(run_concurrent_batches())

    assert results == [[[0.0, 0.0]], [[0.0, 0.0]]]
    assert max_active_requests == 1


def test_embedder_rejects_mismatched_response_count(monkeypatch):
    class FakeClient:
        is_closed = False

        async def post(self, *_args, **_kwargs):
            return httpx.Response(
                200,
                json={"embeddings": []},
                request=httpx.Request("POST", "http://ollama.test/api/embed"),
            )

    monkeypatch.setattr(embedder, "_client", FakeClient())
    monkeypatch.setattr(embedder, "_embedding_semaphore", asyncio.Semaphore(1))

    with pytest.raises(embedder.OllamaEmbeddingError, match="0 embeddings"):
        asyncio.run(embedder.embed_texts(["paper text"]))


def test_embedder_reports_missing_model_without_retries(monkeypatch):
    attempts = 0

    class FakeClient:
        is_closed = False

        async def post(self, *_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                404,
                request=httpx.Request("POST", "http://ollama.test/api/embed"),
            )

    monkeypatch.setattr(embedder, "_client", FakeClient())
    monkeypatch.setattr(embedder, "_embedding_semaphore", asyncio.Semaphore(1))

    with pytest.raises(embedder.OllamaEmbeddingError, match="model is installed"):
        asyncio.run(embedder.embed_texts(["paper text"]))

    assert attempts == 1
