import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from database import chroma_client
from routers import papers as papers_router
from scripts import reindex_papers


def test_relative_chroma_path_is_anchored_to_backend_directory(monkeypatch):
    monkeypatch.setattr(chroma_client.settings, "CHROMA_PERSIST_PATH", "./chroma_db")

    expected = Path(chroma_client.__file__).resolve().parents[1] / "chroma_db"
    assert Path(chroma_client._persist_path()) == expected.resolve()


def test_index_pdf_uses_fallback_chunker_and_requires_stored_chunks(monkeypatch):
    async def fake_extract_text(_):
        return "Extracted paper text."

    async def fake_page_count(_):
        return 2

    def fake_extract_tables(_):
        return []

    async def fake_store_chunks(**kwargs):
        assert kwargs["chunks"] == ["Extracted paper text."]
        return 1

    monkeypatch.setattr(papers_router, "extract_text_from_pdf", fake_extract_text)
    monkeypatch.setattr(papers_router, "get_pdf_page_count", fake_page_count)
    monkeypatch.setattr(papers_router, "parse_pdf", lambda _: [])
    monkeypatch.setattr(papers_router, "extract_tables", fake_extract_tables)
    monkeypatch.setattr(papers_router, "chunk_document", lambda *args: [])
    monkeypatch.setattr(papers_router, "chunk_text", lambda text: [text])
    monkeypatch.setattr(papers_router, "store_chunks", fake_store_chunks)

    assert asyncio.run(
        papers_router._index_pdf("user-1", "paper-1", "Paper", "Library", b"%PDF")
    ) == (2, 1)

    async def store_nothing(**kwargs):
        return 0

    monkeypatch.setattr(papers_router, "store_chunks", store_nothing)
    with pytest.raises(ValueError, match="No searchable text chunks"):
        asyncio.run(papers_router._index_pdf("user-1", "paper-1", "Paper", "Library", b"%PDF"))


def test_list_papers_includes_live_chroma_vector_counts(monkeypatch):
    rows = [
        {"id": "paper-with-vectors", "status": "indexed", "chunks": 3},
        {"id": "paper-without-vectors", "status": "indexed", "chunks": 0},
    ]

    class Query:
        def select(self, *_):
            return self

        def eq(self, *_):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def execute(self):
            return type("Response", (), {"data": rows})()

    class FakeSupabase:
        def table(self, _):
            return Query()

    class FakeCollection:
        def count(self):
            return 3

        def get(self, **kwargs):
            assert kwargs["where"] == {"paper_id": {"$in": [row["id"] for row in rows]}}
            return {
                "metadatas": [
                    {"paper_id": "paper-with-vectors"},
                    {"paper_id": "paper-with-vectors"},
                    {"paper_id": "paper-with-vectors"},
                ]
            }

    monkeypatch.setattr(papers_router, "get_supabase", lambda: FakeSupabase())
    monkeypatch.setattr(papers_router, "get_user_collection", lambda _: FakeCollection())
    app = FastAPI()
    app.include_router(papers_router.router)

    response = TestClient(app).get("/papers/?user_id=user-1")

    assert response.status_code == 200
    assert [paper["vector_chunks"] for paper in response.json()["papers"]] == [3, 0]


def test_reindex_marks_recovered_paper_indexed(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF")
    updates = []

    class FakeCollection:
        def get(self, **kwargs):
            return {"ids": []}

    class UpdateQuery:
        def update(self, payload):
            updates.append(payload)
            return self

        def eq(self, *_):
            return self

        def execute(self):
            return None

    class FakeSupabase:
        def table(self, _):
            return UpdateQuery()

    async def fake_extract_text(_):
        return "Extracted text."

    async def fake_store_chunks(**kwargs):
        return len(kwargs["chunks"])

    monkeypatch.setattr(reindex_papers, "extract_text_from_pdf", fake_extract_text)
    monkeypatch.setattr(
        reindex_papers,
        "extract_pdf_metadata",
        lambda _: {
            "title": "",
            "authors": "",
            "year": None,
            "abstract": "",
            "doi": None,
            "keywords": [],
            "subject": "",
            "pdf_metadata": {},
        },
    )
    monkeypatch.setattr(reindex_papers, "parse_pdf", lambda _: [])
    monkeypatch.setattr(reindex_papers, "extract_tables", lambda _: [])
    monkeypatch.setattr(reindex_papers, "chunk_document", lambda *args: [])
    monkeypatch.setattr(reindex_papers, "chunk_text", lambda _: ["Extracted text."])
    monkeypatch.setattr(reindex_papers, "get_user_collection", lambda _: FakeCollection())
    monkeypatch.setattr(reindex_papers, "store_chunks", fake_store_chunks)
    monkeypatch.setattr(reindex_papers, "get_supabase", lambda: FakeSupabase())

    result = asyncio.run(reindex_papers.reindex_paper(
        {
            "id": "paper-1",
            "user_id": "user-1",
            "title": "Paper",
            "storage_path": str(pdf_path),
        },
        dry_run=False,
    ))

    assert result == "replaced 0 chunks with 1"
    assert updates == [{
        "status": "indexed",
        "chunks": 1,
        "title": "Paper",
        "authors": "",
        "year": None,
        "abstract": "",
        "doi": None,
        "keywords": [],
        "subject": "",
        "pdf_metadata": {},
    }]


def test_retry_index_endpoint_reuses_saved_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF")
    paper = {
        "id": "paper-1",
        "user_id": "user-1",
        "title": "Paper",
        "collection": "Library",
        "status": "failed",
        "storage_path": str(pdf_path),
    }
    updates = []

    class Query:
        update_data = None

        def select(self, *_):
            return self

        def update(self, payload):
            self.update_data = payload
            return self

        def eq(self, *_):
            return self

        def execute(self):
            if self.update_data is not None:
                updates.append(self.update_data)
                return type("Response", (), {"data": [{"id": "paper-1"}]})()
            return type("Response", (), {"data": [paper]})()

    class FakeSupabase:
        def table(self, _):
            return Query()

    async def fake_index_pdf(user_id, paper_id, title, collection, file_bytes):
        assert (user_id, paper_id, title, collection, file_bytes) == (
            "user-1", "paper-1", "Paper", "Library", b"%PDF",
        )
        return 2, 4

    monkeypatch.setattr(papers_router, "get_supabase", lambda: FakeSupabase())
    monkeypatch.setattr(papers_router, "_index_pdf", fake_index_pdf)
    monkeypatch.setattr(
        papers_router,
        "extract_pdf_metadata",
        lambda _: {
            "title": "",
            "authors": "Jane Doe",
            "year": 2024,
            "abstract": "Abstract text.",
            "doi": "10.1234/example",
            "keywords": ["research"],
            "subject": "Research",
            "pdf_metadata": {"creator": "Test"},
        },
    )

    app = FastAPI()
    app.include_router(papers_router.router)
    response = TestClient(app).post("/papers/paper-1/reindex?user_id=user-1")

    assert response.status_code == 200
    assert response.json() == {
        "paper_id": "paper-1",
        "status": "indexed",
        "pages": 2,
        "chunks": 4,
    }
    assert updates == [
        {"status": "processing"},
        {"status": "indexed", "chunks": 4, "pages": 2, "title": "Paper", "authors": "Jane Doe", "year": 2024, "abstract": "Abstract text.", "doi": "10.1234/example", "keywords": ["research"], "subject": "Research", "pdf_metadata": {"creator": "Test"}},
    ]


def test_upload_stores_extracted_pdf_metadata(monkeypatch):
    inserted = []
    updated = []

    class Query:
        operation = "select"
        payload = None

        def select(self, *_):
            self.operation = "select"
            return self

        def insert(self, payload):
            self.operation = "insert"
            self.payload = payload
            return self

        def update(self, payload):
            self.operation = "update"
            self.payload = payload
            return self

        def eq(self, *_):
            return self

        def execute(self):
            if self.operation == "insert":
                inserted.append(self.payload)
            elif self.operation == "update":
                updated.append(self.payload)
            return type("Response", (), {"data": []})()

    class FakeSupabase:
        def table(self, _):
            return Query()

    metadata = {
        "title": "Extracted Research Title",
        "authors": "Jane Doe and John Smith",
        "year": 2024,
        "abstract": "A concise paper abstract.",
        "doi": "10.1234/example.paper",
        "keywords": ["metadata", "indexing"],
        "subject": "PDF indexing",
        "pdf_metadata": {"creator": "Test Publisher"},
    }

    async def fake_index_pdf(*_args):
        return 2, 5

    monkeypatch.setattr(papers_router, "get_supabase", lambda: FakeSupabase())
    monkeypatch.setattr(papers_router, "extract_pdf_metadata", lambda _: metadata)
    monkeypatch.setattr(papers_router, "_save_pdf_file", lambda *_args: "/tmp/test-paper.pdf")
    monkeypatch.setattr(papers_router, "_index_pdf", fake_index_pdf)
    monkeypatch.setattr(papers_router.settings, "DOC_STRUCTURE_ON_UPLOAD", False)

    app = FastAPI()
    app.include_router(papers_router.router)
    response = TestClient(app).post(
        "/papers/upload",
        data={"user_id": "user-1", "collection": "Library", "title": "upload-name"},
        files={"file": ("paper.pdf", b"%PDF test", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Extracted Research Title"
    assert inserted[0]["title"] == "Extracted Research Title"
    assert inserted[0]["authors"] == "Jane Doe and John Smith"
    assert inserted[0]["year"] == 2024
    assert inserted[0]["abstract"] == "A concise paper abstract."
    assert inserted[0]["doi"] == "10.1234/example.paper"
    assert inserted[0]["keywords"] == ["metadata", "indexing"]
    assert inserted[0]["subject"] == "PDF indexing"
    assert inserted[0]["pdf_metadata"] == {"creator": "Test Publisher"}
    assert updated[0]["status"] == "indexed"
