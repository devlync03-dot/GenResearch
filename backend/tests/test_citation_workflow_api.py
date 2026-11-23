from fastapi import FastAPI
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from routers import agent_tasks


def test_citation_audit_accepts_up_to_500000_characters():
    request = agent_tasks.CitationAuditRequest(user_id="user-1", text="x" * 500_000)
    assert len(request.text) == 500_000

    with pytest.raises(ValidationError):
        agent_tasks.CitationAuditRequest(user_id="user-1", text="x" * 500_001)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(agent_tasks.router)
    return TestClient(app)


def test_document_audit_accepts_utf8_text_upload(monkeypatch):
    async def audit_document(user_id, text, paper_ids):
        return {
            "claims": [{"claim": text, "status": "evidence_required"}],
            "paper_ids": paper_ids,
            "user_id": user_id,
        }

    monkeypatch.setattr(agent_tasks, "_audit_document", audit_document)
    response = _client().post(
        "/agents/citations/audit-document",
        data={"user_id": "user-1", "paper_ids_json": '["paper-1"]'},
        files={"file": ("draft.txt", b"An empirical claim that needs citation.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["document_name"] == "draft.txt"
    assert response.json()["claims"][0]["status"] == "evidence_required"
    assert response.json()["paper_ids"] == ["paper-1"]


def test_document_audit_rejects_unsupported_upload_type():
    response = _client().post(
        "/agents/citations/audit-document",
        data={"user_id": "user-1"},
        files={"file": ("draft.rtf", b"not supported", "application/rtf")},
    )

    assert response.status_code == 415
    assert "PDF, DOCX, or TXT" in response.json()["detail"]


def test_find_evidence_endpoint_returns_only_requested_claim_results(monkeypatch):
    async def find_evidence(user_id, paper_ids, claim):
        return {"claim": claim, "candidates": [{"id": "source-1"}]}

    monkeypatch.setattr(agent_tasks, "find_evidence_for_claim", find_evidence)
    response = _client().post(
        "/agents/citations/find-evidence",
        json={
            "user_id": "user-1",
            "paper_ids": ["paper-1"],
            "claim_id": "claim-1",
            "claim": "This selected claim needs evidence.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "claim": "This selected claim needs evidence.",
        "candidates": [{"id": "source-1"}],
        "claim_id": "claim-1",
    }
