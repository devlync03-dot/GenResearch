"""Chat API contract and safe streaming fallback regressions."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import settings
from routers import chat as chat_router
from services import llm_service, rag_service


@pytest.fixture
def chat_client():
    app = FastAPI()
    app.include_router(chat_router.router)
    return TestClient(app)


def _hit():
    return {
        "id": "chunk-1",
        "text": "A controlled study evaluated this approach.",
        "paper_id": "paper-1",
        "title": "A controlled study",
        "distance": 0.2,
    }


def test_chat_ask_uses_llm_service_and_keeps_response_contract(monkeypatch, chat_client):
    async def fake_search(*args, **kwargs):
        return [_hit()]

    async def fake_call_llm(prompt, *, agent_role, temperature, max_tokens):
        assert agent_role == "chat"
        assert temperature == 0.1
        assert max_tokens == 1400
        return (
            "<summary>The cited study reports the relevant result.</summary>"
            "<answer>The study supports the approach.</answer>"
        )

    monkeypatch.setattr(rag_service, "semantic_search", fake_search)
    monkeypatch.setattr(rag_service, "call_llm", fake_call_llm, raising=False)

    response = chat_client.post(
        "/chat/ask",
        json={"user_id": "user-1", "query": "What did the study show?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "answer": "The study supports the approach.",
        "sources": [{"paper_id": "paper-1", "title": "A controlled study", "relevance": 0.8}],
        "chunks_used": 1,
        "model": settings.MODAL_MODEL,
    }


def test_chat_stream_emits_sources_tokens_and_done(monkeypatch, chat_client):
    async def fake_search(*args, **kwargs):
        return [_hit()]

    async def fake_stream(prompt, *, agent_role, temperature, max_tokens):
        assert agent_role == "chat"
        assert temperature == 0.1
        assert max_tokens == 1400
        yield "The study "
        yield "supports the approach."

    monkeypatch.setattr(rag_service, "semantic_search", fake_search)
    monkeypatch.setattr(rag_service, "call_llm_stream", fake_stream, raising=False)

    response = chat_client.post(
        "/chat/ask-stream",
        json={"user_id": "user-1", "query": "What did the study show?"},
    )
    events = [json.loads(line) for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert [event["type"] for event in events] == ["sources", "token", "token", "done"]
    assert events[0]["sources"][0]["paper_id"] == "paper-1"


def test_chat_stream_without_sources_still_uses_sources_token_done(monkeypatch, chat_client):
    async def fake_search(*args, **kwargs):
        return []

    monkeypatch.setattr(rag_service, "semantic_search", fake_search)
    response = chat_client.post(
        "/chat/ask-stream",
        json={"user_id": "user-1", "query": "Question without documents"},
    )
    events = [json.loads(line) for line in response.iter_lines() if line]

    assert [event["type"] for event in events] == ["sources", "token", "done"]
    assert events[0]["sources"] == []


def test_chat_stream_forwards_output_limit_warning(monkeypatch, chat_client):
    async def fake_search(*args, **kwargs):
        return [_hit()]

    async def fake_stream(*args, **kwargs):
        yield ("token", "Partial answer.")
        yield ("warning", "The model reached its output limit.")

    monkeypatch.setattr(rag_service, "semantic_search", fake_search)
    monkeypatch.setattr(rag_service, "call_llm_stream", fake_stream, raising=False)

    response = chat_client.post(
        "/chat/ask-stream",
        json={"user_id": "user-1", "query": "What did the study show?"},
    )
    events = [json.loads(line) for line in response.iter_lines() if line]

    assert [event["type"] for event in events] == [
        "sources", "token", "warning", "done",
    ]




def test_extract_thinking_and_answer_tagged():
    raw = "<think>Let me analyze the documents. The study mentions meta-analysis.</think> The methodology used was meta-analysis."
    thinking, answer = llm_service.extract_thinking_and_answer(raw)
    assert "Let me analyze the documents" in thinking
    assert answer == "The methodology used was meta-analysis."


def test_extract_thinking_and_answer_untagged_monologue():
    raw = (
        "We are given a user question: 'What methodology was used?' and we have to answer based strictly on the provided document excerpts. "
        "First, I need to scan the document excerpts to see if they mention any specific methodology used in the research.\n\n"
        "Based on the excerpts, the methodology was meta-analysis."
    )
    thinking, answer = llm_service.extract_thinking_and_answer(raw)
    assert "We are given a user question" in thinking
    assert "First, I need to scan" in thinking
    assert answer == "Based on the excerpts, the methodology was meta-analysis."


