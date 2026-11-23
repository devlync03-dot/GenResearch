"""Regression coverage for the deterministic sufficiency source-count floor."""
import asyncio
import json

from services.agents.nodes import sufficiency_eval as sufficiency_module
from services.agents.prompts.sufficiency import build_sufficiency_prompt_chunks
from services.agents.pipeline_graph import route_sufficiency


def _run_with_fake_llm(monkeypatch, fake_response: dict, user_sources: list[dict]):
    async def fake_call_llm(**kwargs):
        return json.dumps(fake_response)

    monkeypatch.setattr(sufficiency_module, "call_llm", fake_call_llm)
    state = {
        "topic": "AI in medical science",
        "questionnaire_answers": {"q1": "detailed answer"},
        "user_provided_sources": user_sources,
    }
    return asyncio.run(sufficiency_module.sufficiency_eval_node(state))


def test_sufficient_verdict_is_overridden_below_minimum(monkeypatch):
    result = _run_with_fake_llm(
        monkeypatch,
        {"overall_assessment": "sufficient", "summary": "Looks fine."},
        [{"title": "One source", "tag": "uploaded"}],
    )
    report = result["sufficiency_report"]
    assert report["overall_assessment"] == "needs_more"
    assert "Overridden" in report["summary"]


def test_sufficient_verdict_is_trusted_at_minimum(monkeypatch):
    result = _run_with_fake_llm(
        monkeypatch,
        {"overall_assessment": "sufficient", "summary": "Well covered."},
        [{"title": f"Paper {letter}", "tag": "uploaded"} for letter in "ABC"],
    )
    report = result["sufficiency_report"]
    assert report["overall_assessment"] == "sufficient"
    assert report["summary"] == "Well covered."


def test_needs_more_verdict_is_never_relaxed(monkeypatch):
    result = _run_with_fake_llm(
        monkeypatch,
        {"overall_assessment": "needs_more", "summary": "Thin coverage."},
        [{"title": f"Paper {index}", "tag": "uploaded"} for index in range(4)],
    )
    report = result["sufficiency_report"]
    assert report["overall_assessment"] == "needs_more"
    assert report["summary"] == "Thin coverage."


def test_unparseable_verdict_uses_fallback_without_retry(monkeypatch):
    async def fake_call_llm(**kwargs):
        return "not valid JSON"

    monkeypatch.setattr(sufficiency_module, "call_llm", fake_call_llm)
    result = asyncio.run(
        sufficiency_module.sufficiency_eval_node({
            "topic": "AI in medical science",
            "user_provided_sources": [],
            "retry_counts": {"sufficiency_eval": 0},
        })
    )

    assert result["sufficiency_report"]["evaluation_unavailable"] is True
    assert result["sufficiency_report"]["overall_assessment"] == "needs_more"
    assert "sufficiency_eval" not in result.get("retry_feedback", {})
    assert result["flagged_items"][0]["node"] == "sufficiency_eval"
    assert route_sufficiency(result) == "scrape_permission"


def test_sufficiency_prompt_chunks_preserve_all_long_answer_text():
    answer = ("Research detail with evidence. " * 500)

    prompts = build_sufficiency_prompt_chunks(
        topic="AI in medical science",
        answers={"background": answer},
        available_sources=[],
        max_chunk_chars=500,
    )

    assert len(prompts) > 1
    recovered_parts = [
        prompt.split("User's Answers:\n", 1)[1].split("\n\nAvailable Sources:", 1)[0]
        for prompt in prompts
    ]
    recovered_answer = "".join(
        part.split(": ", 1)[1].rsplit("\n", 1)[0]
        for part in recovered_parts
    )
    assert recovered_answer == answer


def test_long_sufficiency_material_is_chunked_and_merged(monkeypatch):
    requests = []
    response = {
        "sections": {"background": {"confidence": "adequate", "reasoning": "Supported."}},
        "overall_assessment": "sufficient",
        "recency_adequate": True,
        "counter_arguments_present": True,
        "summary": "The material is sufficient.",
    }

    async def fake_call_llm(**kwargs):
        requests.append(kwargs)
        return json.dumps(response)

    monkeypatch.setattr(sufficiency_module, "call_llm", fake_call_llm)
    result = asyncio.run(
        sufficiency_module.sufficiency_eval_node({
            "topic": "AI in medical science",
            "questionnaire_answers": {"draft": "Long draft text. " * 4000},
            "user_provided_sources": [{"title": f"Paper {index}", "tag": "uploaded"} for index in range(3)],
        })
    )

    chunk_requests = [
        request for request in requests
        if request["system"] == sufficiency_module.SUFFICIENCY_CHUNK_SYSTEM
    ]
    merge_requests = [
        request for request in requests
        if request["system"] == sufficiency_module.SUFFICIENCY_MERGE_SYSTEM
    ]
    assert len(chunk_requests) > sufficiency_module.SUFFICIENCY_REDUCE_GROUP_SIZE
    assert len(merge_requests) == 3
    assert all(request["reasoning_effort"] == "none" for request in requests)
    assert chunk_requests[0]["max_tokens"] == sufficiency_module.SUFFICIENCY_CHUNK_MAX_TOKENS
    assert requests[-1]["system"] == sufficiency_module.SUFFICIENCY_MERGE_SYSTEM
    assert all(
        f"Paper {index}" in "\n".join(request["prompt"] for request in chunk_requests)
        for index in range(3)
    )
    assert result["sufficiency_report"]["overall_assessment"] == "sufficient"
