"""Regression tests for shared LLM JSON parsing and judge retry decisions."""
import asyncio

from services.agents.json_utils import coerce_score, parse_llm_json
from services.agents.nodes.citation_verify import citation_verify_node
from services.agents.nodes import citation_verify as citation_verify_module
from services.agents.nodes import final_qa as final_qa_module
from services.agents.nodes import section_critic as section_critic_module
from services.agents.nodes.final_qa import final_qa_node
from services.agents.nodes.section_critic import section_critic_node
from services.agents.pipeline_graph import (
    route_citation_verify,
    route_final_qa,
    route_section_critic,
)
from services.agents.prompts.final_qa import FINAL_QA_SYSTEM
from services.agents.retry import MAX_RETRIES, should_retry
from services.llm_models import ROLE_MODEL_CHAINS
from services.report_store import build_completion_guide_text


def test_parser_handles_fences_nested_braces_and_trailing_text():
    parsed = parse_llm_json(
        '```json\n{"summary":"brace } inside string", "nested":{"score":8}}\n``` done'
    )

    assert parsed == {"summary": "brace } inside string", "nested": {"score": 8}}


def test_parser_returns_none_for_invalid_json():
    assert parse_llm_json("```json\nnot json\n```") is None


def test_score_coercion_accepts_strings_and_percent_coverage():
    assert coerce_score("7.5") == 7.5
    assert coerce_score("85", percentage=True) == 0.85
    assert coerce_score("invalid") is None


def test_retry_decision_uses_explicit_completed_attempts():
    state = {"retry_counts": {"judge": 100}}

    assert should_retry(state, "judge", False, "issue", attempts_done=1) == "retry"
    assert should_retry(state, "judge", False, "issue", attempts_done=MAX_RETRIES) == "retry"
    assert should_retry(state, "judge", False, "issue", attempts_done=MAX_RETRIES + 1) == "flag"


def test_final_qa_flags_failed_judgment_without_retrying(monkeypatch):
    calls = 0

    async def fake_call_llm(**kwargs):
        nonlocal calls
        calls += 1
        return '{"overall_score":"3.5","issues":["insufficient evidence"]}'

    monkeypatch.setattr("services.agents.nodes.final_qa.call_llm", fake_call_llm)
    state = {"draft_text": "Draft text", "retry_counts": {}}
    final_update = None

    final_update = asyncio.run(final_qa_node(state))
    state.update(final_update)

    assert calls == 1
    assert final_update is not None
    assert final_update["flagged_items"]
    assert route_final_qa(state) == "output"


def test_unparseable_final_qa_is_flagged_without_retrying(monkeypatch):
    async def fake_call_llm(**kwargs):
        return "not valid JSON"

    monkeypatch.setattr("services.agents.nodes.final_qa.call_llm", fake_call_llm)
    state = {"draft_text": "Draft text", "retry_counts": {}}

    first = asyncio.run(final_qa_node(state))
    state.update(first)
    assert first["final_qa_result"]["unknown"] is True
    assert first["final_qa_result"]["passed"] is False
    assert first["flagged_items"]
    assert route_final_qa(state) == "output"


def test_final_qa_request_failure_is_flagged_and_pipeline_continues(monkeypatch):
    async def fail_call_llm(**_kwargs):
        raise RuntimeError("Modal request unavailable")

    monkeypatch.setattr("services.agents.nodes.final_qa.call_llm", fail_call_llm)
    result = asyncio.run(
        final_qa_node({
            "draft_text": "The draft.",
            "generated_sections": [{"section_name": "Draft", "text": "The draft."}],
        })
    )

    assert result["final_qa_result"]["unknown"] is True
    assert result["final_qa_result"]["passed"] is False
    assert "Modal request unavailable" in result["flagged_items"][0]["issue"]
    assert route_final_qa(result) == "output"


def test_final_qa_uses_bounded_section_excerpts(monkeypatch):
    captured = {}

    async def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return '{"overall_score":8,"issues":[]}'

    monkeypatch.setattr(final_qa_module.settings, "MODAL_CONTEXT", 8192)
    monkeypatch.setattr(final_qa_module, "call_llm", fake_call_llm)
    result = asyncio.run(
        final_qa_node({
            "draft_text": "The unbounded original draft.",
            "generated_sections": [
                {
                    "section_name": "Introduction",
                    "text": "INTRO_START " + ("x" * 10000) + " INTRO_END",
                },
                {
                    "section_name": "Conclusion",
                    "text": "CONCLUSION_START " + ("y" * 10000) + " CONCLUSION_END",
                },
            ],
        })
    )

    assert "Introduction" in captured["prompt"]
    assert "Conclusion" in captured["prompt"]
    assert "INTRO_START" in captured["prompt"]
    assert "INTRO_END" in captured["prompt"]
    assert "CONCLUSION_START" in captured["prompt"]
    assert "CONCLUSION_END" in captured["prompt"]
    assert "The unbounded original draft." not in captured["prompt"]
    assert len(captured["prompt"]) < 20000
    assert captured["max_tokens"] <= 1500
    assert result["final_qa_result"]["passed"] is True


def test_citation_percentage_and_critic_string_scores_are_normalized(monkeypatch):
    async def fake_citation_llm(**kwargs):
        return '{"coverage_score":"85","unverified_claims":[]}'

    async def fake_critic_llm(**kwargs):
        return '{"overall_score":"8.5","summary":"clear","sections":{}}'

    monkeypatch.setattr("services.agents.nodes.citation_verify.call_llm", fake_citation_llm)
    citation_result = asyncio.run(
        citation_verify_node({"draft_text": "Draft", "citation_registry": []})
    )
    monkeypatch.setattr("services.agents.nodes.section_critic.call_llm", fake_critic_llm)
    critic_result = asyncio.run(
        section_critic_node({"draft_text": "Draft", "outline": {}})
    )

    assert citation_result["citation_verification_result"]["coverage_score"] == 0.85
    assert citation_result["citation_verification_result"]["passed"] is True
    assert critic_result["section_critic_result"]["overall_score"] == 8.5
    assert critic_result["section_critic_result"]["passed"] is True


def test_citation_verifier_splits_long_draft_and_disables_reasoning(monkeypatch):
    requests = []

    async def fake_call_llm(**kwargs):
        requests.append(kwargs)
        return '{"coverage_score":0.95,"unverified_claims":[],"invalid_citations":[]}'

    monkeypatch.setattr(citation_verify_module, "call_llm", fake_call_llm)
    long_section = "Evidence supports this claim. [CR-001]\n\n" * 400
    result = asyncio.run(
        citation_verify_node({
            "draft_text": long_section,
            "citation_registry": [{"id": "CR-001", "title": "Source"}],
            "generated_sections": [
                {"section_name": "Literature Review", "text": long_section},
            ],
        })
    )

    assert len(requests) > 1
    assert all(
        request["max_tokens"] == citation_verify_module.CITATION_VERIFY_CHUNK_MAX_TOKENS
        for request in requests
    )
    assert all(request["reasoning_effort"] == "none" for request in requests)
    assert abs(result["citation_verification_result"]["coverage_score"] - 0.95) < 1e-9
    assert result["citation_verification_result"]["passed"] is True


def test_citation_chunks_run_with_bounded_concurrency(monkeypatch):
    active_requests = 0
    max_active_requests = 0

    async def fake_call_llm(**_kwargs):
        nonlocal active_requests, max_active_requests
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)
        await asyncio.sleep(0.001)
        active_requests -= 1
        return '{"coverage_score":0.95,"unverified_claims":[]}'

    monkeypatch.setattr(citation_verify_module.settings, "MODAL_MAX_CONCURRENT", 2)
    monkeypatch.setattr(citation_verify_module, "call_llm", fake_call_llm)
    sentence = "Evidence supports this claim. "
    section_text = sentence * (
        citation_verify_module.CITATION_VERIFY_CHUNK_CHARS * 4 // len(sentence) + 1
    )
    result = asyncio.run(
        citation_verify_node({
            "draft_text": section_text,
            "citation_registry": [],
        })
    )

    assert max_active_requests == 2
    assert result["citation_verification_result"]["passed"] is True


def test_section_critic_splits_draft_and_disables_reasoning(monkeypatch):
    requests = []

    async def fake_call_llm(**kwargs):
        requests.append(kwargs)
        return '{"overall_score":8,"issues":[],"suggestions":[],"summary":"Clear."}'

    monkeypatch.setattr("services.agents.nodes.section_critic.call_llm", fake_call_llm)
    section_text = "A clear academic statement supports the argument.\n\n" * 250
    result = asyncio.run(
        section_critic_node({
            "draft_text": section_text,
            "outline": {
                "sections": [{
                    "name": "Literature Review",
                    "subsections": ["Themes", "Research Gap"],
                    "guidance": "Compare findings by theme.",
                }],
            },
            "generated_sections": [{
                "section_name": "Literature Review",
                "text": section_text,
            }],
        })
    )

    assert len(requests) > 1
    assert all(
        request["max_tokens"] == section_critic_module.SECTION_CRITIC_CHUNK_MAX_TOKENS
        for request in requests
    )
    assert all(request["reasoning_effort"] == "none" for request in requests)
    report = result["section_critic_result"]
    assert abs(report["overall_score"] - 8) < 1e-9
    assert report["passed"] is True
    assert "Literature Review" in report["sections"]


def test_section_critic_request_failure_is_flagged_without_crashing(monkeypatch):
    async def fail_call_llm(**_kwargs):
        raise RuntimeError("Modal exhausted the output token limit")

    monkeypatch.setattr("services.agents.nodes.section_critic.call_llm", fail_call_llm)
    result = asyncio.run(
        section_critic_node({
            "draft_text": "Draft section content.",
            "outline": {},
            "retry_counts": {"section_critic": 0},
        })
    )

    report = result["section_critic_result"]
    assert report["unknown"] is True
    assert report["passed"] is False
    assert report["failed_excerpts"]
    assert result["flagged_items"][0]["node"] == "section_critic"
    assert route_section_critic(result) == "merge_cd"


def test_citation_verification_request_failure_is_flagged_without_failing_workflow(monkeypatch):
    async def fail_call_llm(**_kwargs):
        raise RuntimeError("Modal exhausted the output token limit")

    monkeypatch.setattr(citation_verify_module, "call_llm", fail_call_llm)
    result = asyncio.run(
        citation_verify_node({
            "draft_text": "A factual claim [CR-099].",
            "citation_registry": [],
            "retry_counts": {"citation_verification": 0},
        })
    )

    report = result["citation_verification_result"]
    assert report["unknown"] is True
    assert report["passed"] is False
    assert report["invalid_citations"][0]["claim"] == "[CR-099]"
    assert result["flagged_items"][0]["node"] == "citation_verification"
    assert route_citation_verify(result) == "merge_cd"


def test_completion_guide_displays_unknown_citation_score_without_crashing():
    guide = build_completion_guide_text({
        "citation_verification_result": {"coverage_score": None, "unknown": True},
    })

    assert "Citation Grounding:** unknown coverage" in guide


def test_final_qa_does_not_judge_references_before_they_exist():
    assert "Reference Completeness" not in FINAL_QA_SYSTEM
    assert ROLE_MODEL_CHAINS["citation_verification"] == ["modal"]
    assert ROLE_MODEL_CHAINS["draft"] == ["modal"]