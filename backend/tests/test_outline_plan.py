"""Coverage for bounded outline generation and its workflow-safe fallback."""
import asyncio
import json

from services.agents.nodes import outline_plan as outline_module
from services.agents.prompts.outline import OUTLINE_SECTION_GROUPS


def test_outline_is_generated_in_bounded_requests(monkeypatch):
    requests = []

    async def fake_call_llm(**kwargs):
        requests.append(kwargs)
        if "Component: outline metadata" in kwargs["prompt"]:
            return json.dumps({
                "title": "Generative AI and Research",
                "abstract_required": True,
                "keywords": ["generative AI", "research"],
                "estimated_word_count": 6000,
                "key_themes": ["adoption"],
                "research_questions": ["RQ1: How is AI used?"],
                "citation_style": "apa",
            })
        section_group = OUTLINE_SECTION_GROUPS[len(requests) - 2]
        return json.dumps({
            "sections": [
                {
                    "name": name,
                    "subsections": ["Overview"],
                    "needs": "Relevant evidence",
                    "guidance": "Write a concise evidence-based section.",
                    "word_count_target": 500,
                }
                for name in section_group
            ]
        })

    monkeypatch.setattr(outline_module, "call_llm", fake_call_llm)
    result = asyncio.run(outline_module.outline_plan_node({
        "topic": "Generative AI and research",
        "questionnaire_answers": {"scope": "A concise evaluation."},
    }))

    outline = result["outline"]
    assert len(requests) == 1 + len(OUTLINE_SECTION_GROUPS)
    assert all(request["reasoning_effort"] == "none" for request in requests)
    assert all(
        request["max_tokens"] <= outline_module.OUTLINE_SECTIONS_MAX_TOKENS
        for request in requests
    )
    assert {
        section["name"]
        for section in outline["sections"]
    } == {
        name
        for group in OUTLINE_SECTION_GROUPS
        for name in group
    }
    assert not outline.get("outline_generation_degraded")


def test_outline_model_failure_uses_fallback_instead_of_failing_workflow(monkeypatch):
    async def fail_call_llm(**_kwargs):
        raise RuntimeError("Modal exhausted the output token limit")

    monkeypatch.setattr(outline_module, "call_llm", fail_call_llm)
    result = asyncio.run(outline_module.outline_plan_node({
        "topic": "Generative AI and research",
        "questionnaire_answers": {},
    }))

    assert result["outline"]["outline_generation_degraded"] is True
    assert result["outline"]["sections"]
    assert result["steps_log"][0].startswith("⚠ Outline request failed")
