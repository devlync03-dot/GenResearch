"""Regression tests for batched source-quality evaluation prompts."""
import asyncio
import json

from services.agents.nodes import source_quality_eval
from services.llm_models import ROLE_MODEL_CHAINS


def test_batch_prompt_encodes_source_fields_as_valid_json(monkeypatch):
    prompts = []

    async def fake_call_llm(**kwargs):
        prompts.append(kwargs["prompt"])
        return '{"0":{"passed":true,"credibility":"high","relevance_score":8}}'

    monkeypatch.setattr(source_quality_eval, "call_llm", fake_call_llm)
    sources = [{
        "title": "Farmer's " + '"resilient"' + " crop study",
        "year": 2024,
        "abstract_snippet": "A useful study\\with quoted text.",
        "citation_count": 12,
    }]

    result = asyncio.run(
        source_quality_eval._evaluate_batch(sources, [3], "crop resilience", "drought")
    )

    serialized_sources = prompts[0].split("Sources to evaluate:\n", 1)[1].split(
        "\n\nEvaluate each source.", 1
    )[0]
    assert json.loads(serialized_sources) == {
        "0": {
            "title": sources[0]["title"],
            "year": 2024,
            "abstract": sources[0]["abstract_snippet"],
            "citations": 12,
            "doi": "",
            "venue": "",
            "source": "unknown",
        }
    }
    assert result[3]["passed"] is True


def test_prefilter_rejects_sources_missing_abstract_or_year():
    source = {
        "title": "Crop resilience study",
        "abstract_snippet": "A detailed study on crop resilience and drought response.",
    }

    assert source_quality_eval._pre_filter_source(source, "crop resilience") == (
        False,
        "Missing or invalid publication year",
    )
    source["year"] = 2024
    source["abstract_snippet"] = ""
    assert source_quality_eval._pre_filter_source(source, "crop resilience") == (
        False,
        "No usable abstract",
    )


def test_prefilter_keeps_relevant_foundational_work():
    source = {
        "title": "Foundational crop resilience research",
        "year": 1975,
        "abstract_snippet": "A foundational study of crop resilience and drought response.",
    }

    assert source_quality_eval._pre_filter_source(source, "crop resilience") == (
        True,
        "pre-filter-pass",
    )


def test_source_quality_evaluation_uses_modal():
    assert ROLE_MODEL_CHAINS["source_quality_evaluator"] == ["modal"]


def test_invalid_quality_response_does_not_pass_source(monkeypatch):
    async def invalid_response(**_kwargs):
        return "not valid json"

    monkeypatch.setattr(source_quality_eval, "call_llm", invalid_response)
    result = asyncio.run(
        source_quality_eval._evaluate_batch(
            [{
                "title": "Crop resilience study",
                "year": 2024,
                "abstract_snippet": "A detailed crop resilience study.",
            }],
            [0],
            "crop resilience",
            "drought",
        )
    )

    assert result[0]["passed"] is False
    assert result[0]["evaluation_status"] == "unavailable"


def test_failed_quality_batch_is_excluded_and_flagged(monkeypatch):
    async def unavailable(**_kwargs):
        raise RuntimeError("quality model unavailable")

    monkeypatch.setattr(source_quality_eval, "call_llm", unavailable)
    result = asyncio.run(
        source_quality_eval.source_quality_eval_node({
            "topic": "crop resilience",
            "scraped_sources": [{
                "title": "Crop resilience research",
                "year": 2024,
                "abstract_snippet": "A detailed study of crop resilience and drought response.",
            }],
        })
    )

    assessment = result["source_quality_results"]["Crop resilience research"]
    assert assessment["passed"] is False
    assert assessment["evaluation_status"] == "unavailable"
    assert result["flagged_items"][0]["node"] == "source_quality_eval"