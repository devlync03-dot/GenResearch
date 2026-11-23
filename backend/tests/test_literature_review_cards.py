import asyncio

from services.literature_review_cards import (
    build_comparison_matrix,
    build_paper_cards,
    chunk_budget_per_paper,
    clean_analysis,
    clean_card,
    quote_is_supported,
    section_word_guide,
    target_words,
    usable_findings,
    word_limits,
    writing_problems,
)
from services.agents.literature_review_agent import (
    _build_context,
    _review_context_char_budget,
    _review_draft_max_tokens,
)
from services import literature_review_cards


def test_quote_check_requires_exact_evidence_fragments():
    evidence = "The randomized trial found a modest improvement in test scores."

    assert quote_is_supported("The randomized trial found a modest improvement.", evidence)
    assert not quote_is_supported("The randomized trial proved a large improvement.", evidence)
    assert not quote_is_supported("test scores", evidence)


def test_clean_card_uses_only_quote_verified_findings_in_comparison():
    source = {"id": "CR-1234567890AB", "title": "Study", "authors": "A. Author", "year": 2024}
    chunks = [{"page": 4, "text": "The study found a modest improvement in test scores."}]
    parsed = {
        "paper_type": "empirical",
        "method": "Randomized trial",
        "findings": [
            {
                "text": "Scores improved.",
                "quote": "The study found a modest improvement in test scores.",
                "page": 4,
                "attribution": "own_finding",
                "strength": "causal",
                "direction": "positive",
            },
            {
                "text": "A large revenue effect.",
                "quote": "The study proved a large increase in revenue.",
                "page": 4,
            },
        ],
    }

    card = clean_card(parsed, source, chunks)

    assert card["quote_verification"] == {"verified": 1, "total": 2}
    assert card["ok"] is True
    assert [finding["text"] for finding in usable_findings(card)] == ["Scores improved."]
    row = build_comparison_matrix([card])[0]
    assert row["key_findings"] == ["Scores improved."]


def test_invalid_quote_does_not_make_card_usable():
    source = {"id": "CR-1234567890AB", "title": "Study"}
    card = clean_card(
        {"findings": [{"text": "Claim.", "quote": "Unsupported exact claim."}]},
        source,
        [{"text": "Different source text."}],
    )

    assert card["ok"] is False
    assert usable_findings(card) == []


def test_build_paper_cards_passes_progress_callback_to_llm(monkeypatch):
    source = {
        "id": "CR-1234567890AB",
        "title": "Study",
        "authors": "A. Author",
        "year": 2024,
    }
    chunks = [{
        "citation_id": source["id"],
        "page": 4,
        "text": "The study found a modest improvement in test scores.",
    }]
    progress_messages = []

    async def report_progress(message):
        progress_messages.append(message)

    async def fake_call_llm(**kwargs):
        assert kwargs["progress_callback"] is report_progress
        return (
            '{"paper_type":"empirical","findings":[{"text":"Scores improved.",'
            '"quote":"The study found a modest improvement in test scores.",'
            '"page":4,"attribution":"own_finding","strength":"causal",'
            '"direction":"positive"}]}'
        )

    monkeypatch.setattr(literature_review_cards, "call_llm", fake_call_llm)
    cards = asyncio.run(
        build_paper_cards(
            [source],
            chunks,
            progress_callback=report_progress,
        )
    )

    assert cards[0]["ok"] is True
    assert cards[0]["quote_verification"] == {"verified": 1, "total": 1}
    assert any("Extracting evidence card" in message for message in progress_messages)
    assert any("Evidence card" in message and "ready" in message for message in progress_messages)


def test_paper_count_scales_context_and_review_length_targets():
    assert chunk_budget_per_paper(1) > chunk_budget_per_paper(20)
    assert target_words(1) == 700
    assert target_words(4) == 820
    assert target_words(20) == 1600
    assert word_limits(4) == (492, 1230)
    assert set(section_word_guide(4)) == {
        "Introduction", "Thematic Analysis", "Critical Synthesis", "Research Gaps", "Conclusion",
    }


def test_literature_review_budget_reserves_space_for_instructions_and_output(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "MODAL_CONTEXT", 16_384)

    budget = _review_context_char_budget(max_output_tokens=2_600)

    assert budget == (16_384 - 2_600 - 2_000) * 2


def test_literature_review_draft_budget_scales_with_target_and_provider_cap(monkeypatch):
    from config import settings

    assert _review_draft_max_tokens(1) == 3_200
    assert _review_draft_max_tokens(20) == 4_800

    monkeypatch.setattr(settings, "MODAL_MAX_TOKENS", 4_000)
    assert _review_draft_max_tokens(20) == 4_000


def test_context_budget_is_enforced_while_retaining_each_source_header():
    registry = [
        {"id": f"CR-{index:012d}", "title": f"Paper {index}", "authors": "", "year": None}
        for index in range(1, 4)
    ]
    chunks = [
        {"citation_id": source["id"], "page": 2, "text": f"evidence-{index} " * 100}
        for index, source in enumerate(registry, start=1)
    ]

    context = _build_context(registry, chunks, char_budget=420)

    assert len(context) <= 420
    for source in registry:
        assert f"=== SOURCE {source['id']}" in context
        assert f"[{source['id']} | page 2]" in context


def test_analysis_discards_invalid_ids_and_uncomparable_one_paper_differences():
    valid = {"CR-1234567890AB", "CR-ABCDEF123456"}
    result = clean_analysis({
        "cohesion": "low",
        "themes": [{"name": "Theme", "paper_ids": ["CR-1234567890AB", "not-valid"]}],
        "agreements": [{"claim": "Shared result", "paper_ids": ["CR-1234567890AB"]}],
        "differences": [{
            "topic": "Different outcome",
            "paper_ids": ["CR-1234567890AB", "CR-ABCDEF123456", "unknown"],
            "comparable": False,
        }],
        "gaps": [{"text": "No prospective evidence", "paper_ids": ["CR-ABCDEF123456"]}],
    }, valid)

    assert result["themes"][0]["paper_ids"] == ["CR-1234567890AB"]
    assert result["agreements"] == []
    assert result["differences"][0]["paper_ids"] == [
        "CR-1234567890AB", "CR-ABCDEF123456",
    ]
    assert result["differences"][0]["comparable"] is False
    assert result["gaps"][0]["origin"] == "inferred_from_set"


def test_writing_lints_flag_overclaiming_citation_stacks_and_missing_scope():
    text = (
        "## Introduction\nThis review demonstrates a result [CR-1234567890AB]"
        "[CR-ABCDEF123456][CR-9876543210AB][CR-111111111111]."
    )
    problems = writing_problems(text, 4)

    assert any("overclaimed" in item for item in problems)
    assert any("three" in item for item in problems)
    assert any("4 supplied papers" in item for item in problems)
