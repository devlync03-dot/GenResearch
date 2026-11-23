from __future__ import annotations

import asyncio

import pytest

from services.agents import citation_agent
from routers import reports
from services.doc_structure.references import (
    ReferenceEntry,
    find_reference_section,
    parse_reference_entries,
    parse_reference_entries_from_pages,
)
from services.agents.citation_agent import _format_reference_entry, _clean_ref_string


def test_find_reference_section_multiline_and_multipage():
    pages = [
        "Title of Paper\nAbstract\nIntroduction...",
        "Methods and Results\nDiscussion...",
        "Conclusion\nReferences\n1. Author A (2020) Paper A. Journal 1\n2. Author B (2021) Paper B. Journal 2",
        "3. Author C (2022) Paper C. Journal 3\n4. Author D (2023) Paper D. Journal 4",
    ]
    start_page, ref_text = find_reference_section(pages)
    assert start_page == 3
    # Check that both page 3 and page 4 are collected
    assert "1. Author A" in ref_text
    assert "4. Author D" in ref_text


def test_reference_heading_variants_and_entries_start_on_next_page():
    pages = [
        "Conclusion\n7. REFERENCES AND BIBLIOGRAPHY",
        "1. Smith, J. (2020). First cited work. Journal One.\n"
        "2. Jones, A. (2021). Second cited work. Journal Two.",
    ]

    bundle = parse_reference_entries_from_pages(pages)

    assert bundle.references_start_page == 1
    assert bundle.reference_count == 2
    assert [entry.ref_number for entry in bundle.entries] == [1, 2]


def test_reference_heading_is_retained_when_entry_format_is_unrecognized():
    bundle = parse_reference_entries_from_pages([
        "Introduction and conclusion text.",
        "References\nA citation format the local parser does not recognize.",
    ])

    assert bundle.references_start_page == 2
    assert bundle.reference_count == 0


def test_reference_sequence_tracking_filters_false_markers():
    text = """
    References
    1. First Author (2018) Intro to AI. AI Journal
    2. Second Author (2019) Deep Learning. DL Journal
    591. Continuing details on line 591
    3. Third Author (2020) Transformers. NLP Journal
    2015. Historical footnote 2015
    4. Fourth Author (2021) Graph Nets. Graph Journal
    """
    bundle = parse_reference_entries(text)
    assert bundle.reference_count == 4
    nums = [e.ref_number for e in bundle.entries]
    assert nums == [1, 2, 3, 4]


def test_flattened_author_year_references_are_split_before_false_numbered_markers():
    text = """References
    1. 3 International Journal of Artificial Intelligence in Education (2025) 35:4057
    Alwasidi, M. A., & Al-Khalifah, K. S. (2025). Assessing writing productivity.
    https://doi.org/10.1000/example.1Bandura, A. (1997). Self-efficacy.
    Barrot, J. S. (2023). Second language writing.
    2. 48550/arXiv.2303.10130 Bastani, H., Bastani, O., Sungu, A., Ge, H.,
    Kabakcı, O., & Mariman, R. (2024). Generative AI can harm learning.
    Cengage Group (2024). Graduate employability report.
    """

    bundle = parse_reference_entries(text)

    assert bundle.reference_count == 5
    assert bundle.entries[0].authors == "Alwasidi, M. A., & Al-Khalifah, K. S"
    assert bundle.entries[1].authors == "Bandura, A"
    assert bundle.entries[2].authors == "Barrot, J. S"
    assert bundle.entries[3].authors.startswith("Bastani, H.")
    assert bundle.entries[4].authors == "Cengage Group"


def test_doi_newline_unwrapping():
    text = """
    References
    1. Christopher MB (1999) Bayesian PCA. Neural Comput. https://doi.org/10.1162/
    089976698300017737
    """
    bundle = parse_reference_entries(text)
    assert len(bundle.entries) == 1
    assert bundle.entries[0].doi == "10.1162/089976698300017737"


def test_citation_agent_formatting_styles():
    entry = ReferenceEntry(
        ref_number=1,
        raw="1. Kaul V, Enslin S, Gross SA (2020) The history of artificial intelligence in medicine. Gastrointest Endosc",
        doi="10.1016/j.gie.2020.04.057",
    )

    apa = _format_reference_entry(entry, 1, "apa")
    ieee = _format_reference_entry(entry, 1, "ieee")

    assert apa.startswith("1. Kaul V")
    assert "https://doi.org/10.1016/j.gie.2020.04.057" in apa
    assert ieee.startswith("[1] Kaul V")
    assert "https://doi.org/10.1016/j.gie.2020.04.057" in ieee


def test_citation_extraction_uses_local_parser_without_search_or_metadata_calls(
    monkeypatch,
    tmp_path,
):
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"fixture")
    entry = ReferenceEntry(
        ref_number=1,
        raw="1. Smith J (2020) Evidence-based title. Journal of Testing",
        authors="Smith J",
        title="Evidence-based title",
        year=2020,
        venue="Journal of Testing",
        parse_confidence=0.8,
    )
    monkeypatch.setattr(
        citation_agent,
        "_load_papers_sync",
        lambda _user_id, _paper_ids: {
            "paper-1": {
                "id": "paper-1",
                "title": "Source paper",
                "storage_path": str(source_pdf),
            }
        },
    )
    monkeypatch.setattr(
        citation_agent,
        "extract_structure",
        lambda _content: type("Structure", (), {"references": [entry]})(),
    )
    monkeypatch.setattr(
        citation_agent,
        "_load_stored_reference_entries_sync",
        lambda _user_id, _paper_id: None,
    )

    async def persist(*_args):
        return None

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Fast citation extraction must not search or verify online.")

    monkeypatch.setattr(citation_agent, "persist_structure", persist)
    monkeypatch.setattr(citation_agent, "_verify_reference", forbidden)
    monkeypatch.setattr(citation_agent, "semantic_search", forbidden)
    monkeypatch.setattr(citation_agent, "call_llm", forbidden)

    result = asyncio.run(
        citation_agent.run_citation_extraction("user-1", ["paper-1"], style="apa")
    )

    reference = result["results"][0]["references"][0]
    assert reference["raw"] == entry.raw
    assert reference["verification_status"] == "not_checked"
    assert "Smith J" in result["results"][0]["citations_text"]
    assert result["results"][0]["chunks_used"] == 0
    assert result["results"][0]["extraction_method"] == "local_pdf_text"
    assert result["results"][0]["metadata_verification"] == "not_performed"
    assert result["results"][0]["bibtex"].startswith("@article{reference1,")


def test_citation_extraction_reuses_currently_stored_references(monkeypatch):
    entry = ReferenceEntry(
        ref_number=1,
        raw="Smith, J. (2020). Stored title. Journal.",
        authors="Smith, J.",
        title="Stored title",
        year=2020,
        venue="Journal",
        parse_confidence=0.9,
    )
    monkeypatch.setattr(
        citation_agent,
        "_load_papers_sync",
        lambda _user_id, _paper_ids: {
            "paper-1": {
                "id": "paper-1",
                "title": "Source paper",
                "storage_path": "/does/not/exist.pdf",
            }
        },
    )
    monkeypatch.setattr(
        citation_agent,
        "_load_stored_reference_entries_sync",
        lambda _user_id, _paper_id: [entry],
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Current stored references should avoid PDF parsing.")

    async def forbidden_async(*_args, **_kwargs):
        raise AssertionError("Current stored references should avoid search and API calls.")

    monkeypatch.setattr(citation_agent, "extract_structure", forbidden)
    monkeypatch.setattr(citation_agent, "semantic_search", forbidden_async)
    monkeypatch.setattr(citation_agent, "_verify_reference", forbidden_async)

    result = asyncio.run(
        citation_agent.run_citation_extraction("user-1", ["paper-1"])
    )

    paper_result = result["results"][0]
    assert paper_result["count"] == 1
    assert paper_result["extraction_method"] == "stored_document_structure"
    assert paper_result["metadata_verification"] == "not_performed"
    assert paper_result["chunks_used"] == 0


def test_citation_extraction_does_not_fall_back_to_vector_chunks(monkeypatch):
    monkeypatch.setattr(
        citation_agent,
        "_load_papers_sync",
        lambda _user_id, _paper_ids: {
            "paper-1": {"id": "paper-1", "title": "Source paper", "storage_path": None}
        },
    )
    monkeypatch.setattr(
        citation_agent,
        "_load_stored_reference_entries_sync",
        lambda _user_id, _paper_id: None,
    )

    async def forbidden_search(**_kwargs):
        raise AssertionError("Local citation extraction must not query vector chunks.")

    monkeypatch.setattr(citation_agent, "semantic_search", forbidden_search)

    result = asyncio.run(
        citation_agent.run_citation_extraction("user-1", ["paper-1"])
    )

    paper_result = result["results"][0]
    assert paper_result["status"] == "source_unavailable"
    assert paper_result["chunks_used"] == 0
    assert "original document file is unavailable" in paper_result["warning"]


def test_citation_metadata_uses_crossref_only_when_source_metadata_matches(monkeypatch):
    entry = ReferenceEntry(
        raw="Doe A (2022) Exact published title. Research Journal",
        authors="Doe A",
        title="Exact published title",
        year=2022,
        doi="10.1234/example",
    )

    async def crossref_metadata(_doi):
        return {
            "title": "Exact published title",
            "authors": ["Alice Doe"],
            "year": 2022,
            "venue": "Research Journal",
            "doi": "10.1234/example",
            "url": "https://doi.org/10.1234/example",
            "citation_count": 12,
            "is_retracted": False,
            "retraction_status": "not_indicated",
        }

    monkeypatch.setattr(citation_agent, "_lookup_crossref_doi", crossref_metadata)
    verified = asyncio.run(citation_agent._verify_reference(entry))

    assert verified["verification_status"] == "verified"
    assert verified["metadata_source"] == "crossref"
    assert verified["authors"] == "Alice Doe"
    assert verified["citation_count"] == 12


def test_claim_audit_rejects_non_verbatim_evidence_quote(monkeypatch):
    async def fake_llm(*_args, **_kwargs):
        return (
            '{"verdict":"supported","source_id":"paper-1",'
            '"evidence_quote":"a fabricated quote","confidence":0.99}'
        )

    monkeypatch.setattr(citation_agent, "call_llm", fake_llm)
    result = asyncio.run(citation_agent._judge_claim(
        "Claim to verify",
        [{
            "id": "paper-1",
            "title": "Source paper",
            "passage": "The original passage from the source.",
        }],
    ))

    assert result["verdict"] == "source_not_found"
    assert result["evidence_excerpt"] == ""
    assert result["confidence"] == 0.0


def test_bibtex_export_uses_available_fields_without_inventing_journal():
    bibtex = citation_agent.format_bibtex([
        {
            "title": "A verified title",
            "authors": "Ada Doe",
            "year": 2024,
            "doi": "10.1234/verified",
        }
    ])

    assert bibtex.startswith("@misc{reference1,")
    assert "title = {A verified title}" in bibtex
    assert "author = {Ada Doe}" in bibtex
    assert "journal =" not in bibtex


def test_report_references_can_be_downloaded_as_bibtex(monkeypatch):
    class Query:
        def select(self, *_args):
            return self

        def eq(self, *_args):
            return self

        def execute(self):
            return type("Response", (), {
                "data": [{
                    "citation_registry": [{
                        "title": "Verified report source",
                        "authors": "Ada Doe",
                        "year": 2024,
                        "doi": "10.1234/verified",
                        "evidence_level": "direct",
                    }]
                }]
            })()

    class Supabase:
        def table(self, _name):
            return Query()

    monkeypatch.setattr(reports, "get_supabase", lambda: Supabase())
    response = asyncio.run(reports.get_references_bibtex("session-1"))

    assert response.media_type == "application/x-bibtex"
    assert response.body.startswith(b"@misc{reference1,")


def test_draft_audit_leaves_uncited_claims_for_user_requested_search(monkeypatch):
    async def extract_claims(_draft):
        return [{"text": "The treatment reduced reported symptoms by 20 percent."}]

    async def forbidden_search(**_kwargs):
        raise AssertionError("Audit must not search for evidence automatically.")

    monkeypatch.setattr(citation_agent, "_extract_claims", extract_claims)
    monkeypatch.setattr(citation_agent, "_load_papers_sync", lambda *_args: {})
    monkeypatch.setattr(citation_agent, "semantic_search", forbidden_search)
    monkeypatch.setattr(citation_agent, "_search_external_candidates", forbidden_search)

    audit = asyncio.run(citation_agent.audit_draft_claims(
        "user-1",
        [],
        "The treatment reduced reported symptoms by 20 percent.",
    ))

    assert audit["claims"][0]["status"] == "evidence_required"
    assert audit["claims"][0]["action"] == "find_evidence"
    assert audit["uncited_claim_count"] == 1
    assert audit["summary"]["total_claims"] == 1
    assert audit["summary"]["uncited_claims"] == 1
    assert audit["summary"]["verified_claims"] == 0
    assert "Manual review is required" in audit["summary"]["assessment"]


def test_empty_claim_audit_explains_that_no_findings_is_not_a_clean_bill_of_health():
    result = citation_agent._empty_audit_result()

    assert result["summary"]["total_claims"] == 0
    assert result["summary"]["status_counts"] == {
        "verified": 0,
        "evidence_required": 0,
        "incorrect_reference": 0,
        "source_mismatch": 0,
        "needs_review": 0,
    }
    assert "does not establish" in result["summary"]["assessment"]


def test_claim_extraction_uses_local_rules_for_cited_and_factual_sentences(monkeypatch):
    async def forbidden_llm(*_args, **_kwargs):
        raise AssertionError("Claim extraction should not call an LLM.")

    monkeypatch.setattr(citation_agent, "call_llm", forbidden_llm)
    draft = (
        "This paper is organized into three sections. "
        "The treatment reduced reported symptoms by 20 percent. "
        "Prior research suggests that feedback improves student performance (Smith et al., 2024). "
        "The authors aim to examine the policy options."
    )

    claims = asyncio.run(citation_agent._extract_claims(draft))

    assert [claim["text"] for claim in claims] == [
        "The treatment reduced reported symptoms by 20 percent.",
        "Prior research suggests that feedback improves student performance (Smith et al., 2024).",
    ]


def test_inline_citation_parser_handles_numbers_author_year_doi_and_footnotes():
    citations = citation_agent._citation_keys(
        "Finding [1, 3-4], supported by (Smith et al., 2021), "
        "Smith et al. (2020) also reported it, DOI 10.1234/example.2, and note [^2]."
    )

    assert sorted(item["value"] for item in citations if item["kind"] == "number") == [1, 2, 3, 4]
    assert any(item["kind"] == "author_year" and "smith" in item["surnames"] for item in citations)
    assert any(item["kind"] == "author_year" and item["year"] == 2020 for item in citations)
    assert any(item["kind"] == "doi" and item["value"] == "10.1234/example.2" for item in citations)


def test_numbered_citation_maps_to_reference_entry_before_resolution():
    entries = [
        ReferenceEntry(ref_number=1, raw="[1] Smith J (2021) First paper.", title="First paper", authors="Smith J", year=2021),
        ReferenceEntry(ref_number=2, raw="[2] Doe A (2022) Second paper.", title="Second paper", authors="Doe A", year=2022),
    ]

    matched = citation_agent._match_citation_references(
        citation_agent._citation_keys("The result was replicated [2]."),
        entries,
    )

    assert matched == [entries[1]]


def test_audit_document_maps_numbered_citations_to_bibliography(monkeypatch):
    async def extract_claims(_draft):
        return [{"text": "The treatment reduced reported symptoms by 20 percent [1]."}]

    async def canonical_metadata(_doi):
        return None

    monkeypatch.setattr(citation_agent, "_extract_claims", extract_claims)
    monkeypatch.setattr(citation_agent, "_load_papers_sync", lambda *_args: {})
    monkeypatch.setattr(citation_agent, "_lookup_crossref_doi", canonical_metadata)
    monkeypatch.setattr(
        citation_agent,
        "_lookup_crossref_title",
        lambda _ref: asyncio.sleep(0, result={
            "title": "Trial findings",
            "authors": ["Alice Smith"],
            "year": 2024,
            "venue": "Research Journal",
            "doi": "10.1234/trial",
            "url": "https://doi.org/10.1234/trial",
            "citation_count": 1,
            "is_retracted": False,
            "retraction_status": "not_indicated",
            "abstract_excerpt": "The intervention reduced the measured symptoms.",
        }),
    )

    async def judge(_claim, candidates):
        return {
            "verdict": "supported",
            "source_id": candidates[0]["id"],
            "evidence_excerpt": "The intervention reduced the measured symptoms.",
            "confidence": 0.9,
            "note": "",
            "citation_context": "background",
        }

    monkeypatch.setattr(citation_agent, "_judge_claim", judge)

    text = (
        "The treatment reduced reported symptoms by 20 percent [1].\n\n"
        "References\n"
        "1. Smith A (2024) Trial findings. Research Journal.\n"
        "2. Doe B (2023) Another study. Journal of Testing."
    )
    audit = asyncio.run(citation_agent.audit_draft_claims("user-1", [], text))

    assert audit["claims"][0]["status"] == "needs_review"
    assert audit["claims"][0]["verdict"] == "partially_supported"
    assert audit["claims"][0]["evidence_excerpt"] == "The intervention reduced the measured symptoms."


def test_full_text_source_mismatch_is_flagged_without_searching_replacements(monkeypatch):
    async def extract_claims(_draft):
        return [{"text": "The treatment reduced reported symptoms by 20 percent [1]."}]

    async def canonical_metadata(_reference):
        return {
            "title": "Trial findings",
            "authors": ["Alice Smith"],
            "year": 2024,
            "venue": "Research Journal",
            "doi": "10.1234/trial",
            "url": "https://doi.org/10.1234/trial",
            "citation_count": 1,
            "is_retracted": False,
            "retraction_status": "not_indicated",
        }

    async def local_search(**kwargs):
        assert kwargs["paper_id"] == "paper-1"
        return [{
            "paper_id": "paper-1",
            "title": "Trial findings",
            "text": "The treatment did not reduce symptoms in the trial.",
            "metadata": {"authors": "Alice Smith", "year": 2024},
        }]

    async def judge(_claim, candidates):
        return {
            "verdict": "contradicted",
            "source_id": candidates[0]["id"],
            "evidence_excerpt": "The treatment did not reduce symptoms in the trial.",
            "confidence": 0.9,
            "note": "",
            "citation_context": "results",
        }

    async def forbidden_search(_claim):
        raise AssertionError("Replacement-source search requires an explicit user request.")

    monkeypatch.setattr(citation_agent, "_extract_claims", extract_claims)
    monkeypatch.setattr(
        citation_agent,
        "_load_papers_sync",
        lambda *_args: {"paper-1": {"id": "paper-1", "title": "Trial findings"}},
    )
    monkeypatch.setattr(citation_agent, "_lookup_crossref_title", canonical_metadata)
    monkeypatch.setattr(citation_agent, "semantic_search", local_search)
    monkeypatch.setattr(citation_agent, "_judge_claim", judge)
    monkeypatch.setattr(citation_agent, "_search_external_candidates", forbidden_search)

    audit = asyncio.run(citation_agent.audit_draft_claims(
        "user-1",
        ["paper-1"],
        "The treatment reduced reported symptoms by 20 percent [1].\n"
        "References\n1. Smith A (2024) Trial findings. Research Journal.",
    ))

    assert audit["claims"][0]["status"] == "source_mismatch"
    assert audit["claims"][0]["evidence_excerpt"] == "The treatment did not reduce symptoms in the trial."


def test_reference_year_mismatch_is_not_verified(monkeypatch):
    async def canonical_metadata(_reference):
        return {
            "title": "A verified paper",
            "authors": ["Alice Smith"],
            "year": 2021,
            "venue": "Research Journal",
            "doi": "10.1234/paper",
            "url": "https://doi.org/10.1234/paper",
            "citation_count": 1,
            "is_retracted": False,
            "retraction_status": "not_indicated",
        }

    monkeypatch.setattr(citation_agent, "_lookup_crossref_title", canonical_metadata)
    result = asyncio.run(citation_agent._verify_reference(ReferenceEntry(
        raw="Smith J (2020) A verified paper. Research Journal.",
        authors="Smith J",
        title="A verified paper",
        year=2020,
    )))

    assert result["verification_status"] == "metadata_mismatch"
    assert result["metadata_matches"] is False


def test_arxiv_reference_flags_matching_published_version(monkeypatch):
    async def arxiv_search(_query, limit=3):
        return {
            "status": "success",
            "results": [{
                "title": "A study of evidence",
                "authors": "Alice Smith",
                "year": 2020,
                "url": "https://arxiv.org/abs/2301.00001",
                "doi": "",
                "abstract_snippet": "Evidence was examined.",
            }],
        }

    async def crossref_search(_query, limit=3):
        return {
            "status": "success",
            "results": [{
                "title": "A study of evidence",
                "authors": "Alice Smith",
                "year": 2022,
                "doi": "10.1234/published",
                "url": "https://doi.org/10.1234/published",
            }],
        }

    citation_agent._source_cache.pop("arxiv:arxiv:2301.00001", None)
    citation_agent._source_cache.pop("arxiv:2301.00001", None)
    monkeypatch.setattr(citation_agent, "search_arxiv", arxiv_search)
    monkeypatch.setattr(citation_agent, "search_crossref", crossref_search)
    result = asyncio.run(citation_agent._verify_reference(ReferenceEntry(
        raw="Smith A (2020) A study of evidence. arXiv:2301.00001",
        authors="Smith A",
        title="A study of evidence",
        year=2020,
        arxiv_id="arXiv:2301.00001",
    )))

    assert result["verification_status"] == "verified"
    assert result["published_version"]["doi"] == "10.1234/published"
    assert result["published_version_status"] == "possible_published_version_found"


def test_find_evidence_caps_abstract_only_support_and_returns_score(monkeypatch):
    async def external_search(_claim):
        return [{
            "id": "EXT-123",
            "title": "Relevant paper",
            "authors": "Ada Doe",
            "year": 2024,
            "doi": "10.1234/relevant",
            "passage": "The treatment reduced reported symptoms.",
            "source_type": "crossref",
        }]

    async def judge(_claim, candidates):
        return {
            "verdict": "supported",
            "source_id": candidates[0]["id"],
            "evidence_excerpt": "The treatment reduced reported symptoms.",
            "confidence": 0.96,
            "note": "",
            "citation_context": "background",
        }

    async def verified(_candidate):
        return {"metadata_verification": "verified", "retraction_status": "not_indicated"}

    monkeypatch.setattr(citation_agent, "_search_external_candidates", external_search)
    monkeypatch.setattr(citation_agent, "_judge_claim", judge)
    monkeypatch.setattr(citation_agent, "_verify_candidate_metadata", verified)
    result = asyncio.run(citation_agent.find_evidence_for_claim(
        "user-1",
        [],
        "The treatment reduced reported symptoms.",
    ))

    assert result["candidates"][0]["support_level"] == "partly_supports"
    assert result["candidates"][0]["verdict"] == "partially_supported"
    assert result["candidates"][0]["evidence_level"] == "abstract"
    assert result["candidates"][0]["support_score"] == 79


def test_write_check_without_citation_routes_to_user_requested_search(monkeypatch):
    async def search(user_id, paper_ids, text):
        return {"claim": text, "candidates": [{"id": "source-1"}]}

    monkeypatch.setattr(citation_agent, "find_evidence_for_claim", search)
    result = asyncio.run(citation_agent.check_text(
        "user-1",
        "A claim with no citation.",
    ))

    assert result["mode"] == "search"
    assert result["candidates"] == [{"id": "source-1"}]


def test_accepted_reference_formatter_uses_metadata_deterministically():
    formatted = citation_agent.format_accepted_references(
        [{
            "title": "A verified paper",
            "authors": "Ada Doe",
            "year": 2024,
            "doi": "10.1234/paper",
            "support_score": 94,
        }],
        "apa",
    )

    assert "A verified paper" in formatted["citations_text"]
    assert "10.1234/paper" in formatted["bibtex"]
    assert "support_score" not in formatted["references"][0]
