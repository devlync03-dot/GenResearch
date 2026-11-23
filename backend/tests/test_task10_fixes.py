"""Regression coverage for Task 10 reporting and data-contract fixes."""
import asyncio
import hashlib
import json
import logging

from docx import Document
from fastapi import HTTPException

from routers import agent_tasks, pipeline
from services import source_gathering, tracking
from services.agents.assembly_agent import assemble_docx, human_input_checklist
from services.agents.literature_review_agent import run_literature_review
from services.agents.nodes.scrape_permission import scrape_permission_node
from services.agents.pipeline_graph import route_scrape_permission
from services.agents.proposal_graph import review_proposal
from models.schemas import GeneratedSection, ProposalOutput


def test_completion_guide_handles_malformed_sections_and_scores():
    from services.report_store import build_completion_guide_text

    guide = build_completion_guide_text({
        "topic": "test",
        "sufficiency_report": {
            "overall_assessment": "adequate",
            "sections": {"Introduction": "weak", "Methods": None},
        },
        "citation_verification_result": {"coverage_score": "not-a-score"},
        "section_critic_result": {"overall_score": "unknown"},
        "final_qa_result": {"overall_score": None},
    })

    assert "Introduction" in guide
    assert "Citation Grounding:** unknown coverage" in guide
    assert "Writing Quality:** unknown/10" in guide
    assert "Overall Pipeline QA:** unknown/10" in guide


def test_quality_scores_are_scaled_before_database_storage():
    assert pipeline._scale_qa_score(7.8) == 78
    assert pipeline._scale_qa_score(None) is None
    assert pipeline._scale_qa_score("bad") is None


def test_agent_task_types_map_to_database_enum(monkeypatch):
    stored = []
    monkeypatch.setattr(agent_tasks.tracking, "create_task", lambda *args, **kwargs: "task-1")
    monkeypatch.setattr(agent_tasks.tracking, "complete_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_tasks.tracking, "save_task_result", lambda **kwargs: stored.append(kwargs))
    monkeypatch.setattr(agent_tasks.tracking, "log_agent_event", lambda *args, **kwargs: None)

    async def run_agent():
        return {"summary": "text"}

    asyncio.run(
        agent_tasks._run_tracked_agent(
            user_id="user-1",
            title="summary",
            agent_type="summarization",
            paper_ids=["paper-1"],
            run=run_agent,
        )
    )

    assert stored[0]["type_"] == "summary"


def test_private_citation_audit_does_not_persist_result(monkeypatch):
    stored = []
    monkeypatch.setattr(agent_tasks.tracking, "create_task", lambda *args, **kwargs: "task-1")
    monkeypatch.setattr(agent_tasks.tracking, "complete_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_tasks.tracking, "save_task_result", lambda **kwargs: stored.append(kwargs))
    monkeypatch.setattr(agent_tasks.tracking, "log_agent_event", lambda *args, **kwargs: None)

    async def run_workflow(*args, **kwargs):
        return {"claims": [{"claim": "private document text"}]}

    monkeypatch.setattr(agent_tasks, "run_citation_workflow", run_workflow)
    result = asyncio.run(agent_tasks._audit_document(
        user_id="user-1",
        text="private document text",
        paper_ids=[],
    ))

    assert result["claims"][0]["claim"] == "private document text"
    assert stored == []


def test_summary_without_coverage_gate_completes_tracked_task(monkeypatch):
    task_statuses = []
    monkeypatch.setattr(agent_tasks.tracking, "create_task", lambda *args, **kwargs: "task-1")
    monkeypatch.setattr(agent_tasks.tracking, "complete_task", lambda _task, status: task_statuses.append(status))
    monkeypatch.setattr(agent_tasks.tracking, "save_task_result", lambda **kwargs: None)
    monkeypatch.setattr(agent_tasks.tracking, "log_agent_event", lambda *args, **kwargs: None)

    async def run_agent():
        return {"summaries": [{"status": "completed"}]}

    result = asyncio.run(agent_tasks._run_tracked_agent(
        user_id="user-1",
        title="summary",
        agent_type="summarization",
        paper_ids=["paper-1"],
        run=run_agent,
    ))

    assert result["summaries"][0]["status"] == "completed"
    assert task_statuses == ["completed"]


def test_cancel_endpoint_cancels_only_the_own_users_active_task():
    async def run():
        started = asyncio.Event()

        async def active_work():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(active_work())
        await started.wait()
        agent_tasks._active_tasks["task-1"] = ("user-1", task)

        try:
            try:
                await agent_tasks.cancel_agent_task(
                    agent_tasks.CancelRequest(user_id="user-2", task_id="task-1")
                )
            except HTTPException as error:
                assert error.status_code == 404
            else:
                raise AssertionError("A different user must not be able to cancel the task.")
            assert not task.cancelled()

            response = await agent_tasks.cancel_agent_task(
                agent_tasks.CancelRequest(user_id="user-1", task_id="task-1")
            )
            assert response == {"status": "cancellation_requested", "task_id": "task-1"}
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("The active task should be cancelled.")
        finally:
            agent_tasks._active_tasks.pop("task-1", None)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(run())


def test_summarization_stream_emits_progress_and_final_result(monkeypatch):
    async def run_agent(**kwargs):
        return await kwargs["run"]()

    async def summarize(*, user_id, paper_ids, progress_callback, detail):
        assert detail == "standard"
        await progress_callback("Paper 1/1: loading indexed text.")
        return {"summaries": [{"title": "Test paper", "summary": "Short summary"}]}

    monkeypatch.setattr(agent_tasks, "_run_tracked_agent", run_agent)
    monkeypatch.setattr(agent_tasks, "run_summarization", summarize)

    async def collect():
        response = await agent_tasks.summarize_papers_stream(
            agent_tasks.SummarizationRequest(
                user_id="user-1",
                paper_ids=["paper-1"],
                detail="standard",
            )
        )
        return [line async for line in response.body_iterator]

    events = [json.loads(line) for line in asyncio.run(collect())]

    assert events[0] == {
        "type": "progress",
        "message": "Paper 1/1: loading indexed text.",
    }
    assert events[1]["type"] == "done"
    assert events[1]["result"]["summaries"][0]["summary"] == "Short summary"


def test_literature_review_stream_emits_progress_and_final_result(monkeypatch):
    async def run_agent(**kwargs):
        return await kwargs["run"]()

    async def review(*, user_id, paper_ids, focus_topic, progress_callback):
        assert focus_topic == "test topic"
        await progress_callback("Retrieving evidence.")
        await progress_callback("Drafting review.")
        return {"review": "Completed review", "status": "completed"}

    monkeypatch.setattr(agent_tasks, "_run_tracked_agent", run_agent)
    monkeypatch.setattr(agent_tasks, "run_literature_review", review)

    async def collect():
        response = await agent_tasks.literature_review_stream(
            agent_tasks.LitReviewRequest(
                user_id="user-1",
                paper_ids=["paper-1"],
                focus_topic="test topic",
            )
        )
        return [line async for line in response.body_iterator]

    events = [json.loads(line) for line in asyncio.run(collect())]

    assert events == [
        {"type": "progress", "message": "Retrieving evidence."},
        {"type": "progress", "message": "Drafting review."},
        {"type": "done", "result": {"review": "Completed review", "status": "completed"}},
    ]


def test_tracking_insert_logs_failure_at_warning(monkeypatch, caplog):
    def fail_client():
        raise RuntimeError("database offline")

    monkeypatch.setattr(tracking, "get_supabase", fail_client)
    with caplog.at_level(logging.WARNING):
        tracking._insert("task_results", {"type": "summary"})

    assert "tracking_insert_failed" in caplog.text


def test_scrape_denial_flag_is_emitted_by_node_not_router():
    state = {"scrape_permission_granted": False, "flagged_items": []}
    destinations = route_scrape_permission(state)
    node_result = asyncio.run(scrape_permission_node(state))

    assert destinations == ["outline_plan", "ingestion"]
    assert state["flagged_items"] == []
    assert node_result["flagged_items"][0]["node"] == "scrape_permission"


def test_docx_checklist_and_markdown_rendering_are_case_insensitive_and_clean(tmp_path):
    output = ProposalOutput(
        topic="Test Proposal",
        sections=[GeneratedSection(
            section_name="Methodology",
            text="### Data Collection\n\n**interviews** with participants.",
        )],
    )

    checklist = human_input_checklist(output)
    assert any("interview" in item.lower() for item in checklist)
    path = tmp_path / "proposal.docx"
    assemble_docx(output, path)
    paragraphs = Document(path).paragraphs

    assert any(p.text == "Data Collection" and p.style.name == "Heading 3" for p in paragraphs)
    assert all("**" not in p.text and not p.text.startswith("###") for p in paragraphs)


def test_crossref_falls_back_for_year_cleans_jats_and_keeps_long_abstract():
    abstract = "<jats:p>" + ("A detailed result. " * 100) + "</jats:p>"

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self):
            return {"message": {"items": [{
                "title": ["Crossref result"],
                "issued": {"date-parts": [[2022, 4, 1]]},
                "abstract": abstract,
                "container-title": ["Research Journal"],
                "volume": "5",
                "issue": "2",
                "page": "10-25",
                "publisher": "Academic Publisher",
            }]}}

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    result = asyncio.run(source_gathering.search_crossref("query", session=Session()))
    source = result["results"][0]

    assert source["year"] == 2022
    assert "<jats:p>" not in source["abstract_snippet"]
    assert len(source["abstract_snippet"]) > 300
    assert source["journal"] == "Research Journal"
    assert source["volume"] == "5"
    assert source["issue"] == "2"
    assert source["pages"] == "10-25"


def test_source_gathering_handles_none_titles_without_crashing(monkeypatch):
    async def one_source(query, limit=5, session=None):
        return {"status": "success", "results": [{"title": None, "abstract_snippet": "evidence"}]}

    async def no_sources(query, limit=5, session=None):
        return {"status": "success", "results": []}

    monkeypatch.setattr(source_gathering, "search_semantic_scholar", one_source)
    monkeypatch.setattr(source_gathering, "search_arxiv", no_sources)
    monkeypatch.setattr(source_gathering, "search_crossref", no_sources)
    monkeypatch.setattr(source_gathering, "search_openalex", no_sources)

    found, unfilled, errors = asyncio.run(
        source_gathering.gather_sources_for_gaps(
            [{"topic": "test", "search_query": "test"}], sources_per_gap=1
        )
    )

    assert not found
    assert len(unfilled) == 1
    assert not errors


def test_source_gathering_uses_and_deduplicates_api_results(monkeypatch):
    async def semantic_scholar(query, limit=5, session=None):
        return {"status": "success", "results": [{"title": "Shared paper"}]}

    async def crossref(query, limit=5, session=None):
        return {"status": "success", "results": [
            {"title": "Shared paper"},
            {"title": "Crossref paper"},
        ]}

    async def openalex(query, limit=5, session=None):
        return {"status": "success", "results": [{"title": "OpenAlex paper"}]}

    async def arxiv(query, limit=5, session=None):
        return {"status": "success", "results": [{"title": "arXiv paper"}]}

    monkeypatch.setattr(source_gathering, "search_semantic_scholar", semantic_scholar)
    monkeypatch.setattr(source_gathering, "search_crossref", crossref)
    monkeypatch.setattr(source_gathering, "search_openalex", openalex)
    monkeypatch.setattr(source_gathering, "search_arxiv", arxiv)

    found, unfilled, errors = asyncio.run(
        source_gathering.gather_sources_for_gaps(
            [{"topic": "Crop resilience", "search_query": "Crop resilience"}],
            sources_per_gap=3,
        )
    )

    assert [source["title"] for source in found] == [
        "Shared paper",
        "Crossref paper",
        "OpenAlex paper",
    ]
    assert len({source["title"] for source in found}) == len(found)
    assert not unfilled
    assert not errors


def test_source_gathering_merges_metadata_from_all_search_engines(monkeypatch):
    async def semantic_scholar(query, limit=5, session=None):
        return {
            "status": "success",
            "results": [{
                "title": "Shared academic paper",
                "authors": "A. Researcher",
            }],
        }

    async def crossref(query, limit=5, session=None):
        return {
            "status": "success",
            "results": [{
                "title": "Shared academic paper.",
                "doi": "https://doi.org/10.1234/abc",
                "year": 2024,
                "journal": "Journal of Research",
                "volume": "12",
                "pages": "20-32",
                "abstract_snippet": "Detailed evidence. " * 20,
            }],
        }

    async def openalex(query, limit=5, session=None):
        return {
            "status": "success",
            "results": [{
                "title": "A distinct paper",
                "abstract_snippet": "A shorter abstract.",
            }],
        }

    async def arxiv(query, limit=5, session=None):
        return {"status": "success", "results": []}

    monkeypatch.setattr(source_gathering, "search_semantic_scholar", semantic_scholar)
    monkeypatch.setattr(source_gathering, "search_crossref", crossref)
    monkeypatch.setattr(source_gathering, "search_openalex", openalex)
    monkeypatch.setattr(source_gathering, "search_arxiv", arxiv)

    found, _, _ = asyncio.run(
        source_gathering.gather_sources_for_gaps(
            [{"topic": "Research", "search_query": "research"}],
            sources_per_gap=2,
        )
    )

    assert len(found) == 2
    merged = next(source for source in found if source["title"] == "Shared academic paper")
    assert merged["doi"] == "https://doi.org/10.1234/abc"
    assert merged["journal"] == "Journal of Research"
    assert merged["pages"] == "20-32"
    assert len(merged["abstract_snippet"]) > 300


def test_literature_review_keeps_top_chunks_per_paper(monkeypatch):
    seen_prompt = []
    seen_queries = []
    distances = {"paper-1": 0.0, "paper-2": 0.3, "paper-3": 0.6}

    async def fake_search(user_id, query, top_k, paper_id):
        seen_queries.append(query)
        return [{
            "id": f"{paper_id}-{index}",
            "paper_id": paper_id,
            "title": f"Paper {paper_id}",
            "metadata": {"authors": "Smith, J.", "year": 2024},
                "text": f"Evidence from {paper_id}, {query}, chunk {index}.",
            "distance": distances[paper_id] + index / 1000,
        } for index in range(10)]

    async def fake_llm(*args, **kwargs):
        seen_prompt.append(kwargs["prompt"])
        if "Audit the draft" in kwargs["prompt"]:
            source_id = "CR-" + hashlib.sha256(b"paper-1").hexdigest()[:12].upper()
            return (
                '{"issues": [], "corrected_review": "## Introduction\\n'
                f'Evidence from this paper demonstrates significant clinical benefit [{source_id}]. '
                'Another finding remains entirely unsupported by any literature cited."}'
            )
        if "Extract bibliographic metadata" in kwargs["prompt"]:
            return '{"sources": []}'
        return "Draft review"

    async def fake_records(user_id, paper_ids):
        return {
            paper_id: {
                "id": paper_id,
                "title": f"Paper {paper_id}",
                "authors": "Smith, J.",
                "year": 2024,
                "file_name": f"{paper_id}.pdf",
            }
            for paper_id in paper_ids
        }

    async def fake_cards(registry, chunks, focus_topic, progress_callback=None):
        return [
            {
                "paper_id": source["id"],
                "label": source["title"],
                "paper_type": "empirical",
                "research_question": "How does the intervention affect outcomes?",
                "method": "Observational analysis",
                "data_sample": "Administrative data",
                "key_findings": [],
                "findings": [{
                    "text": f"Finding from {source['title']}.",
                    "quote": f"Evidence from {source['paper_id']}",
                    "quote_verified": True,
                    "attribution": "own_finding",
                    "strength": "correlational",
                    "direction": "positive",
                }],
                "stated_limitations": [],
                "quote_verification": {"verified": 1, "total": 1},
                "ok": True,
            }
            for source in registry
        ]

    async def fake_analysis(cards, focus_topic, progress_callback=None):
        return {
            "cohesion": "low",
            "cohesion_note": "The selected papers address different questions.",
            "themes": [],
            "agreements": [],
            "differences": [],
            "gaps": [],
        }

    progress_messages = []

    async def report_progress(message):
        progress_messages.append(message)

    monkeypatch.setattr("services.agents.literature_review_agent.semantic_search", fake_search)
    monkeypatch.setattr("services.agents.literature_review_agent.call_llm", fake_llm)
    monkeypatch.setattr("services.agents.literature_review_agent._load_paper_records", fake_records)
    monkeypatch.setattr("services.agents.literature_review_agent.build_paper_cards", fake_cards)
    monkeypatch.setattr("services.agents.literature_review_agent.analyze_cards", fake_analysis)
    result = asyncio.run(
        run_literature_review(
            "user-1",
            ["paper-1", "paper-2", "paper-3"],
            progress_callback=report_progress,
        )
    )

    review_prompt = next(prompt for prompt in seen_prompt if "SOURCE KEY" in prompt)
    for paper_id in ("paper-1", "paper-2", "paper-3"):
        assert f"Paper {paper_id}" in review_prompt
    assert len(seen_queries) == 24
    assert any("title authors affiliations abstract" in query for query in seen_queries)
    assert any("methodology" in query for query in seen_queries)
    assert any("limitations" in query for query in seen_queries)
    assert any("constructs variables" in query for query in seen_queries)
    assert any("journal DOI" in query for query in seen_queries)
    assert result["papers_analyzed"] == 3
    assert result["chunks_used"] == 18
    assert "(Smith, 2024)" in result["review"]
    assert "## References" in result["review"]
    assert result["verification"]["completed"] is True
    # [CITATION NEEDED] must NOT appear in the final text — uncited claims go to review_list
    assert "[CITATION NEEDED]" not in result["review"]
    assert result["citation_quality"]["uncited_sentences"] == 1
    assert result["citation_quality"]["coverage"] < 1
    assert result["citation_quality"]["sources_cited"] == 1
    assert len(result["citation_quality"]["sources_uncited"]) == 2
    assert len(result["paper_cards"]) == 3
    assert len(result["comparison_matrix"]) == 3
    assert result["analysis"]["cohesion"] == "low"
    assert result["scope_warning"] == result["analysis"]["cohesion_note"]
    assert "CROSS-PAPER ANALYSIS" in review_prompt
    assert "Cohesion: low" in review_prompt
    assert any("Retrieving evidence for paper" in message for message in progress_messages)
    assert any("quote-grounded evidence cards" in message for message in progress_messages)
    assert any("Comparing study themes" in message for message in progress_messages)
    assert any("Drafting a review" in message for message in progress_messages)
    assert any("Checking the draft" in message for message in progress_messages)
    assert progress_messages[-1].startswith("Literature review ready")


def test_literature_review_citation_ids_are_stable_for_each_paper():
    from services.agents.literature_review_agent import _paper_metadata

    chunk = {"title": "A verifiable title", "metadata": {}}
    first = _paper_metadata([chunk], "paper-id-a")
    second = _paper_metadata([chunk], "paper-id-b")
    assert first["id"] == _paper_metadata([chunk], "paper-id-a")["id"]
    assert first["id"] != second["id"]


def test_literature_review_retrieval_prioritizes_dedicated_first_page_query(monkeypatch):
    from services.agents import literature_review_agent

    queries = []

    async def search(*, user_id, query, top_k, paper_id):
        queries.append(query)
        if query == literature_review_agent.FIRST_PAGE_QUERY:
            return [{"id": "metadata", "page": 1, "text": "Title and author details."}]
        return [
            {"id": "body-1", "page": 3, "text": "Study findings."},
            {"id": "body-2", "page": 4, "text": "Study limitations."},
        ]

    monkeypatch.setattr(literature_review_agent, "semantic_search", search)
    chunks = asyncio.run(
        literature_review_agent._retrieve_paper_chunks(
            "user-1", "paper-1", "focus", max_chunks=3
        )
    )

    assert len(queries) == 1 + len(literature_review_agent.REVIEW_QUERIES)
    assert queries[0] == literature_review_agent.FIRST_PAGE_QUERY
    assert chunks[0]["id"] == "metadata"
    assert {chunk["id"] for chunk in chunks} == {"metadata", "body-1", "body-2"}


def test_proposal_review_keeps_original_when_rewrite_is_too_short(monkeypatch):
    original = "A complete original proposal. " * 50

    async def fake_llm(**kwargs):
        return "A much shorter rewrite."

    monkeypatch.setattr("services.agents.proposal_graph.call_llm", fake_llm)
    result = asyncio.run(review_proposal({"proposal": original}))

    assert result["proposal"] == original