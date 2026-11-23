import asyncio

from services.agents import summarization_agent


def test_retrieve_chunks_reads_every_page_and_sorts_by_document_position(monkeypatch):
    rows = [
        ("later", "Second page, first chunk", {"page": 2, "chunk_index": 0}),
        ("first", "First page", {"page": 1, "chunk_index": 0}),
        ("last", "Second page, second chunk", {"page": 2, "chunk_index": 1}),
    ]
    offsets = []

    class FakeCollection:
        def get(self, *, where, limit, offset, include):
            offsets.append(offset)
            page = rows[offset:offset + limit]
            return {
                "ids": [row[0] for row in page],
                "documents": [row[1] for row in page],
                "metadatas": [row[2] for row in page],
            }

    monkeypatch.setattr(summarization_agent, "CHROMA_PAGE_SIZE", 2)
    monkeypatch.setattr(summarization_agent, "get_user_collection", lambda _user_id: FakeCollection())

    chunks = asyncio.run(summarization_agent._retrieve_chunks("user", "paper"))

    assert offsets == [0, 2]
    assert [chunk["chunk_id"] for chunk in chunks] == ["first", "later", "last"]
    assert len(chunks) == 3


def test_contexts_preserve_all_chunk_text_and_stable_excerpt_ids():
    chunks = [
        {"text": "first passage " * 8, "page": 1},
        {"text": "second passage " * 8, "page": 2},
        {"text": "third passage " * 8, "page": 3},
    ]

    contexts = summarization_agent._contexts_within_limit(chunks, max_chars=90)
    combined = "\n".join(contexts)

    assert len(contexts) > 1
    assert "first passage" in combined
    assert "second passage" in combined
    assert "third passage" in combined
    assert "[E1 | page 1]" in combined
    assert "[E2 | page 2]" in combined
    assert "[E3 | page 3]" in combined


def test_sections_keep_all_chunks_together_when_within_context_limit():
    chunks = [
        {"text": f"unique passage {index}", "page": index}
        for index in range(1, 18)
    ]

    sections = summarization_agent._section_contexts(
        [{"title": "Introduction", "chunks": chunks}],
        max_chars=10_000,
    )

    assert len(sections) == 1
    assert sections[0][0] == "Introduction"
    combined = sections[0][1]
    lines = combined.splitlines()
    for index in range(1, 18):
        assert combined.count(f"[E{index} | page {index}]") == 1
        assert lines.count(f"unique passage {index}") == 1


def test_oversized_section_uses_continuations_without_combining_other_sections():
    sections = summarization_agent._section_contexts(
        [
            {
                "title": "Introduction",
                "chunks": [{"text": "intro text " * 20, "page": 1}],
            },
            {"title": "Methods", "chunks": [{"text": "methods text", "page": 2}]},
        ],
        max_chars=80,
    )

    assert [title for title, _ in sections] == [
        "Introduction (continued, part 1/4)",
        "Introduction (continued, part 2/4)",
        "Introduction (continued, part 3/4)",
        "Introduction (continued, part 4/4)",
        "Methods",
    ]
    assert all("methods text" not in context for _, context in sections[:-1])
    assert "methods text" in sections[-1][1]


def test_exclude_reference_chunks_trims_heading_chunk_and_drops_following_pages():
    chunks = [
        {"text": "The paper's conclusion is supported [E1].\n\nReferences\nSmith (2020). Example.", "page": 8},
        {"text": "Jones (2021). Another cited paper.", "page": 9, "section_heading": "References"},
        {"text": "More references.", "page": 10},
    ]

    kept = summarization_agent._exclude_reference_chunks(chunks)

    assert len(kept) == 1
    assert kept[0]["text"] == "The paper's conclusion is supported [E1]."


def test_group_chunks_merges_lettered_subsections_under_major_headings():
    chunks = [
        {"text": "Main section", "page": 1, "section_heading": "I. Data and Method"},
        {"text": "Subsection A", "page": 2, "section_heading": "A. Exposure measure"},
        {"text": "Subsection B", "page": 3, "section_heading": "B. Firm value"},
        {"text": "Main section two", "page": 4, "section_heading": "II. Findings"},
    ]

    groups = summarization_agent._group_chunks_by_section(chunks)

    assert [group["title"] for group in groups] == [
        "I. Data and Method", "II. Findings",
    ]
    assert [chunk["text"] for chunk in groups[0]["chunks"]] == [
        "Main section", "Subsection A", "Subsection B",
    ]
    assert groups[0]["subsections"] == ["A. Exposure measure", "B. Firm value"]


def test_summary_targets_scale_with_paper_size_and_detail():
    assert summarization_agent._target_words_for_detail("quick", 6) == 300
    assert summarization_agent._target_words_for_detail("quick", 76) == 650
    assert summarization_agent._target_words_for_detail("standard", 76) == 800
    assert summarization_agent._target_words_for_detail("deep", 76) == 1200


def test_summary_uses_single_modal_model_for_section_analysis_and_synthesis():
    from services.llm_models import ROLE_MODEL_CHAINS

    assert ROLE_MODEL_CHAINS["summarization_final"] == ["modal"]
    assert ROLE_MODEL_CHAINS["summarization"] == ["modal"]


def test_final_summary_drafts_report_sections_in_separate_requests_and_retries_truncation(monkeypatch):
    requests = []

    async def fake_call_llm(*, prompt, agent_role, system, temperature, max_tokens):
        requests.append((prompt, agent_role, max_tokens))
        if len(requests) == 1:
            return "[The response reached the model output limit and may be incomplete.]"
        requested_section = next(
            title
            for title in summarization_agent._summary_section_titles("empirical")
            if f"Write only the “{title}” section" in prompt
        )
        return f"Content for {requested_section} [E1]."

    monkeypatch.setattr(summarization_agent, "call_llm", fake_call_llm)

    result = asyncio.run(summarization_agent._draft_summary(
        "Paper title",
        {"paper_type": "empirical"},
        ["Section Methods notes: design [E1].", "Section Results notes: result [E2]."],
        800,
        "standard",
        ["Methods", "Results"],
        "Author and year",
        [("Methods", ["[E1]"]), ("Results", ["[E2]"])],
    ))

    expected_titles = summarization_agent._summary_section_titles("empirical")
    assert len(requests) == len(expected_titles) + 1
    assert requests[0][0] == requests[1][0]
    assert requests[0][2] == summarization_agent.SUMMARY_SECTION_INITIAL_MAX_TOKENS
    assert requests[1][2] == summarization_agent.SUMMARY_SECTION_INITIAL_MAX_TOKENS * 2
    assert all(role == "summarization_final" for _, role, _ in requests)
    assert [f"## {title}" in result for title in expected_titles] == [True] * len(expected_titles)
    assert result.index(f"## {expected_titles[0]}") < result.index(f"## {expected_titles[-1]}")
    assert "[The response reached the model output limit and may be incomplete.]" not in result


def test_run_summarization_maps_paper_without_verification_or_metadata_calls(monkeypatch):
    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "page": index,
            "text": f"source {index}",
            "section_heading": ["Introduction", "Methods", "Results"][index - 1],
        }
        for index in range(1, 4)
    ] + [{"chunk_id": "references", "page": 4, "text": "References\nAuthor (2020). Cited title."}]
    mapped_sections = []
    reported_progress = []
    draft_targets = []

    async def retrieve(_user_id, _paper_id):
        return chunks

    async def map_section(context, section_title):
        mapped_sections.append((context, section_title))
        return f"concise notes for {section_title} {context}"

    async def reduce_notes(notes):
        return notes

    async def draft(
        _title,
        _profile,
        notes,
        target_words,
        _detail,
        section_titles,
        _front_matter,
        section_source_ids,
    ):
        draft_targets.append(target_words)
        assert section_titles == ["Introduction", "Methods", "Results"]
        assert [note.splitlines()[0] for note in notes] == [
            "Section Introduction notes:",
            "Section Methods notes:",
            "Section Results notes:",
        ]
        assert section_source_ids == [
            ("Introduction", ["[E1]"]),
            ("Methods", ["[E2]"]),
            ("Results", ["[E3]"]),
        ]
        return "## Findings\n\n**Supported result** [E1]"

    async def report_progress(message):
        reported_progress.append(message)

    monkeypatch.setattr(summarization_agent, "_retrieve_chunks", retrieve)
    monkeypatch.setattr(summarization_agent, "_summarize_section", map_section)
    monkeypatch.setattr(summarization_agent, "_reduce_notes", reduce_notes)
    monkeypatch.setattr(summarization_agent, "_draft_summary", draft)

    result = asyncio.run(summarization_agent.run_summarization(
        "user",
        ["paper"],
        progress_callback=report_progress,
    ))
    summary = result["summaries"][0]

    assert [title for _, title in mapped_sections] == [
        "Introduction",
        "Methods",
        "Results",
    ]
    assert all(f"source {index}" in context for index, (context, _) in enumerate(mapped_sections, start=1))
    assert summary["chunks_used"] == 3
    assert summary["sections_analyzed"] == 3
    assert summary["sources"][0]["text"] == "source 1"
    assert summary["excluded_reference_chunks"] == 1
    assert summary["reproducibility"]["prompt_version"] == summarization_agent.PROMPT_VERSION
    assert draft_targets == [450]
    assert "verification" not in summary
    assert "coverage" not in summary
    assert "**" not in summary["summary"]
    assert "Supported result [E1]" in summary["summary"]
    assert not any("verification" in item.casefold() for item in reported_progress)
    assert any("All paper summaries are ready" in item for item in reported_progress)
    assert any("3 sections" in item for item in reported_progress)
    assert any("summarizing section Introduction" in item for item in reported_progress)
