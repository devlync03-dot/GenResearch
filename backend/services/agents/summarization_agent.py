import asyncio
from difflib import SequenceMatcher
import logging
import re
import time
from collections.abc import Awaitable, Callable

from config import settings
from database.chroma_client import get_user_collection
from services.llm_models import get_chain_for_role
from services.llm_service import call_llm


logger = logging.getLogger(__name__)
CHROMA_PAGE_SIZE = 500
MAX_NOTES_CHARS = 14_000
MAX_CONCURRENT_PAPERS = 3
MAX_CONCURRENT_SUMMARIZATION_REQUESTS = 2
SUMMARY_SECTION_INITIAL_MAX_TOKENS = 1_000
SUMMARY_SECTION_MAX_TOKENS = 3_000
PROMPT_VERSION = "summarization-v8"
_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(?:references|bibliography|works cited|literature cited)"
    r"(?:\s*\(continued\))?[ \t]*:?[ \t]*$"
)
BASE_RULES = """You are GenResearch's academic paper explainer.
Explain the paper so an interested reader can understand its complete argument
without needing to decode dense academic prose. Use only the paper excerpts and
section notes supplied in the prompt; do not use outside knowledge or invent
facts, causes, implications, or examples.

Explain ideas in plain, precise language while retaining the paper's technical
meaning. Explain specialized terms when the paper defines or uses them. Explain
key mathematical expressions and analytical methods: reproduce equations faithfully,
identify symbols and assumptions only when the excerpts define them, and describe what
the expression does in the paper's method or analysis. Explain the analytics,
study design, data, comparisons, reported results, uncertainty, limitations,
and conclusions in enough detail to make the reasoning understandable.

Ground important claims, numbers, methods, equations, and results with source
labels exactly in the form [E3]. Preserve the authors' qualifications.
If the excerpts do not establish a detail, state that it is not specified in the
provided text rather than filling the gap. Distinguish reported findings from
the authors' interpretation and from limitations.

Write professional, descriptive prose with clear Markdown section headings and
plain bullet lists where useful. Do not use Markdown bold syntax (** or __).
Keep equations readable and explain them without deriving new results. End with
3-5 concise key takeaways."""

SUMMARY_STRUCTURES = {
    "review": """## Paper at a glance
## Review purpose and scope
## Review approach, when described
## Main themes and synthesis
## Gaps and limitations stated by the authors
## Conclusions and key takeaways""",
    "empirical": """## Paper at a glance
## Research question and background
## Study design, data, and measures
## Analysis and results
## Authors' interpretation and implications
## Limitations stated by the authors
## Conclusion and key takeaways""",
    "survey": """## Paper at a glance
## Survey purpose and scope
## Participants, sampling, and measures
## Main findings
## Interpretation and implications stated by the authors
## Limitations stated by the authors
## Conclusion and key takeaways""",
    "framework_position": """## Paper at a glance
## Problem and purpose
## Proposed framework or position
## Framework components and relationships
## Worked examples, if provided
## Recommendations stated by the authors
## Limitations and open questions
## Conclusion and key takeaways""",
    "system_tool": """## Paper at a glance
## Problem and design goals
## Proposed system or tool
## Architecture and operation
## Evaluation and reported results, if provided
## Limitations stated by the authors
## Conclusion and key takeaways""",
    "theory": """## Paper at a glance
## Problem and theoretical context
## Main concepts and assumptions
## Proposed theory, propositions, or derivation
## Implications stated by the authors
## Limitations and open questions
## Conclusion and key takeaways""",
    "unclassified": """## Paper at a glance
## Purpose and central ideas
## Main sections and arguments
## Evidence and examples presented
## Limitations stated by the authors
## Conclusion and key takeaways""",
}

ProgressCallback = Callable[[str], Awaitable[None]]


async def _report_progress(callback: ProgressCallback | None, message: str) -> None:
    logger.info("summarization_progress: %s", message)
    if callback:
        await callback(message)


def _normalize_section_title(title: str) -> str:
    without_number = re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?\s+|[IVXLCDM]+\.\s+)", "", title, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", " ", without_number.casefold()).strip()


def _group_chunks_by_section(
    chunks: list[dict],
    toc_sections: list[dict] | None = None,
    page_count: int | None = None,
) -> list[dict]:
    """Assign each excerpt to the best matching TOC section or a heading fallback."""
    valid_sections = [
        dict(section)
        for section in (toc_sections or [])
        if str(section.get("title") or "").strip()
    ]
    has_toc = bool(valid_sections)
    if not valid_sections:
        headings: list[str] = []
        for chunk in chunks:
            heading = str(chunk.get("section_heading") or "").strip()
            if heading and _normalize_section_title(heading) not in {
                _normalize_section_title(item) for item in headings
            }:
                headings.append(heading)
        valid_sections = [
            {
                "title": heading,
                "coverage_expected": not bool(re.match(r"(?i)^page window\b", heading)),
            }
            for heading in headings
        ]
    if not valid_sections:
        valid_sections = [{"title": "Paper body", "coverage_expected": False}]

    ordered = sorted(valid_sections, key=lambda section: (
        int(section.get("page_start") or 0),
        int(section.get("ordinal") or 0),
    ))
    groups = [
        {
            "title": str(section["title"]),
            "chunks": [],
            "expected": (
                bool(section.get("coverage_expected"))
                if "coverage_expected" in section
                else has_toc and not bool(section.get("is_fallback_window"))
            ),
        }
        for section in ordered
    ]

    for section_index, section in enumerate(ordered):
        start = section.get("page_start")
        end = section.get("page_end")
        if start is not None and (end is None or int(end) <= int(start)):
            later_starts = [
                candidate.get("page_start")
                for candidate in ordered[section_index + 1:]
                if candidate.get("page_start") is not None
                and int(candidate["page_start"]) > int(start)
            ]
            end = min(later_starts) - 1 if later_starts else page_count
        section["_effective_start"] = int(start) if start is not None else None
        section["_effective_end"] = int(end) if end is not None else None
        section["_normalized_title"] = _normalize_section_title(str(section["title"]))

    for chunk_index, chunk in enumerate(chunks):
        chunk["_excerpt_index"] = chunk_index + 1
        heading = str(chunk.get("section_heading") or "")
        normalized_heading = _normalize_section_title(heading)
        matched_index = None
        if normalized_heading:
            matches = [
                index for index, section in enumerate(ordered)
                if normalized_heading == section["_normalized_title"]
                or SequenceMatcher(
                    None, normalized_heading, section["_normalized_title"]
                ).ratio() >= 0.9
            ]
            if matches:
                matched_index = max(
                    matches,
                    key=lambda index: SequenceMatcher(
                        None, normalized_heading, ordered[index]["_normalized_title"]
                    ).ratio(),
                )
        if matched_index is None:
            try:
                page = int(chunk.get("page"))
            except (TypeError, ValueError):
                page = None
            if page is not None:
                page_matches = [
                    index for index, section in enumerate(ordered)
                    if section["_effective_start"] is not None
                    and page >= section["_effective_start"]
                    and (
                        section["_effective_end"] is None
                        or page <= section["_effective_end"]
                    )
                ]
                if page_matches:
                    matched_index = page_matches[-1]
        if matched_index is None:
            matched_index = 0
        groups[matched_index]["chunks"].append(chunk)

    return _merge_subsections(groups)


_MAJOR_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:[IVXLCDM]+\.|\d+\.?)\s+", re.IGNORECASE)
_SUBSECTION_HEADING_RE = re.compile(r"^\s*[A-Z]\.\s+")


def _merge_subsections(groups: list[dict]) -> list[dict]:
    """Combine lettered subsections under the preceding numbered heading."""
    merged: list[dict] = []
    current_major: dict | None = None
    for group in groups:
        title = re.sub(r"\s+", " ", str(group.get("title") or "")).strip()
        group["title"] = title or "Paper section"
        if _MAJOR_NUMBERED_HEADING_RE.match(title):
            current_major = {
                **group,
                "chunks": list(group["chunks"]),
                "subsections": [],
            }
            merged.append(current_major)
        elif current_major and _SUBSECTION_HEADING_RE.match(title):
            current_major["chunks"].extend(group["chunks"])
            current_major["subsections"].append(title)
        else:
            current_major = None
            merged.append(group)
    return merged


def _front_matter(chunks: list[dict]) -> tuple[str, list[str], int | None]:
    first_pages = []
    for chunk in chunks:
        try:
            page = int(chunk.get("page"))
        except (TypeError, ValueError):
            page = 1 if not first_pages else None
        if page is None or page <= 2:
            first_pages.append(str(chunk.get("text") or ""))
    text = "\n\n".join(first_pages)[:8_000]
    authors: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:authors?|by)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if match:
            authors = [
                author.strip()
                for author in re.split(r"\s*;\s*|\s+and\s+|,\s*(?=[A-Z])", match.group(1))
                if author.strip()
            ]
            break
    year_match = re.search(r"\b(?:19|20)\d{2}\b", text)
    year = int(year_match.group()) if year_match else None
    return text, authors, year


def _infer_paper_type(text: str) -> str:
    sample = text[:20_000].casefold()
    if "systematic review" in sample or "literature review" in sample:
        return "review"
    if "framework" in sample and ("layer" in sample or "framework" in sample):
        return "framework_position"
    if "survey" in sample and ("respondents" in sample or "questionnaire" in sample):
        return "survey"
    if "theorem" in sample or "proposition" in sample or "formal proof" in sample:
        return "theory"
    if "architecture" in sample and ("system" in sample or "tool" in sample):
        return "system_tool"
    if any(term in sample for term in ("regression", "sample size", "dataset", "experiment")):
        return "empirical"
    return "unclassified"


def _target_words_for_detail(detail: str, page_count: int | None) -> int:
    is_long_paper = page_count is not None and page_count > 10
    targets = {
        "quick": (300, 650),
        "standard": (450, 800),
        "deep": (700, 1200),
    }
    short_target, long_target = targets.get(detail, targets["quick"])
    return long_target if is_long_paper else short_target


def _summary_section_titles(paper_type: str) -> list[str]:
    structure = SUMMARY_STRUCTURES.get(
        paper_type,
        SUMMARY_STRUCTURES["unclassified"],
    )
    titles = []
    for line in structure.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            title = re.sub(
                r", (?:when described|if provided|if present)$",
                "",
                match.group(1),
                flags=re.IGNORECASE,
            )
            titles.append(title)
    return titles


def _page_sort_key(chunk: dict) -> tuple[int, int]:
    try:
        page = int(chunk.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    try:
        chunk_index = int(chunk.get("chunk_index") or 0)
    except (TypeError, ValueError):
        chunk_index = 0
    return page, chunk_index


def _read_chunk_page(collection, paper_id: str, offset: int) -> dict:
    return collection.get(
        where={"paper_id": paper_id},
        limit=CHROMA_PAGE_SIZE,
        offset=offset,
        include=["documents", "metadatas"],
    )


async def _retrieve_chunks(user_id: str, paper_id: str) -> list[dict]:
    """Read every indexed chunk for a paper from the user's Chroma collection."""
    collection = get_user_collection(user_id)
    chunks: list[dict] = []
    offset = 0
    while True:
        result = await asyncio.to_thread(_read_chunk_page, collection, paper_id, offset)
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for index, chunk_id in enumerate(ids):
            text = documents[index] if index < len(documents) else None
            if not isinstance(text, str) or not text.strip():
                continue
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata = metadata if isinstance(metadata, dict) else {}
            chunks.append({
                **metadata,
                "chunk_id": chunk_id,
                "text": text,
                "paper_id": metadata.get("paper_id", paper_id),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index", index + offset),
                "title": metadata.get("title"),
            })
        offset += len(ids)
        if len(ids) < CHROMA_PAGE_SIZE:
            break
    chunks.sort(key=_page_sort_key)
    return chunks


def _exclude_reference_chunks(chunks: list[dict]) -> list[dict]:
    """Keep the paper body and discard its bibliography and all following chunks."""
    body: list[dict] = []
    for chunk in chunks:
        section_heading = re.sub(r"\s+", " ", str(chunk.get("section_heading") or "")).strip()
        if _REFERENCE_HEADING_RE.fullmatch(section_heading):
            break
        text = str(chunk.get("text") or "")
        heading = _REFERENCE_HEADING_RE.search(text)
        if heading:
            preceding_text = text[:heading.start()].strip()
            if preceding_text:
                body.append({**chunk, "text": preceding_text})
            break
        body.append(chunk)
    return body


def _format_excerpt(index: int, chunk: dict) -> str:
    excerpt_index = chunk.get("_excerpt_index", index)
    page = chunk.get("page")
    page_label = f"page {page}" if page is not None else "page not identified"
    heading = chunk.get("section_heading")
    heading_label = f" | {heading}" if heading else ""
    return f"[E{excerpt_index} | {page_label}{heading_label}]\n{chunk.get('text', '')}"


def _summary_context_char_budget() -> int:
    return max(8_000, min(45_000, (settings.MODAL_CONTEXT - 2_000) * 2))


def _contexts_within_limit(chunks: list[dict], max_chars: int | None = None) -> list[str]:
    """Split excerpts only when needed to keep each context within the model limit."""
    max_chars = max_chars or _summary_context_char_budget()
    contexts: list[str] = []
    current: list[str] = []
    current_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        excerpt = _format_excerpt(index, chunk)
        if current and current_chars + len(excerpt) + 10 > max_chars:
            contexts.append("\n\n---\n\n".join(current))
            current = []
            current_chars = 0
        if len(excerpt) > max_chars:
            text = chunk.get("text", "")
            prefix = _format_excerpt(index, {**chunk, "text": ""})
            available = max(1, max_chars - len(prefix) - 2)
            for start in range(0, len(text), available):
                part = f"{prefix}{text[start:start + available]}"
                if current:
                    contexts.append("\n\n---\n\n".join(current))
                    current = []
                    current_chars = 0
                contexts.append(part)
            continue
        current.append(excerpt)
        current_chars += len(excerpt) + 10
    if current:
        contexts.append("\n\n---\n\n".join(current))
    return contexts


def _section_contexts(
    sections: list[dict],
    max_chars: int | None = None,
) -> list[tuple[str, str]]:
    """Build analysis contexts from paper sections, splitting only oversized sections."""
    section_contexts: list[tuple[str, str]] = []
    for section in sections:
        title = str(section.get("title") or "Paper section")
        contexts = _contexts_within_limit(section.get("chunks") or [], max_chars)
        for part_index, context in enumerate(contexts, start=1):
            part_title = (
                f"{title} (continued, part {part_index}/{len(contexts)})"
                if len(contexts) > 1
                else title
            )
            section_contexts.append((part_title, context))
    return section_contexts


def _plain_markdown(text: str) -> str:
    """Remove bold delimiters without altering mathematical asterisks."""
    return re.sub(
        r"(?<!\*)\*\*(.+?)\*\*(?!\*)",
        lambda match: match.group(1),
        text,
        flags=re.DOTALL,
    ).strip()


async def _summarize_section(context: str, section_title: str) -> str:
    section_instruction = (
        f"Focus only on the paper section titled {section_title!r}. "
        "Preserve its role and how it relates to the paper's argument."
    )
    prompt = f"""Make concise but informative notes for this paper section.
Retain essential content: main
argument, methods, data, mathematical equations and what the authors say they
mean, analytical procedures, quantitative results, examples, implications,
recommendations, and stated limits. Preserve source identifiers exactly (e.g.
[E12]); do not verify claims or add external facts. Keep front-matter author
names and publication dates when present. Use concise bullets.
{section_instruction}

--- SOURCE EXCERPTS ---
{context}
--- END OF EXCERPTS ---"""
    return _plain_markdown(await call_llm(
        prompt=prompt,
        agent_role="summarization",
        system=BASE_RULES,
        temperature=0.0,
        max_tokens=3000,
    ))


async def _reduce_notes(notes: list[str]) -> list[str]:
    """Hierarchically condense section notes while retaining citations."""
    current = notes
    while sum(map(len, current)) > MAX_NOTES_CHARS:
        groups: list[list[str]] = []
        group: list[str] = []
        group_chars = 0
        for note in current:
            if group and group_chars + len(note) + 2 > MAX_NOTES_CHARS:
                groups.append(group)
                group = []
                group_chars = 0
            group.append(note)
            group_chars += len(note) + 2
        if group:
            groups.append(group)

        async def reduce_group(group_notes: list[str]) -> str:
            joined_notes = "\n\n---\n\n".join(group_notes)
            prompt = f"""Synthesize these section notes into concise notes
for a later complete-paper explainer. Preserve every distinct method, equation
and symbol explanation, quantitative result, qualification, limitation, and
source ID. Preserve every section label so the final summary can cover the full
paper. Do not introduce facts. Keep citations beside
the claims they support.
Use plain descriptive prose, no bold syntax, and target about 450 words.

--- SECTION NOTES ---
{joined_notes}
--- END SECTION NOTES ---"""
            return _plain_markdown(await call_llm(
                prompt=prompt,
                agent_role="summarization",
                system=BASE_RULES,
                temperature=0.1,
                max_tokens=2400,
            ))

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUMMARIZATION_REQUESTS)

        async def limited_reduce(group_notes: list[str]) -> str:
            async with semaphore:
                return await reduce_group(group_notes)

        reduced = await asyncio.gather(*(limited_reduce(notes_group) for notes_group in groups))
        if len(reduced) >= len(current):
            break
        current = reduced
    return current


async def _draft_summary(
    title: str,
    profile: dict,
    section_notes: list[str],
    target_words: int,
    detail: str,
    section_titles: list[str],
    front_matter: str,
    section_source_ids: list[tuple[str, list[str]]],
) -> str:
    paper_type = profile.get("paper_type") or "unclassified"
    structure_titles = _summary_section_titles(paper_type)
    notes_context = "\n\n--- SECTION NOTES ---\n\n".join(section_notes)
    source_context = "\n".join(
        f"{section_title}: {', '.join(ids)}"
        for section_title, ids in section_source_ids
        if ids
    )
    max_section_tokens = min(
        SUMMARY_SECTION_MAX_TOKENS,
        settings.MODAL_MAX_TOKENS,
    )
    initial_section_tokens = min(
        SUMMARY_SECTION_INITIAL_MAX_TOKENS,
        max_section_tokens,
    )
    base_rules = BASE_RULES.replace(
        "End with 3-5 concise key takeaways.",
        "Return only the requested report section.",
    )
    notes_prefix = f"""Paper title: {title}
Paper type: {paper_type}
Detail level: {detail}. Total report target: approximately {target_words} words.
The source section titles are: {", ".join(section_titles)}.
Include the authors and publication year when established by the front matter.
For empirical papers, distinguish study design, results, interpretation, and
author-stated limitations. Do not infer unsupported information.

Paper front matter (for names and publication date only):
{front_matter or "Not present in the indexed opening pages."}

The following are concise notes mapped from the paper's sections. They may have
been hierarchically condensed; use only facts and citations that remain here.
Use these notes to answer the requested section. Explain technical terms,
equations, methods, and reported results in plain language when relevant.
Distinguish the paper's own findings from prior work it cites. Do not treat
reference-list entries as findings or invent missing information.

--- SECTION NOTES ---
{notes_context}
--- END SECTION NOTES ---

Section source citation lists:
{source_context}

"""
    word_target = max(60, target_words // max(1, len(structure_titles)))
    output_sections: list[str] = []
    for section_title in structure_titles:
        prompt = (
            f"{notes_prefix}Write only the “{section_title}” section, in "
            f"approximately {word_target} words. Use relevant citations from "
            "the section source citation lists. Return the section body without "
            "a Markdown heading. If the evidence does not support this section, "
            "write one concise sentence stating that the supplied text does not "
            "specify it. Do not include an introduction or other report sections."
        )
        if "key takeaways" in section_title.casefold():
            prompt += " End with 3-5 concise, evidence-grounded takeaways."
        max_tokens = initial_section_tokens
        while True:
            result = _plain_markdown(await call_llm(
                prompt=prompt,
                agent_role="summarization_final",
                system=base_rules,
                temperature=0.1,
                max_tokens=max_tokens,
            ))
            if "[The response reached the model output limit and may be incomplete.]" not in result:
                break
            if max_tokens >= max_section_tokens:
                raise RuntimeError(
                    f"Final summary section '{section_title}' reached the "
                    f"{max_section_tokens}-token limit."
                )
            max_tokens = min(max_section_tokens, max_tokens * 2)
        output_sections.append(f"## {section_title}\n\n{result}")

    return "\n\n".join(output_sections)


async def run_summarization(
    user_id: str,
    paper_ids: list[str],
    target_words: int | None = None,
    progress_callback: ProgressCallback | None = None,
    detail: str = "standard",
) -> dict:
    """Explain all indexed paper sections without running a claim-verification pass."""
    paper_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAPERS)

    async def summarize_paper(paper_id: str, paper_number: int) -> dict:
        async with paper_semaphore:
            started_at = time.perf_counter()
            try:
                await _report_progress(
                    progress_callback,
                    f"Paper {paper_number}/{len(paper_ids)}: loading indexed text.",
                )
                retrieved_chunks = await _retrieve_chunks(user_id, paper_id)
                chunks = _exclude_reference_chunks(retrieved_chunks)
                if not chunks:
                    await _report_progress(
                        progress_callback,
                        f"Paper {paper_number}/{len(paper_ids)}: no body text found.",
                    )
                    return {
                        "paper_id": paper_id,
                        "title": "Unknown",
                        "summary": "No body text found for this paper. It may not be indexed yet, or its indexed text may contain only references.",
                        "status": "empty",
                    }

                page_numbers = []
                for chunk in chunks:
                    try:
                        page_numbers.append(int(chunk.get("page")))
                    except (TypeError, ValueError):
                        continue
                page_count = max(page_numbers, default=None)
                for excerpt_index, chunk in enumerate(chunks, start=1):
                    chunk["_excerpt_index"] = excerpt_index
                groups = _group_chunks_by_section(chunks, page_count=page_count)
                section_groups = [group for group in groups if group["chunks"]]
                section_contexts = _section_contexts(section_groups)
                section_source_ids = [
                    (
                        section_title,
                        list(dict.fromkeys(
                        f"{excerpt_id}]"
                        for excerpt_id in re.findall(r"\[E\d+\b", context)
                        )),
                    )
                    for section_title, context in section_contexts
                ]
                target = target_words or _target_words_for_detail(detail, page_count)
                selected_detail = detail if detail in {"quick", "standard", "deep"} else "standard"
                chunk_title = next(
                    (chunk.get("title") for chunk in chunks if chunk.get("title")),
                    None,
                )
                title = chunk_title or "Untitled Paper"
                front_matter, authors, year = _front_matter(chunks)
                profile = {
                    "authors": authors,
                    "year": year,
                    "paper_type": _infer_paper_type(front_matter + "\n" + "\n".join(
                        str(chunk.get("text") or "") for chunk in chunks[:8]
                    )),
                }
                await _report_progress(
                    progress_callback,
                    f"Paper {paper_number}/{len(paper_ids)}: mapped {len(chunks)} chunks into {len(section_groups)} sections; preparing a {selected_detail} summary.",
                )

                section_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUMMARIZATION_REQUESTS)
                section_progress_lock = asyncio.Lock()
                completed_sections = 0

                async def summarize_limited(
                    section_number: int,
                    section_title: str,
                    context: str,
                ) -> str:
                    nonlocal completed_sections
                    async with section_semaphore:
                        async with section_progress_lock:
                            await _report_progress(
                                progress_callback,
                                f"Paper {paper_number}/{len(paper_ids)}: summarizing section {section_title} ({section_number}/{len(section_contexts)}).",
                            )
                        section_started_at = time.perf_counter()
                        note = await _summarize_section(context, section_title)
                        async with section_progress_lock:
                            completed_sections += 1
                            await _report_progress(
                                progress_callback,
                                f"Paper {paper_number}/{len(paper_ids)}: sections {completed_sections}/{len(section_contexts)} analyzed; latest took {time.perf_counter() - section_started_at:.1f}s.",
                            )
                        return f"Section {section_title} notes:\n{note}"

                notes = await asyncio.gather(*(
                    summarize_limited(index, section_title, context)
                    for index, (section_title, context) in enumerate(section_contexts, start=1)
                ))
                if sum(map(len, notes)) > MAX_NOTES_CHARS:
                    await _report_progress(
                        progress_callback,
                        f"Paper {paper_number}/{len(paper_ids)}: condensing evidence notes.",
                    )
                notes = await _reduce_notes(notes)
                await _report_progress(
                    progress_callback,
                    f"Paper {paper_number}/{len(paper_ids)}: drafting a {selected_detail} report near {target} words.",
                )
                draft = await _draft_summary(
                    title,
                    profile,
                    notes,
                    target,
                    selected_detail,
                    [group["title"] for group in groups],
                    front_matter,
                    section_source_ids,
                )

                model_ids = list(dict.fromkeys(
                    spec.model_id
                    for role in ("summarization", "summarization_final")
                    for spec in get_chain_for_role(role)
                ))

                await _report_progress(
                    progress_callback,
                    f"Paper {paper_number}/{len(paper_ids)}: summary ready in {time.perf_counter() - started_at:.1f}s.",
                )
                return {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "year": profile.get("year"),
                    "paper_type": profile.get("paper_type") or "unclassified",
                    "summary": _plain_markdown(draft),
                    "sources": [
                        {
                            "excerpt": f"E{i + 1}",
                            "page": chunk.get("page"),
                            "chunk_id": chunk.get("chunk_id"),
                            "section": chunk.get("section_heading"),
                            "text": chunk.get("text", ""),
                        }
                        for i, chunk in enumerate(chunks)
                    ],
                    "chunks_used": len(chunks),
                    "sections_analyzed": len(section_groups),
                    "detail": selected_detail,
                    "excluded_reference_chunks": len(retrieved_chunks) - len(chunks),
                    "reproducibility": {
                        "models": model_ids,
                        "embedding_model": settings.OLLAMA_EMBED_MODEL,
                        "chunk_size": settings.CHUNK_SIZE,
                        "prompt_version": PROMPT_VERSION,
                        "runtime_seconds": round(time.perf_counter() - started_at, 2),
                    },
                    "status": "completed",
                }
            except Exception as exc:
                await _report_progress(
                    progress_callback,
                    f"Paper {paper_number}/{len(paper_ids)}: failed; check the returned error details.",
                )
                return {
                    "paper_id": paper_id,
                    "title": "Unknown",
                    "summary": "",
                    "status": "error",
                    "error": str(exc),
                }

    await _report_progress(
        progress_callback,
        f"Starting concise summaries for {len(paper_ids)} paper(s).",
    )
    summaries = await asyncio.gather(*(
        summarize_paper(paper_id, index)
        for index, paper_id in enumerate(paper_ids, start=1)
    ))
    await _report_progress(progress_callback, "All paper summaries are ready.")
    return {
        "agent": "summarization",
        "summaries": summaries,
        "total_papers": len(paper_ids),
    }
