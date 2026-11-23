# ============================================================
# GenResearch — Literature Review Agent
# Generates comparative literature reviews from multiple papers
# ============================================================
import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable

from config import settings
from services.citation_formatting import (
    CITATION_TAG,
    drop_unknown_citation_tags,
    find_unsupported_numbers,
    flag_uncited_sentences,
    normalize_citation_tags,
    resolve_and_append_references,
)
from services.literature_review_cards import (
    analyze_cards,
    build_comparison_matrix,
    build_paper_cards,
    chunk_budget_per_paper,
    format_analysis_for_prompt,
    format_cards_for_prompt,
    scope_note,
    section_word_guide,
    target_words,
    word_limits,
    writing_problems,
)
from database.supabase_client import get_supabase
from services.llm_service import call_llm
from services.rag_service import semantic_search


TOP_K_PER_QUERY = 6
FIRST_PAGE_CHUNKS = 2  # enough for bibliographic metadata without crowding out findings
MAX_CHUNKS_PER_PAPER = 16
FIRST_PAGE_QUERY = "title authors affiliations abstract keywords journal doi published"
FIRST_PAGE_TOP_K = 8

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str], Awaitable[None]]

REQUIRED_SECTIONS = (
    "Introduction",
    "Thematic Analysis",
    "Critical Synthesis",
    "Research Gaps",
    "Conclusion",
)
REVIEW_QUERIES = {
    "overview": "research question objectives problem scope motivation",
    "methods": "methodology study design data collection sample participants measures analysis",
    "findings": "results findings outcomes evidence reported results",
    "limitations": "limitations weaknesses threats to validity constraints",
    "implications": "discussion implications contributions future work recommendations",
    "constructs": "definition of key concepts constructs variables operationalization theoretical framework",
    "bibliography": "paper title authors publication year journal DOI citation",
}


LITERATURE_REVIEW_SYSTEM = """You are an expert academic literature-review writer. Synthesize the supplied
evidence into a critical, coherent review; do not write a sequence of paper summaries and do not
write a summary of a single paper when several are supplied.

Requirements:
1. Use exactly these five Markdown H2 headings, in this order: Introduction, Thematic Analysis,
   Critical Synthesis, Research Gaps, Conclusion. Add no other top-level sections (no
   "Recommendations", "Final Thoughts", "Key Findings" or "References"). Follow the word guide
   in the user message, excluding references.
2. Organize the thematic analysis around claims or debates (use ### subheadings per theme), not one
   paragraph per paper.
3. Compare studies directly only after checking that they address the same outcome, unit,
   population, context, and method. When they do not, describe complementary scope, not
   disagreement. Explain reasons only when the excerpts support them.
4. Evaluate evidence quality and limitations (study design, sample, time frame, retrospective vs
   prospective evidence, reliance on authors' interpretation). Preserve uncertainty and causal
   restraint. Match verbs to design: use "found", "reported", "suggests" or "is consistent with"
   for observational, survey, or event-study evidence, and reserve causal language for causal
   designs. Do not praise sources; assess them.
5. Give every selected source a substantive, evidence-backed contribution. Report countervailing,
   negative or null findings alongside positive findings. Do not drop a paper because it does not
   fit the dominant theme.
6. Keep distinct mechanisms and outcomes distinct. Where sources describe trade-offs, name the
   specific intervention, taxon/metric/population, and the direction of each claim, and never turn
   a risk tied to one strategy into a general effect of the whole approach. (Example: if a yield
   review presents land sparing as an opportunity for biodiversity while noting reduced genetic
   diversity in high-yielding varieties, keep those two points separate.)
7. Derive research gaps from limitations or missing comparisons in this supplied set. Attribute
   author-stated gaps to the authors; phrase inferred gaps as missing from this set. Do not claim
   a gap is new to the whole field. Name the specific comparison, population, method or outcome.
8. Use ONLY the supplied excerpts. Cite each specific finding, methodological detail, number,
   limitation, comparison, or gap with one or more supplied [CR-...] tags. Never credit a paper
   with another study's result.
9. Never invent citations, authors, years, results, or relationships between studies. If the
   excerpts do not support a point, qualify it or omit it. Do not add recommendations or advice
   that the excerpts do not contain.
10. Name the source study and design beside precise counts or meta-analysis results; if the
    cited study cannot be identified from the evidence, omit the count. Avoid generic conclusions
    such as "complex," "context-specific," or "interdisciplinary" without stating what evidence
    makes them so. Never leave template placeholders or write [CITATION NEEDED].
11. Make the synthesis explain what the evidence collectively means, where it conflicts, and
    what its limits imply for the strength and scope of conclusions.
12. State the number of papers synthesized in the Introduction and limit claims to this selected set.
13. Use no more than three citation tags per sentence and attribute claims to the sources supporting them.
14. Write plain prose: no bold text, no horizontal rules, no tables, no bullet lists. Do not refer
    to "excerpts", "the provided sources", or tag IDs in the prose; refer to studies by author-year
    only when that metadata is in the source key, otherwise describe them (e.g. "a systematic
    review of adoption factors")."""


# ------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------
def _filename_title(title: str, file_name: str) -> bool:
    stem = file_name.rsplit(".", 1)[0].strip().casefold() if file_name else ""
    normalized = title.strip().casefold()
    return bool(
        (stem and normalized == stem)
        or (" " not in normalized and re.search(r"\d", normalized))
    )


def _paper_metadata(chunks: list[dict], paper_id: str, record: dict | None = None) -> dict:
    chunk = next((item for item in chunks if item.get("title")), chunks[0])
    metadata = chunk.get("metadata") or {}
    record = record or {}
    authors = record.get("authors") or metadata.get("authors") or chunk.get("authors", "")
    if isinstance(authors, list):
        authors = ", ".join(str(author) for author in authors if author)
    title = record.get("title") or metadata.get("title") or chunk.get("title") or ""
    file_name = record.get("file_name", "")
    if _filename_title(title, file_name):
        title = ""
    if isinstance(authors, str) and authors.strip().casefold() in {"unknown", "n/a", "none"}:
        authors = ""
    return {
        "id": "CR-" + hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:12].upper(),
        "paper_id": paper_id,
        "title": title,
        "authors": authors if isinstance(authors, str) else "",
        "year": record.get("year") or metadata.get("year") or chunk.get("year"),
        "doi": metadata.get("doi", chunk.get("doi", "")),
        "url": metadata.get("url", chunk.get("url", "")),
        "journal": metadata.get("journal", "") or "",
        "volume": "",
        "issue": "",
        "pages": "",
        "file_name": file_name,
        "evidence_level": "direct",
    }


def _load_paper_records_sync(user_id: str, paper_ids: list[str]) -> dict[str, dict]:
    response = (
        get_supabase()
        .table("papers")
        .select("id,title,authors,year,file_name")
        .eq("user_id", user_id)
        .in_("id", paper_ids)
        .execute()
    )
    return {str(row["id"]): row for row in (response.data or []) if row.get("id")}


async def _load_paper_records(user_id: str, paper_ids: list[str]) -> dict[str, dict]:
    try:
        return await asyncio.to_thread(_load_paper_records_sync, user_id, paper_ids)
    except Exception:
        return {}


def _normalize_metadata_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _metadata_value_is_supported(value: str, evidence: str) -> bool:
    normalized_value = _normalize_metadata_text(value)
    normalized_evidence = _normalize_metadata_text(evidence)
    return bool(normalized_value and normalized_value in normalized_evidence)


def _parse_json_object(response: str) -> dict:
    """Parse a JSON object out of an LLM response; return {} on any failure."""
    try:
        start, end = response.index("{"), response.rindex("}") + 1
        parsed = json.loads(response[start:end])
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, json.JSONDecodeError, AttributeError):
        return {}


def _bibliography_complete(source: dict) -> bool:
    return bool(
        source.get("title") and source.get("authors") and source.get("year")
        and (source.get("journal") or source.get("doi"))
    )


async def _extract_missing_bibliography(
    citation_registry: list[dict],
    all_chunks: list[dict],
    progress_callback: ProgressCallback | None = None,
) -> None:
    evidence_by_source = {
        source["id"]: "\n".join(
            chunk.get("text", "")
            for chunk in all_chunks
            if chunk["citation_id"] == source["id"] and (chunk.get("page") == 1 or not chunk.get("page"))
        )
        for source in citation_registry
    }
    missing = [
        source for source in citation_registry
        if not all((source.get(field) for field in ("title", "authors", "year", "journal")))
    ]
    for source in citation_registry:
        doi_match = re.search(
            r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+",
            evidence_by_source[source["id"]],
            re.IGNORECASE,
        )
        if doi_match and not source.get("doi"):
            source["doi"] = doi_match.group(0).rstrip(".,;:)").lower()

    if missing:
        prompt = f"""Extract bibliographic metadata only when it is explicitly printed in the supplied
first-page excerpts. Do not use outside knowledge or infer missing fields. Return JSON only:
{{"sources": [{{"id": "CR-...", "title": null, "authors": [], "year": null,
"journal": null, "volume": null, "issue": null, "pages": null}}]}}
Keep title, author names, and journal wording as printed. Include one object for each source.

--- FIRST-PAGE EVIDENCE ---
{chr(10).join(f"SOURCE {source['id']}\n{evidence_by_source[source['id']]}" for source in missing)}
--- END EVIDENCE ---"""
        try:
            raw = await call_llm(
                prompt=prompt,
                agent_role="literature_review",
                system="Extract only verbatim bibliographic facts. Return valid JSON only.",
                temperature=0.0,
                max_tokens=1800,
                progress_callback=progress_callback,
            )
        except Exception:
            raw = ""
        parsed = _parse_json_object(raw)
        extracted = {
            item.get("id"): item
            for item in (parsed.get("sources") or [])
            if isinstance(item, dict) and item.get("id")
        }
        for source in missing:
            item = extracted.get(source["id"], {})
            evidence = evidence_by_source[source["id"]]
            title = item.get("title")
            if not source.get("title") and isinstance(title, str) and _metadata_value_is_supported(title, evidence):
                source["title"] = title.strip()
            authors = item.get("authors")
            if not source.get("authors") and isinstance(authors, list) and authors:
                names = [str(name).strip() for name in authors if str(name).strip()]
                evidence_words = set(re.findall(r"[^\W\d_]+", evidence.casefold()))
                if names and all(
                    name.split() and name.split()[-1].casefold().rstrip(".,") in evidence_words
                    for name in names
                ):
                    source["authors"] = ", ".join(names)
            year = str(item.get("year") or "")
            if not source.get("year") and re.fullmatch(r"(?:18|19|20|21)\d{2}", year) and year in evidence:
                source["year"] = int(year)
            for field in ("journal", "volume", "issue", "pages"):
                value = item.get(field)
                if value and not source.get(field) and _metadata_value_is_supported(str(value), evidence):
                    source[field] = str(value).strip()

    for source in citation_registry:
        source["bibliography_complete"] = _bibliography_complete(source)


def _apply_title_fallbacks(citation_registry: list[dict]) -> None:
    """Keep sources whose title could not be read in the reference list, clearly marked.

    Run only AFTER generation so the model never sees (or cites) a filename as a title.
    """
    for source in citation_registry:
        source["title_verified"] = bool(source.get("title"))
        if not source.get("title"):
            stem = (source.get("file_name") or "").rsplit(".", 1)[0].strip()
            source["title"] = stem or "Untitled source"
            source["evidence_level"] = "partial"


# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------
def _review_context_char_budget(max_output_tokens: int = 2_600) -> int:
    prompt_reserve_tokens = 2_000
    available_tokens = max(
        1_000,
        settings.MODAL_CONTEXT - max_output_tokens - prompt_reserve_tokens,
    )
    return available_tokens * 2


def _review_draft_max_tokens(n_papers: int) -> int:
    target = target_words(n_papers)
    desired = max(3_200, int(target * 3))
    return min(6_000, settings.MODAL_MAX_TOKENS, desired)


def _chunks_per_paper(paper_count: int) -> int:
    return chunk_budget_per_paper(
        paper_count,
        char_budget=_review_context_char_budget(),
    )


def _chunk_key(chunk: dict) -> str:
    return str(
        chunk.get("chunk_id")
        or chunk.get("id")
        or hashlib.sha1(chunk.get("text", "").encode("utf-8")).hexdigest()
    )


async def _retrieve_paper_chunks(
    user_id: str,
    paper_id: str,
    focus_topic: str,
    max_chunks: int = MAX_CHUNKS_PER_PAPER,
) -> list[dict]:
    queries = [
        f"{focus_topic} {query}".strip() if focus_topic else query
        for query in REVIEW_QUERIES.values()
    ]
    first_page_result, *results = await asyncio.gather(
        semantic_search(
            user_id=user_id,
            query=FIRST_PAGE_QUERY,
            top_k=FIRST_PAGE_TOP_K,
            paper_id=paper_id,
        ),
        *(
            semantic_search(
                user_id=user_id,
                query=query,
                top_k=TOP_K_PER_QUERY,
                paper_id=paper_id,
            )
            for query in queries
        ),
        return_exceptions=True,
    )

    query_groups: list[list[dict]] = []
    for result in results:
        if isinstance(result, Exception):
            query_groups.append([])
            continue
        query_groups.append(list(result or []))

    def relevance(chunk: dict) -> float:
        if chunk.get("hybrid_score") is not None:
            return float(chunk["hybrid_score"])
        return -float(chunk.get("distance", 1.0))

    for group in query_groups:
        group.sort(key=relevance, reverse=True)

    first_page_pool = (
        [] if isinstance(first_page_result, Exception) else list(first_page_result or [])
    )
    first_page_pool.sort(key=relevance, reverse=True)

    selected: list[dict] = []
    seen: set[str] = set()

    def add_chunk(chunk: dict) -> None:
        key = _chunk_key(chunk)
        if key not in seen and len(selected) < max_chunks:
            seen.add(key)
            selected.append(chunk)

    first_page_chunks: dict[str, dict] = {}
    for chunk in first_page_pool:
        if chunk.get("page") == 1:
            first_page_chunks.setdefault(_chunk_key(chunk), chunk)
    for group in query_groups:
        for chunk in group:
            if chunk.get("page") == 1:
                first_page_chunks.setdefault(_chunk_key(chunk), chunk)
    for chunk in list(first_page_chunks.values())[:FIRST_PAGE_CHUNKS]:
        add_chunk(chunk)

    for rank in range(TOP_K_PER_QUERY):
        for group in query_groups:
            if rank < len(group):
                add_chunk(group[rank])
            if len(selected) >= max_chunks:
                break
        if len(selected) >= max_chunks:
            break
    return selected


# ------------------------------------------------------------
# Review text normalisation + validation
# ------------------------------------------------------------
_SECTION_NAMES = {name.casefold(): name for name in REQUIRED_SECTIONS}
_SECTION_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:\d+[.)]\s*)?"
    r"(Introduction|Thematic Analysis|Critical Synthesis|Research Gaps|Conclusion)"
    r"\s*(?:\*\*|__)?\s*:?\s*(?:\*\*|__)?\s*$",
    re.IGNORECASE,
)
_RULE_LINE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_REFERENCES_HEADING = re.compile(
    r"^\s*#{1,6}\s*\**\s*(?:references|bibliography|works cited)\b", re.IGNORECASE
)
_MODEL_MARKER = re.compile(r"[ \t]*\[CITATION NEEDED\]")


def _normalize_review_markup(text: str) -> str:
    """Clean model output: canonical H2 headings, no rules/markers/self-written references."""
    text = (text or "").replace("\r\n", "\n").strip()
    text = re.sub(r"^```(?:markdown|md)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)

    lines: list[str] = []
    for line in text.split("\n"):
        if _REFERENCES_HEADING.match(line):
            break  # the reference list is generated from the registry
        if _RULE_LINE.match(line):
            continue
        section = _SECTION_LINE.match(line)
        if section:
            lines.append(f"## {_SECTION_NAMES[section.group(1).casefold()]}")
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = _MODEL_MARKER.sub("", text)
    text = normalize_citation_tags(text)
    text = re.sub(r"(?m)^(#{1,6} .+)$", r"\n\1\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _review_problems(text: str, registry: list[dict]) -> list[str]:
    """Structural, source coverage, length, and writing-quality repair checks."""
    problems: list[str] = []
    headings = re.findall(r"(?m)^##\s+(.+?)\s*$", text)
    present = {heading.casefold() for heading in headings}

    missing = [name for name in REQUIRED_SECTIONS if name.casefold() not in present]
    if missing:
        problems.append("Missing required section heading(s): " + ", ".join(missing) + ".")
    extra = [heading for heading in headings if heading.casefold() not in _SECTION_NAMES]
    if extra:
        problems.append(
            "Remove unexpected top-level section(s): " + ", ".join(extra)
            + ". Use only the five required headings."
        )

    valid = {source["id"] for source in registry}
    cited = {item.upper() for item in CITATION_TAG.findall(text)}
    if not cited & valid:
        problems.append("No valid [CR-...] citation tags were found; cite every substantive sentence.")
    unknown = cited - valid
    if unknown:
        problems.append("Unknown citation tags used: " + ", ".join(sorted(unknown)) + ".")
    uncited = [source for source in registry if source["id"] not in cited]
    if uncited:
        problems.append(
            "These sources are never cited; give each a substantive, correctly cited point: "
            + "; ".join(f"{s['id']} ({s.get('title') or 'title not verified'})" for s in uncited)
            + "."
        )

    n_papers = len(registry)
    low, high = word_limits(n_papers)
    words = len(re.findall(r"\b\w+\b", CITATION_TAG.sub("", text)))
    if words < low:
        problems.append(f"Too short ({words} words); target about {target_words(n_papers)} words.")
    elif words > high:
        problems.append(f"Too long ({words} words); target about {target_words(n_papers)} words.")
    problems.extend(writing_problems(text, n_papers))
    return problems


# ------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------
def _source_key(registry: list[dict]) -> str:
    return "\n".join(
        f"- {item['id']}: {item['title'] or 'Title not verified'} | "
        f"{item['authors'] or 'Author unknown'} | {item['year'] or 'Year unknown'}"
        for item in registry
    )


def _trim_context_to_budget(
    registry: list[dict],
    all_chunks: list[dict],
    char_budget: int | None = None,
) -> list[dict]:
    """Keep one excerpt per source where possible, then fill remaining prompt space."""
    char_budget = char_budget or _review_context_char_budget()
    by_source: dict[str, list[dict]] = {s["id"]: [] for s in registry}
    for chunk in all_chunks:
        cid = chunk.get("citation_id", "")
        if cid in by_source:
            by_source[cid].append(chunk)

    def header(source: dict) -> str:
        return (
            f"=== SOURCE {source['id']} | {source['title'] or 'Title not verified'} | "
            f"{source['authors'] or 'Author unknown'} | {source['year'] or 'Year unknown'} ==="
        )

    header_cost = sum(len(header(source)) + 2 for source in registry)
    header_cost += 2 * max(0, len(registry) - 1)
    remaining = max(0, char_budget - header_cost)
    selected: list[dict] = []
    selected_keys: set[tuple[str, str]] = set()

    # First pass spreads available excerpt space across sources, clipping only if needed.
    sources_left = len(registry)
    for source in registry:
        candidates = by_source[source["id"]]
        if not candidates or remaining <= 0:
            sources_left -= 1
            continue
        chosen = next((chunk for chunk in candidates if chunk.get("page") != 1), candidates[0])
        prefix = f"[{source['id']} | page {chosen.get('page') or 'unknown'}]\n"
        share = remaining // max(sources_left, 1)
        text_budget = min(len(str(chosen.get("text") or "")), max(0, share - len(prefix)))
        if text_budget:
            selected.append({**chosen, "text": str(chosen.get("text") or "")[:text_budget]})
            selected_keys.add((source["id"], _chunk_key(chosen)))
            remaining -= len(prefix) + text_budget
        sources_left -= 1

    # Then add further high-ranked excerpts while they fit.
    for source in registry:
        for chunk in by_source[source["id"]]:
            if (source["id"], _chunk_key(chunk)) in selected_keys:
                continue
            prefix = f"[{source['id']} | page {chunk.get('page') or 'unknown'}]\n"
            separator = 2 if any(item.get("citation_id") == source["id"] for item in selected) else 0
            size = len(str(chunk.get("text") or ""))
            available = remaining - separator - len(prefix)
            if available <= 0:
                break
            text = str(chunk.get("text") or "")
            take = min(size, available)
            if take:
                selected.append({**chunk, "text": text[:take]})
                selected_keys.add((source["id"], _chunk_key(chunk)))
                remaining -= separator + len(prefix) + take
            if take < size:
                break
    return selected


def _build_context(
    registry: list[dict],
    all_chunks: list[dict],
    char_budget: int | None = None,
) -> str:
    """Group excerpts by source so the model can see every paper is present.

    Trims the chunk list to the configured Ollama context budget before building it so
    the resulting prompt never overflows the model's context window.
    """
    char_budget = char_budget or _review_context_char_budget()
    trimmed = _trim_context_to_budget(registry, all_chunks, char_budget)
    blocks = []
    for source in registry:
        header = (
            f"=== SOURCE {source['id']} | {source['title'] or 'Title not verified'} | "
            f"{source['authors'] or 'Author unknown'} | {source['year'] or 'Year unknown'} ==="
        )
        body = "\n\n".join(
            f"[{source['id']} | page {chunk.get('page') or 'unknown'}]\n{chunk.get('text', '')}"
            for chunk in trimmed
            if chunk["citation_id"] == source["id"]
        )
        blocks.append(f"{header}\n\n{body}")
    context = "\n\n".join(blocks)
    if len(context) > char_budget:
        raise ValueError(
            f"Literature-review context exceeded its {char_budget}-character budget."
        )
    return context



def _build_generation_prompt(
    registry: list[dict],
    context: str,
    focus_topic: str,
    structured_block: str = "",
    extra_instruction: str = "",
) -> str:
    topic_instruction = (
        f'Focus the review on the topic: "{focus_topic}".'
        if focus_topic
        else "Identify the central themes and debates represented in the supplied papers."
    )
    comparison_instruction = (
        "Compare only like with like. Before calling findings contradictory, check outcome, "
        "unit, population, context, methods, and time scale; otherwise frame them as "
        "complementary evidence with different scopes."
        if len(registry) > 1
        else "This is a single-paper review: critically synthesize its evidence and do not imply cross-study comparisons."
    )
    example_tag = f"[{registry[0]['id']}]"
    checklist = "\n".join(f"- [{item['id']}]" for item in registry)
    guide = section_word_guide(len(registry))
    length_guide = "; ".join(f"{name} {words} words" for name, words in guide.items())
    notes = (
        f"\n--- STRUCTURED NOTES (per-paper evidence cards and cross-paper analysis) ---\n"
        f"{structured_block}\n--- END STRUCTURED NOTES ---\n"
        "Use these notes to plan the review. They are summaries: support every statement with "
        "the excerpts below and cite it. Findings marked as cited from other work are not the "
        "source paper's own findings.\n"
        if structured_block
        else ""
    )

    return f"""Write a critical literature review grounded ONLY in the excerpts below.

SOURCE KEY (cite using these exact tags; never cite filenames):
{_source_key(registry)}

{topic_instruction}
{comparison_instruction}
{extra_instruction}

REQUIRED FORMAT
- Exactly these five H2 headings, in this order: ## Introduction, ## Thematic Analysis, ## Critical Synthesis, ## Research Gaps, ## Conclusion.
- Target about {target_words(len(registry))} words overall. Section guide: {length_guide} (### subheadings per theme or debate, never per paper).
- The Introduction must state that this review synthesizes {len(registry)} selected paper{'s' if len(registry) != 1 else ''} and that conclusions are limited to them.
- Plain prose only. No bold, no horizontal rules, no tables, no lists, no reference list, no extra sections.
- Place citation tags immediately before the sentence's final punctuation, for example: "...was reported in the trial {example_tag}." Use no more than three tags per sentence.
- Never write [CITATION NEEDED].

COVERAGE CHECKLIST: every tag below must be cited at least once with a substantive, evidence-based point
(a review that rests on only some of the papers is a failure):
{checklist}

Plan silently before writing: which evidence supports each theme, each paper's countervailing findings,
methodological differences, limitations, and gaps (distinguish author-stated gaps from those inferred from
this set). Compare only when outcome, population, method, and time scale are sufficiently similar; otherwise
describe complementary scope, not disagreement. Cite both studies for every direct comparison. Use cautious
verbs for observational evidence and cite precise counts with the study and design. Use author-year prose only
when metadata is present in the source key.
{notes}

--- PAPER EXCERPTS (grouped by source) ---

{context}

--- END OF EXCERPTS ---

Write the review now:"""


def _build_repair_prompt(
    registry: list[dict], context: str, draft: str, problems: list[str]
) -> str:
    return f"""Revise the draft literature review so that every listed problem is fixed.

PROBLEMS TO FIX:
{chr(10).join(f"- {problem}" for problem in problems)}

SOURCE KEY:
{_source_key(registry)}

Rules: use exactly the five H2 headings (Introduction, Thematic Analysis, Critical Synthesis, Research Gaps,
Conclusion) in that order; plain prose with ### subheadings only inside Thematic Analysis; about
{target_words(len(registry))} words;
use ONLY the excerpts below; cite every substantive sentence with the exact [CR-...] tags from the source key
(tag immediately before final punctuation, no more than three tags per sentence); every source must receive
at least one substantive cited point; compare studies only when they measure the same outcome and population;
use cautious language for observational evidence; no praise, no recommendations
that the excerpts do not contain, no reference list, never write [CITATION NEEDED]. Keep supported content
from the draft; remove or qualify anything the excerpts do not support.

--- SOURCE EXCERPTS ---
{context}
--- END SOURCE EXCERPTS ---

--- DRAFT ---
{draft}
--- END DRAFT ---

Return ONLY the revised review in Markdown:"""


def _build_audit_prompt(context: str, draft: str) -> str:
    return f"""Audit the draft against the source excerpts. Check whether every selected paper contributes
at least one substantive, correctly cited point; whether every specific claim, number, comparison, limitation
and proposed gap is supported by the cited source(s); whether contradictions and uncertainty are represented
fairly; and whether the draft synthesizes rather than lists papers. Do not describe differences between
non-comparable outcomes (for example livestock vs crop outcomes) as contradictory findings, and do not merge
distinct risks and opportunities. Remove or qualify unsupported claims; never add facts; never drop a source;
keep the same five H2 headings and the valid [CR-...] citation tags; do not add sections or a reference list.
If no correction is needed, return the draft unchanged.

Reply in EXACTLY this format and nothing else:

ISSUES:
- one concise issue per line (write "- none" if there are no issues)

CORRECTED_REVIEW:
<the full corrected review in Markdown>

--- SOURCE EXCERPTS ---
{context}
--- END SOURCE EXCERPTS ---

--- DRAFT ---
{draft}
--- END DRAFT ---"""


_AUDIT_FORMAT = re.compile(r"ISSUES:\s*(.*?)\s*CORRECTED_REVIEW:\s*(.*)\Z", re.DOTALL | re.IGNORECASE)


def _parse_audit(response: str) -> dict:
    """Parse the auditor's reply (delimited format preferred, JSON accepted)."""
    if not response:
        return {}
    match = _AUDIT_FORMAT.search(response)
    if match:
        issues = [
            re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            for line in match.group(1).splitlines()
            if line.strip()
        ]
        issues = [item for item in issues if item.casefold().rstrip(".") != "none"]
        return {"issues": issues, "corrected_review": match.group(2).strip()}
    parsed = _parse_json_object(response)
    if parsed:
        issues = parsed.get("issues")
        parsed["issues"] = [str(item) for item in issues] if isinstance(issues, list) else []
    return parsed


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
async def run_literature_review(
    user_id: str,
    paper_ids: list[str],
    focus_topic: str = "",
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """
    Generate a comparative literature review from multiple papers.
    Uses RAG to retrieve relevant content and synthesizes across papers.
    """
    started_at = time.perf_counter()

    async def report_progress(message: str) -> None:
        logger.info("literature_review_progress: %s", message)
        if progress_callback:
            await progress_callback(message)

    all_chunks: list[dict] = []
    citation_registry: list[dict] = []
    paper_titles: list[str] = []
    unique_paper_ids = list(dict.fromkeys(paper_ids))
    await report_progress(
        f"Starting literature review for {len(unique_paper_ids)} selected paper(s)."
    )
    await report_progress("Loading paper titles and bibliographic details.")
    paper_records = await _load_paper_records(user_id, unique_paper_ids)

    per_paper = _chunks_per_paper(len(unique_paper_ids))
    await report_progress(
        f"Searching {len(unique_paper_ids)} paper(s) for study aims, methods, "
        "findings, limitations, and key concepts."
    )

    async def retrieve_one(index: int, paper_id: str) -> tuple[str, list[dict]]:
        await report_progress(
            f"Retrieving evidence for paper {index}/{len(unique_paper_ids)}."
        )
        chunks = await _retrieve_paper_chunks(
            user_id, paper_id, focus_topic, per_paper
        )
        await report_progress(
            f"Retrieved {len(chunks)} evidence excerpts for paper "
            f"{index}/{len(unique_paper_ids)}."
        )
        return paper_id, chunks

    retrieved = await asyncio.gather(*(
        retrieve_one(index, paper_id)
        for index, paper_id in enumerate(unique_paper_ids, start=1)
    ))
    for paper_id, chunks in retrieved:
        if not chunks:
            continue
        source = _paper_metadata(chunks, paper_id, paper_records.get(paper_id))
        citation_registry.append(source)
        for chunk in chunks:
            all_chunks.append({**chunk, "citation_id": source["id"]})

    if not citation_registry:
        await report_progress("No indexed content was found in the selected papers.")
        return {
            "agent": "literature_review",
            "review": "No content found in the selected papers.",
            "papers_analyzed": 0,
            "status": "empty",
        }

    await report_progress("Checking title-page metadata and resolving source details.")
    await _extract_missing_bibliography(
        citation_registry,
        all_chunks,
        progress_callback=report_progress,
    )
    paper_titles = [source["title"] or "Title not verified" for source in citation_registry]

    valid_ids = {source["id"] for source in citation_registry}
    evidence_by_source = {
        source["id"]: "\n".join(
            chunk.get("text", "") for chunk in all_chunks if chunk["citation_id"] == source["id"]
        )
        for source in citation_registry
    }
    await report_progress(
        f"Extracting quote-grounded evidence cards from {len(citation_registry)} paper(s)."
    )
    cards = await build_paper_cards(
        citation_registry,
        all_chunks,
        focus_topic,
        progress_callback=report_progress,
    )
    analysis_error = ""
    try:
        await report_progress("Comparing study themes, methods, findings, and limitations.")
        analysis = await analyze_cards(
            cards,
            focus_topic,
            progress_callback=report_progress,
        )
        if analysis:
            await report_progress(
                f"Cross-paper analysis identified {len(analysis.get('themes', []))} theme(s), "
                f"{len(analysis.get('agreements', []))} agreement(s), and "
                f"{len(analysis.get('gaps', []))} research gap(s)."
            )
        else:
            await report_progress(
                "Not enough quote-checked cards for cross-paper analysis; continuing from paper evidence."
            )
    except Exception as exc:
        analysis = {}
        analysis_error = str(exc)
        logger.warning("literature_review_analysis_failed: %s", exc)
        await report_progress(
            "Cross-paper analysis was unavailable; continuing with individual paper evidence."
        )
    comparison_matrix = build_comparison_matrix(cards)
    target_length = target_words(len(citation_registry))
    draft_max_tokens = _review_draft_max_tokens(len(citation_registry))
    context_budget = _review_context_char_budget(draft_max_tokens)
    cards_block = format_cards_for_prompt(cards, max_chars=context_budget // 3)
    analysis_block = format_analysis_for_prompt(analysis)
    if len(analysis_block) > context_budget // 5:
        analysis_block = ""
    structured_block = "\n\n".join(
        part for part in (cards_block, analysis_block) if part
    )
    excerpt_budget = max(1, context_budget - len(structured_block))
    context = _build_context(citation_registry, all_chunks, excerpt_budget)

    # ---- 1. Draft ----------------------------------------------------------
    await report_progress(
        f"Drafting a review across {len(citation_registry)} paper(s) "
        f"using {len(all_chunks)} retrieved excerpts."
    )
    try:
        draft = await call_llm(
            prompt=_build_generation_prompt(
                citation_registry,
                context,
                focus_topic,
                structured_block=structured_block,
                extra_instruction=scope_note(analysis, focus_topic),
            ),
            agent_role="literature_review",
            system=LITERATURE_REVIEW_SYSTEM,
            temperature=0.1,
            max_tokens=draft_max_tokens,
            progress_callback=report_progress,
        )
    except Exception as exc:
        await report_progress("Review drafting failed; see the error details in the result.")
        return {
            "agent": "literature_review",
            "review": f"The review could not be generated: {exc}",
            "papers_analyzed": len(paper_titles),
            "paper_titles": paper_titles,
            "sources": citation_registry,
            "paper_cards": cards,
            "comparison_matrix": comparison_matrix,
            "status": "error",
        }

    candidate = _normalize_review_markup(draft)
    problems = _review_problems(candidate, citation_registry)

    # ---- 2. Structural repair (one targeted retry) -------------------------
    repair_attempted = False
    if problems:
        repair_attempted = True
        await report_progress(
            f"Improving review structure and citation coverage ({len(problems)} issue(s) found)."
        )
        try:
            repaired_raw = await call_llm(
                prompt=_build_repair_prompt(citation_registry, context, candidate, problems),
                agent_role="literature_review",
                system=LITERATURE_REVIEW_SYSTEM,
                temperature=0.1,
                max_tokens=draft_max_tokens,
                progress_callback=report_progress,
            )
        except Exception:
            repaired_raw = ""
        repaired = _normalize_review_markup(repaired_raw)
        if repaired:
            repaired_problems = _review_problems(repaired, citation_registry)
            if len(repaired_problems) <= len(problems):
                candidate, problems = repaired, repaired_problems

    # ---- 3. Evidence audit ---------------------------------------------------
    audit_issues: list[str] = []
    audit_accepted = False
    await report_progress("Checking the draft against the retrieved paper excerpts.")
    try:
        audit_raw = await call_llm(
            prompt=_build_audit_prompt(context, candidate),
            agent_role="literature_review",
            system="You are a strict academic evidence auditor. Follow the requested output format exactly.",
            temperature=0.0,
            max_tokens=min(3_200, draft_max_tokens + 400),
            progress_callback=report_progress,
        )
    except Exception:
        audit_raw = ""
    audit = _parse_audit(audit_raw)
    audit_issues = [str(item) for item in (audit.get("issues") or [])]
    corrected = audit.get("corrected_review")
    if isinstance(corrected, str) and corrected.strip():
        corrected = _normalize_review_markup(corrected)
        corrected_problems = _review_problems(corrected, citation_registry)
        if corrected and len(corrected_problems) <= len(problems):
            candidate, problems, audit_accepted = corrected, corrected_problems, True
        else:
            audit_issues.append(
                "The auditor's rewrite was discarded because it broke the required structure or dropped sources."
            )
    if not audit_accepted and not audit_issues:
        audit_issues.append("Evidence audit unavailable; draft was not independently verified.")

    # ---- 4. Post-processing --------------------------------------------------
    review_text = drop_unknown_citation_tags(candidate, valid_ids)
    number_warnings = find_unsupported_numbers(
        review_text,
        evidence_by_source,
        valid_ids,
        extra_allowed=_source_key(citation_registry),
    )

    cited_source_ids = {item.upper() for item in CITATION_TAG.findall(review_text)}
    uncited_sources = [
        {"id": source["id"], "title": source.get("title") or "Title not verified"}
        for source in citation_registry
        if source["id"] not in cited_source_ids
    ]

    _apply_title_fallbacks(citation_registry)
    # Resolve [CR-...] tags to author-year text BEFORE the uncited-sentence check.
    # If we flag first, the [CR-...] tags are still present and valid sentences pass
    # the check — but the old code was already correct in that regard.  The critical
    # fix here is that resolve_and_append_references now runs with freeform-citation
    # checking AFTER tag resolution, eliminating the double-flagging that produced
    # "[CITATION NEEDED]" next to "(Baudry et al., 2018)" in the final output.
    review_text = resolve_and_append_references(review_text, citation_registry)
    review_text, uncited_count, sentence_count = flag_uncited_sentences(review_text, valid_ids)

    await report_progress(
        f"Literature review ready in {time.perf_counter() - started_at:.1f}s."
    )

    return {
        "agent": "literature_review",
        "review": review_text,
        "papers_analyzed": len(paper_titles),
        "paper_titles": paper_titles,
        "chunks_used": len(all_chunks),
        "focus_topic": focus_topic,
        "sources": citation_registry,
        "paper_cards": cards,
        "analysis": analysis,
        "comparison_matrix": comparison_matrix,
        "scope_warning": (
            analysis.get("cohesion_note", "")
            if analysis.get("cohesion") == "low"
            else ""
        ),
        "bibliography_quality": {
            "complete_sources": sum(bool(source.get("bibliography_complete")) for source in citation_registry),
            "total_sources": len(citation_registry),
        },
        "evidence_quality": {
            "cards_ok": sum(1 for card in cards if card.get("ok")),
            "quotes_verified": sum(
                int(card.get("quote_verification", {}).get("verified", 0))
                for card in cards
            ),
            "quotes_checked": sum(
                int(card.get("quote_verification", {}).get("total", 0))
                for card in cards
            ),
            "card_errors": [
                {"paper_id": card["paper_id"], "error": card["error"]}
                for card in cards
                if card.get("error")
            ],
        },
        "analysis_error": analysis_error or None,
        "verification": {
            "completed": audit_accepted,
            "issues": audit_issues,
        },
        "structure": {
            "repair_attempted": repair_attempted,
            "remaining_problems": problems,
        },
        "warnings": number_warnings,
        "citation_quality": {
            "coverage": round((sentence_count - uncited_count) / sentence_count, 3)
            if sentence_count
            else 1.0,
            "uncited_sentences": uncited_count,
            "sentences_checked": sentence_count,
            "sources_cited": len(citation_registry) - len(uncited_sources),
            "sources_uncited": uncited_sources,
        },
        "status": "completed",
    }