"""Structured evidence extraction and comparison helpers for literature reviews."""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable

from config import settings
from services.llm_service import call_llm

CARD_CONCURRENCY = 3
MAX_FINDINGS = 6
MAX_LIMITATIONS = 4
MAX_CONSTRUCTS = 6
MAX_THEMES = 5

PAPER_TYPES = {"empirical", "survey", "framework", "theory", "system", "other"}
ATTRIBUTIONS = {"own_finding", "cited_from_other_work", "unclear"}
STRENGTHS = {"causal", "correlational", "descriptive", "theoretical", "unclear"}
DIRECTIONS = {"positive", "negative", "mixed", "null", "n/a"}
GAP_ORIGINS = {"stated_by_authors", "inferred_from_set"}
COHESION_LEVELS = {"high", "medium", "low"}
ProgressCallback = Callable[[str], Awaitable[None]]

_TAG = re.compile(r"\[CR-[0-9A-Za-z]{12}\]")
_STRONG_TERMS = re.compile(
    r"\b(demonstrat(?:e|es|ed|ing)|prove[sd]?|proving|reliably|consistently|"
    r"conclusively|definitively|undoubtedly|unequivocally|clearly "
    r"(?:show|shows|showed|indicate|indicates|indicated))\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_SCOPE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"(?:\s+[\w-]+){0,2}?\s+(?:papers|studies|sources|articles|publications|works)\b",
    re.IGNORECASE,
)


def parse_json_object(response: str) -> dict:
    """Parse a JSON object from model output, including simple fenced responses."""
    try:
        start, end = response.index("{"), response.rindex("}") + 1
        parsed = json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError, AttributeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clip(value: object, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def quote_is_supported(quote: str, evidence: str) -> bool:
    """Check that each meaningful quote fragment appears in retrieved evidence."""
    fragments = [
        fragment
        for fragment in re.split(r"\.\.\.|…", quote or "")
        if len(_norm(fragment).split()) >= 4
    ]
    haystack = _norm(evidence)
    return bool(fragments) and all(_norm(fragment) in haystack for fragment in fragments)


def paper_label(source: dict) -> str:
    authors = str(source.get("authors") or "").strip()
    year = source.get("year")
    if authors and year:
        parts = [part.strip() for part in re.split(r",| and | & ", authors) if part.strip()]
        words = parts[0].split() if parts else []
        surname = words[-1] if words else authors
        return f"{surname}{' et al.' if len(parts) > 1 else ''} ({year})"
    title = str(source.get("title") or "").strip() or "Untitled source"
    return title if len(title) <= 60 else title[:57].rstrip() + "..."


def chunk_budget_per_paper(
    n_papers: int,
    char_budget: int = 80_000,
    avg_chunk_chars: int = 1_200,
    header_chars: int = 150,
    floor: int = 3,
    ceiling: int = 16,
) -> int:
    count = max(n_papers, 1)
    usable = max(char_budget - count * header_chars, 0)
    per_paper_budget = max(1, char_budget // (avg_chunk_chars * count))
    effective_floor = min(floor, per_paper_budget)
    return max(effective_floor, min(ceiling, usable // (avg_chunk_chars * count)))


def target_words(n_papers: int) -> int:
    return max(700, min(1_600, 300 + 130 * max(n_papers, 1)))


def word_limits(n_papers: int) -> tuple[int, int]:
    target = target_words(n_papers)
    return int(target * 0.6), int(target * 1.5)


def section_word_guide(n_papers: int) -> dict[str, str]:
    target = target_words(n_papers)
    shares = {
        "Introduction": 0.12,
        "Thematic Analysis": 0.45,
        "Critical Synthesis": 0.20,
        "Research Gaps": 0.13,
        "Conclusion": 0.10,
    }
    return {
        name: (
            f"{int(round(target * share * 0.8 / 10) * 10)}-"
            f"{int(round(target * share * 1.2 / 10) * 10)}"
        )
        for name, share in shares.items()
    }


def _card_prompt(source: dict, chunks: list[dict], focus_topic: str) -> str:
    excerpts = "\n\n".join(
        f"[page {chunk.get('page') or 'unknown'}]\n{chunk.get('text', '')}"
        for chunk in chunks
    )
    focus = f'Review focus: "{focus_topic}".\n' if focus_topic else ""
    return f"""Extract a structured evidence card for ONE paper, using ONLY the excerpts below.
{focus}Return JSON only, with this shape:
{{
  "paper_type": "empirical | survey | framework | theory | system | other",
  "research_question": "one sentence, or null",
  "method": "design and analysis approach in one sentence, or null",
  "data_sample": "data source, sample size and period, or null",
  "constructs": {{"term": "how THIS paper defines or measures it"}},
  "findings": [
    {{"text": "one specific finding, with numbers where given",
      "quote": "verbatim passage from the excerpts that supports it (at most 200 characters)",
      "page": 12,
      "attribution": "own_finding | cited_from_other_work | unclear",
      "strength": "causal | correlational | descriptive | theoretical | unclear",
      "direction": "positive | negative | mixed | null | n/a"}}
  ],
  "stated_limitations": ["a limitation the authors themselves state"]
}}

Rules:
- At most {MAX_FINDINGS} findings; include negative, null or mixed results when present.
- Mark cited_from_other_work whenever the paper reports another study's result.
- Quotes must be copied exactly from the excerpts; never paraphrase inside quote.
- Use null or [] when the excerpts do not say. Do not use outside knowledge.

--- EXCERPTS ({source['id']}) ---
{excerpts}
--- END EXCERPTS ---"""


def _empty_card(source: dict, error: str = "") -> dict:
    return {
        "paper_id": source["id"],
        "label": paper_label(source),
        "paper_type": "other",
        "research_question": "",
        "method": "",
        "data_sample": "",
        "constructs": {},
        "findings": [],
        "stated_limitations": [],
        "quote_verification": {"verified": 0, "total": 0},
        "ok": False,
        "error": error,
    }


def _choice(value: object, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def clean_card(parsed: dict, source: dict, chunks: list[dict]) -> dict:
    """Validate a model-produced card against retrieved evidence and known page numbers."""
    card = _empty_card(source)
    if not parsed:
        return card

    evidence = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
    pages = {chunk.get("page") for chunk in chunks if chunk.get("page")}
    findings = []
    raw_findings = parsed.get("findings")
    for item in (raw_findings if isinstance(raw_findings, list) else [])[:MAX_FINDINGS]:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        quote = _clip(item.get("quote"), 300)
        findings.append({
            "text": _clip(item.get("text"), 400),
            "quote": quote,
            "page": item.get("page") if isinstance(item.get("page"), int) and item["page"] in pages else None,
            "attribution": _choice(item.get("attribution"), ATTRIBUTIONS, "unclear"),
            "strength": _choice(item.get("strength"), STRENGTHS, "unclear"),
            "direction": _choice(item.get("direction"), DIRECTIONS, "n/a"),
            "quote_verified": quote_is_supported(quote, evidence),
        })

    raw_constructs = parsed.get("constructs")
    constructs = {}
    if isinstance(raw_constructs, dict):
        constructs = {
            _clip(key, 80): _clip(value, 200)
            for key, value in list(raw_constructs.items())[:MAX_CONSTRUCTS]
            if str(key).strip() and str(value or "").strip()
        }

    raw_limitations = parsed.get("stated_limitations")
    limitations = [
        _clip(item, 300)
        for item in (raw_limitations if isinstance(raw_limitations, list) else [])[:MAX_LIMITATIONS]
        if str(item or "").strip()
    ]
    verified = sum(1 for finding in findings if finding["quote_verified"])
    card.update({
        "paper_type": _choice(parsed.get("paper_type"), PAPER_TYPES, "other"),
        "research_question": _clip(parsed.get("research_question"), 300),
        "method": _clip(parsed.get("method"), 300),
        "data_sample": _clip(parsed.get("data_sample"), 300),
        "constructs": constructs,
        "findings": findings,
        "stated_limitations": limitations,
        "quote_verification": {"verified": verified, "total": len(findings)},
        "ok": bool(verified),
    })
    return card


def usable_findings(card: dict) -> list[dict]:
    """Return only findings whose verbatim evidence quote passed the deterministic check."""
    return [finding for finding in card.get("findings", []) if finding.get("quote_verified")]


async def extract_paper_card(
    source: dict,
    chunks: list[dict],
    focus_topic: str = "",
    semaphore: asyncio.Semaphore | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    if not chunks:
        return _empty_card(source, "No retrieved excerpts for this paper.")

    async def call() -> str:
        return await call_llm(
            prompt=_card_prompt(source, chunks, focus_topic),
            agent_role="literature_review",
            system="Extract only evidence grounded in the supplied excerpts. Return valid JSON only.",
            temperature=0.0,
            max_tokens=2400,
            progress_callback=progress_callback,
        )

    try:
        if semaphore is None:
            raw = await call()
        else:
            async with semaphore:
                raw = await call()
    except Exception as exc:
        return _empty_card(source, f"Evidence-card extraction failed: {exc}")
    card = clean_card(parse_json_object(raw), source, chunks)
    if not card["ok"]:
        card["error"] = "No finding had a verifiable evidence quote."
    return card


async def build_paper_cards(
    registry: list[dict],
    all_chunks: list[dict],
    focus_topic: str = "",
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Build one bounded evidence card for each selected source."""
    semaphore = asyncio.Semaphore(CARD_CONCURRENCY)
    by_source: dict[str, list[dict]] = {source["id"]: [] for source in registry}
    for chunk in all_chunks:
        if chunk.get("citation_id") in by_source:
            by_source[chunk["citation_id"]].append(chunk)

    async def extract_one(index: int, source: dict) -> tuple[int, dict]:
        if progress_callback:
            await progress_callback(
                f"Extracting evidence card {index}/{len(registry)}: {paper_label(source)}."
            )
        card = await extract_paper_card(
            source,
            by_source[source["id"]],
            focus_topic,
            semaphore,
            progress_callback,
        )
        if progress_callback:
            checked = card["quote_verification"]
            await progress_callback(
                f"Evidence card {index}/{len(registry)} ready: {paper_label(source)} "
                f"({checked['verified']}/{checked['total']} quotes matched)."
            )
        return index, card

    results = await asyncio.gather(*(
        extract_one(index, source)
        for index, source in enumerate(registry, start=1)
    ))
    return [card for _, card in sorted(results, key=lambda item: item[0])]


def format_cards_for_prompt(cards: list[dict], max_chars: int | None = None) -> str:
    blocks = []
    for card in cards:
        lines = [f"CARD [{card['paper_id']}] {card['label']} | type: {card.get('paper_type') or 'unknown'}"]
        for label, key in (
            ("Question", "research_question"),
            ("Method", "method"),
            ("Data/sample", "data_sample"),
        ):
            if card.get(key):
                lines.append(f"  {label}: {_clip(card[key], 180)}")
        if card.get("constructs"):
            lines.append("  Constructs: " + "; ".join(
                f"{_clip(key, 60)} = {_clip(value, 120)}"
                for key, value in list(card["constructs"].items())[:3]
            ))
        for finding in usable_findings(card)[:3]:
            page = f" (p.{finding['page']})" if finding.get("page") else ""
            lines.append(
                f"  Finding [{finding['attribution']}, {finding['strength']}, "
                f"{finding['direction']}]{page}: {_clip(finding['text'], 220)}"
            )
        lines.extend(
            f"  Stated limitation: {_clip(limitation, 160)}"
            for limitation in card.get("stated_limitations", [])[:2]
        )
        if not card.get("ok"):
            lines.append("  (No quote-verified card finding; use the paper excerpts directly.)")
        blocks.append("\n".join(lines))
    if max_chars is not None and blocks:
        per_card_budget = max(120, max_chars // len(blocks))
        bounded_blocks = []
        for block in blocks:
            lines = block.splitlines()
            kept = [lines[0][:per_card_budget]]
            used = len(kept[0])
            for line in lines[1:]:
                if used + len(line) + 1 > per_card_budget:
                    break
                kept.append(line)
                used += len(line) + 1
            bounded_blocks.append("\n".join(kept))
        blocks = bounded_blocks
    return "\n\n".join(blocks)


def _analysis_prompt(cards: list[dict], focus_topic: str, max_chars: int | None = None) -> str:
    focus = f'Review focus: "{focus_topic}".\n' if focus_topic else "No review focus was given.\n"
    return f"""Using ONLY these evidence cards, produce a cross-paper analysis. {focus}
Return JSON only:
{{
  "cohesion": "high | medium | low",
  "cohesion_note": "one sentence: do these papers address a shared question?",
  "themes": [{{"name": "...", "paper_ids": ["CR-..."], "summary": "one or two sentences"}}],
  "agreements": [{{"claim": "...", "paper_ids": ["CR-...", "CR-..."]}}],
  "differences": [{{"topic": "...", "paper_ids": ["CR-...", "CR-..."],
                    "reason": "different outcome | population | method | period | definition | other",
                    "comparable": true}}],
  "gaps": [{{"text": "...", "origin": "stated_by_authors | inferred_from_set", "paper_ids": ["CR-..."]}}]
}}

Rules:
- Provide 3 to {MAX_THEMES} themes when evidence permits. Themes are claims or debates, not paper summaries.
- Agreements require comparable measures. Set comparable=false for different outcomes, populations,
  definitions, or periods; these are complementary findings, not contradictions.
- Stated gaps must come from a card's stated limitations. Inferred gaps name a specific omission in this set.
- Use only paper IDs in the cards. A finding attributed to prior work is not the card's own evidence.

--- EVIDENCE CARDS ---
{format_cards_for_prompt(cards, max_chars=max_chars)}
--- END EVIDENCE CARDS ---"""


def _valid_ids(items: object, valid_ids: set[str]) -> list[str]:
    if not isinstance(items, list):
        return []
    return list(dict.fromkeys(str(item) for item in items if str(item) in valid_ids))


def clean_analysis(parsed: dict, valid_ids: set[str]) -> dict:
    if not parsed:
        return {}
    themes = []
    for item in (parsed.get("themes") if isinstance(parsed.get("themes"), list) else [])[:MAX_THEMES]:
        if isinstance(item, dict):
            ids = _valid_ids(item.get("paper_ids"), valid_ids)
            name = _clip(item.get("name"), 120)
            if name and ids:
                themes.append({"name": name, "paper_ids": ids, "summary": _clip(item.get("summary"), 400)})

    agreements = []
    for item in (parsed.get("agreements") if isinstance(parsed.get("agreements"), list) else [])[:6]:
        if isinstance(item, dict):
            ids = _valid_ids(item.get("paper_ids"), valid_ids)
            claim = _clip(item.get("claim"), 400)
            if len(ids) >= 2 and claim:
                agreements.append({"claim": claim, "paper_ids": ids})

    differences = []
    for item in (parsed.get("differences") if isinstance(parsed.get("differences"), list) else [])[:6]:
        if isinstance(item, dict):
            ids = _valid_ids(item.get("paper_ids"), valid_ids)
            topic = _clip(item.get("topic"), 300)
            if len(ids) >= 2 and topic:
                differences.append({
                    "topic": topic,
                    "paper_ids": ids,
                    "reason": _clip(item.get("reason"), 80),
                    "comparable": item.get("comparable") is True,
                })

    gaps = []
    for item in (parsed.get("gaps") if isinstance(parsed.get("gaps"), list) else [])[:6]:
        if isinstance(item, dict):
            text = _clip(item.get("text"), 400)
            ids = _valid_ids(item.get("paper_ids"), valid_ids)
            if text and ids:
                gaps.append({
                    "text": text,
                    "origin": _choice(item.get("origin"), GAP_ORIGINS, "inferred_from_set"),
                    "paper_ids": ids,
                })
    return {
        "cohesion": _choice(parsed.get("cohesion"), COHESION_LEVELS, "medium"),
        "cohesion_note": _clip(parsed.get("cohesion_note"), 300),
        "themes": themes,
        "agreements": agreements,
        "differences": differences,
        "gaps": gaps,
    }


async def analyze_cards(
    cards: list[dict],
    focus_topic: str = "",
    progress_callback: ProgressCallback | None = None,
) -> dict:
    if sum(bool(card.get("ok")) for card in cards) < 2:
        return {}
    prompt_budget = max(
        3_000,
        (settings.MODAL_CONTEXT - 4_000) * 2,
    )
    raw = await call_llm(
        prompt=_analysis_prompt(cards, focus_topic, max_chars=prompt_budget),
        agent_role="literature_review",
        system="You are a careful research analyst. Return valid JSON only.",
        temperature=0.1,
        max_tokens=2600,
        progress_callback=progress_callback,
    )
    return clean_analysis(parse_json_object(raw), {card["paper_id"] for card in cards})


def format_analysis_for_prompt(analysis: dict) -> str:
    if not analysis:
        return ""
    tags = lambda ids: "".join(f"[{paper_id}]" for paper_id in ids)
    lines = [
        "CROSS-PAPER ANALYSIS (planning aid; cite claims from the supplied excerpts)",
        f"Cohesion: {analysis.get('cohesion', 'unknown')}"
        + (f" — {analysis['cohesion_note']}" if analysis.get("cohesion_note") else ""),
    ]
    for section, key in (
        ("Candidate themes", "themes"),
        ("Agreements", "agreements"),
        ("Differences", "differences"),
        ("Candidate gaps", "gaps"),
    ):
        items = analysis.get(key) or []
        if not items:
            continue
        lines.append(section + ":")
        for item in items:
            if key == "themes":
                lines.append(f"- {item['name']}: {item['summary']} (papers: {tags(item['paper_ids'])})")
            elif key == "agreements":
                lines.append(f"- {item['claim']} (papers: {tags(item['paper_ids'])})")
            elif key == "differences":
                qualifier = "comparable" if item["comparable"] else "not comparable; complementary scope"
                lines.append(
                    f"- {item['topic']} ({item['reason'] or 'reason unspecified'}; {qualifier}; "
                    f"papers: {tags(item['paper_ids'])})"
                )
            else:
                origin = "stated by authors" if item["origin"] == "stated_by_authors" else "inferred from this set"
                lines.append(f"- [{origin}] {item['text']} (papers: {tags(item['paper_ids'])})")
    return "\n".join(lines)


def scope_note(analysis: dict, focus_topic: str) -> str:
    if analysis.get("cohesion") == "low" and not focus_topic:
        return (
            "The supplied papers address different questions. Say so plainly in the Introduction, organize "
            "themes around shared methodological or conceptual issues, and do not invent a common research "
            "question that the papers do not share."
        )
    return ""


def build_comparison_matrix(cards: list[dict]) -> list[dict]:
    rows = []
    for card in cards:
        findings = usable_findings(card)
        rows.append({
            "paper_id": card["paper_id"],
            "paper": card["label"],
            "paper_type": card.get("paper_type", ""),
            "research_question": card.get("research_question", ""),
            "method": card.get("method", ""),
            "data_sample": card.get("data_sample", ""),
            "key_findings": [finding["text"] for finding in findings[:3]],
            "evidence_strength": sorted({
                finding["strength"] for finding in findings if finding["strength"] != "unclear"
            }),
            "stated_limitations": card.get("stated_limitations", [])[:2],
            "quote_verification": card.get("quote_verification", {"verified": 0, "total": 0}),
        })
    return rows


def _sentences(text: str) -> list[str]:
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]


def _snippet(sentence: str, limit: int = 140) -> str:
    plain = _TAG.sub("", sentence).strip()
    return plain if len(plain) <= limit else plain[:limit - 3].rstrip() + "..."


def overclaim_hits(text: str) -> list[str]:
    return [_snippet(sentence) for sentence in _sentences(text) if _STRONG_TERMS.search(sentence)]


def citation_stack_hits(text: str, max_tags: int = 3) -> list[str]:
    return [_snippet(sentence) for sentence in _sentences(text) if len(_TAG.findall(sentence)) > max_tags]


def states_scope(text: str, n_papers: int) -> bool:
    for match in _SCOPE.finditer(text):
        token = match.group(1).lower()
        if (int(token) if token.isdigit() else _NUMBER_WORDS.get(token)) == n_papers:
            return True
    return False


def writing_problems(text: str, n_papers: int) -> list[str]:
    problems = []
    overclaims = overclaim_hits(text)
    if overclaims:
        problems.append(
            "Soften overclaimed wording; use cautious verbs unless the evidence supports causal language. "
            "Sentences: " + " | ".join(overclaims[:3])
        )
    stacks = citation_stack_hits(text)
    if stacks:
        problems.append(
            "Use no more than three [CR-...] citations per sentence and attribute clauses precisely: "
            + " | ".join(stacks[:3])
        )
    if not states_scope(text, n_papers):
        problems.append(
            f"State that this review synthesizes {n_papers} supplied "
            f"paper{'s' if n_papers != 1 else ''} and conclusions are limited to them."
        )
    return problems
