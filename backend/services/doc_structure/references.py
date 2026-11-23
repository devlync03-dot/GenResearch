from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

_REFERENCE_HEADING_NAMES = (
    r"references?(?:[ \t]+(?:list|and[ \t]+(?:bibliography|notes)))?"
    r"|bibliography(?:[ \t]+and[ \t]+references)?"
    r"|works[ \t]+cited"
    r"|literature[ \t]+cited"
)
_HEADING_RE = re.compile(
    rf"(?im)^[ \t]*(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)]?[ \t]+)?"
    rf"(?:{_REFERENCE_HEADING_NAMES})[ \t]*:?[ \t]*(?:\d{{1,3}})?[ \t]*$"
)
_REFERENCE_MARKER_RE = re.compile(r"(?m)^\s*(?:\[(\d+)\]|(\d+)\.|\((\d+)\))\s+")
_AUTHOR_YEAR_START_RE = re.compile(
    r"(?P<authors>"
    r"[A-ZÀ-ÖØ-Þ][\w'’\-]+,\s*[A-Z]\.(?:\s*[A-Z]\.)*"
    r"(?:,\s*(?:(?:&|and)\s*)?[A-ZÀ-ÖØ-Þ][\w'’\-]+,\s*[A-Z]\.(?:\s*[A-Z]\.)*)*"
    r"(?:\s+et\s+al\.?)?"
    r"|[A-ZÀ-ÖØ-Þ][\w'’\-]+"
    r"(?:\s+(?:Group|University|Institute|Company|Center|Centre|Department|Bureau|Organization|Organisation))"
    r"|[A-ZÀ-ÖØ-Þ][\w'’\-]+(?=\s*\((?:19|20)\d{2},\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\))"
    r")\s*\((?P<year>(?:19|20)\d{2}[a-z]?)"
    r"(?:,\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2})?\)",
)
_REF_START_PATTERN = re.compile(
    r"(?i)(?:\[\s*1\s*\]|\b1\.\s+|\(1\)|[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ'\-]+(?:\s+[A-Z]{1,3}|\s*,\s*[A-Z]{1,3}|,\s+[A-Z]\.?|\s+(?:and|&)\s+[A-Z]|\s+et\s+al\.)*.*?\(\d{4}[a-z]?\))"
)


@dataclass
class ReferenceEntry:
    ref_number: int | None = None
    raw: str = ""
    authors: str | None = None
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    parse_confidence: float = 0.0
    verified: bool = False
    verification_source: str | None = None


@dataclass
class ReferenceBundle:
    entries: list[ReferenceEntry] = field(default_factory=list)
    references_start_page: int | None = None
    reference_count: int = 0
    expected_reference_count: int = 0
    missing_reference_ids: list[int] = field(default_factory=list)


def _normalize_reference_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[A-Za-z])", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _find_reference_heading(text: str) -> tuple[int, str] | None:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return None
    for heading in reversed(matches):
        start = heading.end()
        window = text[start:start + 800]
        if _REF_START_PATTERN.search(window):
            return heading.start(), text[start:]
    # Fallback to the last match even if window is short
    heading = matches[-1]
    return heading.start(), text[heading.end():]


def _reference_start_follows_heading(
    page_texts: list[str],
    page_index: int,
    heading_end: int,
) -> bool:
    lookahead = [page_texts[page_index][heading_end:]]
    lookahead.extend(page_texts[page_index + 1:page_index + 3])
    return bool(_REF_START_PATTERN.search("\n".join(lookahead)[:2400]))


def _extract_ids_from_text(text: str) -> list[int]:
    ids = []
    for match in re.finditer(r"(?<![A-Za-z])\[(\d+)\](?![A-Za-z])", text):
        ids.append(int(match.group(1)))
    for match in re.finditer(r"(?m)^\s*(\d+)\.\s+", text):
        ids.append(int(match.group(1)))
    return ids


def _coerce_year(raw: str) -> int | None:
    match = re.search(r"(?<!\d)(19\d\d|20\d\d)(?!\d)", raw)
    if not match:
        return None
    return int(match.group(1))


def _coerce_doi(raw: str) -> str | None:
    # Strip zero-width characters only (not regular spaces)
    cleaned = raw.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    # Unwrap newlines that break up DOIs across lines in PDFs —
    # only join when the next line continues with DOI-like characters (not author names)
    cleaned = re.sub(r"(10\.\d{4,9}/[^\s\n]*)\n\s*([^\s\n]*[0-9/.\-]+[^\s\n]*)", r"\1\2", cleaned)
    match = re.search(r"10\.\d{4,9}/[^\s,;\"'<>]+", cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(0).rstrip(".")


def _coerce_arxiv(raw: str) -> str | None:
    cleaned = raw.replace("\u200b", "")
    match = re.search(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    return f"arXiv:{match.group(1)}"


def _strip_ref_noise(raw: str) -> str:
    raw = re.sub(r"(?i)\b(doi|arxiv|url|available at|accessed on).*?$", "", raw)
    return raw.strip(" -•\n")


def _parse_reference_entry(raw: str) -> ReferenceEntry:
    cleaned = _normalize_reference_text(raw)
    entry = ReferenceEntry(raw=cleaned)
    match = re.search(r"^\s*(?:\[(\d+)\]|(\d+)\.|\((\d+)\))\s*", cleaned)
    if match:
        num_str = match.group(1) or match.group(2) or match.group(3)
        entry.ref_number = int(num_str)
        cleaned = cleaned[match.end():].lstrip()
    entry.year = _coerce_year(cleaned)
    entry.doi = _coerce_doi(cleaned)
    entry.arxiv_id = _coerce_arxiv(cleaned)
    entry.authors = None
    entry.title = None
    year_match = re.search(r"\((?:19|20)\d{2}[a-z]?\)", cleaned, re.IGNORECASE)
    if year_match:
        entry.authors = cleaned[:year_match.start()].strip(" .,:;")
        title_and_venue = cleaned[year_match.end():].strip(" .,:;")
        title_parts = re.split(r"\.\s+(?=[A-Z])", title_and_venue, maxsplit=1)
        if title_parts and title_parts[0]:
            entry.title = title_parts[0].strip(" .\"'")
            if len(title_parts) > 1:
                entry.venue = title_parts[1].strip(" .") or None
    title_match = re.search(r"\s+([A-Z][^.;]+(?:\.[^.;]+)*)\.?\s*(?:\.|\b(?:In|Proc\.|Proceedings|Journal|Transactions|Conference|ACM|IEEE|arXiv|doi|URL))", cleaned)
    if title_match and not entry.title:
        entry.title = title_match.group(1).strip(" .")
        entry.authors = cleaned[:title_match.start()].strip(" .")
    if not entry.authors:
        author_bits = re.split(r"\s+(?:[A-Z][^\n]+?\.\s+(?:[A-Z][^\n]+?\.)?)\s+", cleaned, maxsplit=1)
        if len(author_bits) > 1 and author_bits[0]:
            entry.authors = author_bits[0].strip(" .")
    if not entry.title:
        entry.title = re.sub(r"^.*?\s+(?=[A-Z][^\n]+\.)", "", cleaned, count=1)[:200]
    if entry.title and entry.venue is None:
        rest = cleaned.split(entry.title, 1)[1] if entry.title in cleaned else ""
        if rest:
            venue_candidate = re.sub(r"^\s*[-:;,.\s]+", "", rest)
            entry.venue = venue_candidate[:200].strip(" .") or None
    entry.parse_confidence = 0.5
    if entry.year is not None:
        entry.parse_confidence += 0.2
    if entry.title:
        entry.parse_confidence += 0.2
    if entry.doi or entry.arxiv_id:
        entry.parse_confidence += 0.1
    entry.parse_confidence = min(1.0, entry.parse_confidence)
    return entry


def _split_numbered_entries(text: str) -> list[str]:
    cleaned = _normalize_reference_text(text)
    matches = list(_REFERENCE_MARKER_RE.finditer(cleaned))
    if not matches:
        return []

    # Sequence tracking: filter out false marker lines (e.g., page/vol numbers like 591., 2015.)
    seq_matches: list[tuple[int, int]] = []
    expected = 1
    for m in matches:
        n = int(m.group(1) or m.group(2) or m.group(3))
        if not seq_matches:
            if n <= 5:
                seq_matches.append((n, m.start()))
                expected = n + 1
        else:
            if n == expected:
                seq_matches.append((n, m.start()))
                expected += 1
            elif n > expected and n <= expected + 2:
                seq_matches.append((n, m.start()))
                expected = n + 1

    if len(seq_matches) < 2:
        return []
    chosen_matches = seq_matches

    entries: list[str] = []
    for idx in range(len(chosen_matches)):
        start = chosen_matches[idx][1]
        end = chosen_matches[idx + 1][1] if idx + 1 < len(chosen_matches) else len(cleaned)
        segment = cleaned[start:end].strip()
        if segment:
            entries.append(segment)
    return entries


def _extract_reference_entries_from_text(text: str) -> list[ReferenceEntry]:
    cleaned = _normalize_reference_text(text)
    numbered_entries = [
        _parse_reference_entry(segment)
        for segment in _split_numbered_entries(cleaned)
        if segment.strip()
    ]

    author_year_matches = list(_AUTHOR_YEAR_START_RE.finditer(cleaned))
    author_year_entries: list[ReferenceEntry] = []
    if len(author_year_matches) >= 2:
        for idx, match in enumerate(author_year_matches):
            start = match.start()
            end = (
                author_year_matches[idx + 1].start()
                if idx + 1 < len(author_year_matches)
                else len(cleaned)
            )
            segment = cleaned[start:end].strip()
            if segment:
                author_year_entries.append(_parse_reference_entry(segment))

    # PDF text extraction can surface page or DOI numbers as apparent numbered
    # entries. Prefer author-year boundaries when they recover more real entries.
    if len(author_year_entries) > len(numbered_entries):
        return author_year_entries
    if numbered_entries:
        return numbered_entries

    # Recovery pass for inline markers or glued references
    recovered: list[ReferenceEntry] = []
    marker_pattern = re.compile(
        r"(?:\[(\d+)\]|^\s*(\d+)\.)\s*([^\[\n]+(?:\n(?!\s*(?:\[\d+\]|\d+\.))[^\[\n]+)*)",
        re.MULTILINE,
    )
    for match in marker_pattern.finditer(cleaned):
        raw = match.group(0).strip()
        if raw:
            recovered.append(_parse_reference_entry(raw))
    if recovered:
        return recovered

    # Line-oriented fallback for styles without initials immediately before the year.
    author_year_re = re.compile(
        r"(?m)^\s*([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ'\-]+(?:\s+[A-Z]{1,3}|\s*,\s*[A-Z]{1,3}|,\s+[A-Z]\.?|\s+(?:and|&)\s+[A-Z]|\s+et\s+al\.)*.*?\(\d{4}[a-z]?\))"
    )
    ay_matches = list(author_year_re.finditer(cleaned))
    if len(ay_matches) >= 2:
        for idx, m in enumerate(ay_matches):
            start = m.start()
            end = ay_matches[idx + 1].start() if idx + 1 < len(ay_matches) else len(cleaned)
            segment = cleaned[start:end].strip()
            if segment:
                recovered.append(_parse_reference_entry(segment))
    return recovered


def parse_reference_entries(reference_text: str) -> ReferenceBundle:
    text = _normalize_reference_text(reference_text)
    heading_info = _find_reference_heading(text)
    if heading_info:
        _, reference_body = heading_info
        text = reference_body
    ids: list[int] = []
    entries = _extract_reference_entries_from_text(text)
    for entry in entries:
        if entry.ref_number is not None:
            ids.append(entry.ref_number)
    if not ids:
        ids = _extract_ids_from_text(text)
    expected = max(ids) if ids else len(entries)
    seen = {entry.ref_number for entry in entries if entry.ref_number is not None}
    missing = sorted({n for n in range(1, expected + 1) if n not in seen}) if seen else []
    return ReferenceBundle(
        entries=entries,
        reference_count=len(entries),
        expected_reference_count=expected,
        missing_reference_ids=missing,
    )


def find_reference_section(pages: Iterable[str]) -> tuple[int | None, str]:
    page_texts = list(pages)
    last_heading: tuple[int, int] | None = None
    last_heading_with_entries: tuple[int, int] | None = None
    for idx, page in enumerate(page_texts):
        matches = list(_HEADING_RE.finditer(page))
        for heading in matches:
            pos = heading.end()
            last_heading = (idx, pos)
            if _reference_start_follows_heading(page_texts, idx, pos):
                last_heading_with_entries = (idx, pos)

    section_start = last_heading_with_entries or last_heading
    if section_start is None:
        return None, ""

    idx, pos = section_start
    combined = [page_texts[idx][pos:]]
    for following_page in page_texts[idx + 1:]:
        appendix_match = re.search(r"(?im)^[ \t]*(?:Appendix|Appendices|Index)\b", following_page)
        if appendix_match:
            combined.append(following_page[:appendix_match.start()])
            break
        combined.append(following_page)
    return (idx + 1, "\n".join(combined))


def parse_reference_entries_from_pages(pages: list[str]) -> ReferenceBundle:
    ref_page, ref_text = find_reference_section(pages)
    if not ref_text:
        return ReferenceBundle(references_start_page=ref_page, entries=[], reference_count=0, expected_reference_count=0, missing_reference_ids=[])
    bundle = parse_reference_entries(ref_text)
    bundle.references_start_page = ref_page
    return bundle
