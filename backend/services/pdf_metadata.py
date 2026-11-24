from __future__ import annotations

import re
from typing import Any

import fitz


_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_ABSTRACT_END = re.compile(
    r"(?i)^\s*(?:keywords?\b|index terms\b|1\.?\s+introduction\b|introduction\b)"
)
_NON_AUTHOR = re.compile(
    r"(?i)\b(?:abstract|keywords?|university|department|institute|faculty|"
    r"school|laboratory|email|correspondence|doi|received|accepted|copyright|"
    r"journal|proceedings|volume|vol\.|issue|issn|www\.)\b|@|https?://"
)


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _page_lines(page: fitz.Page) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = _clean(" ".join(span.get("text", "") for span in spans))
            if text:
                lines.append({
                    "text": text,
                    "size": max((float(span.get("size", 0)) for span in spans), default=0),
                    "y": float(line.get("bbox", [0, 0, 0, 0])[1]),
                })
    return sorted(lines, key=lambda item: (item["y"], -item["size"]))


def _guess_title(lines: list[dict[str, Any]]) -> str:
    candidates = [
        (index, line)
        for index, line in enumerate(lines[:30])
        if len(line["text"]) >= 8
        and len(line["text"]) <= 240
        and not _NON_AUTHOR.search(line["text"])
    ]
    if not candidates:
        return ""
    largest_size = max(line["size"] for _, line in candidates)
    if largest_size <= 0:
        return candidates[0][1]["text"]
    _, first = min(
        (item for item in candidates if item[1]["size"] >= largest_size * 0.9),
        key=lambda item: item[1]["y"],
    )
    title_lines = [first["text"]]
    for _, line in candidates:
        if line["y"] <= first["y"] or line["y"] - first["y"] > 55:
            continue
        if line["size"] >= largest_size * 0.9:
            title_lines.append(line["text"])
    return _clean(" ".join(title_lines))


def _guess_authors(lines: list[dict[str, Any]], title: str) -> str:
    matching_title = next(
        (
            index for index, line in enumerate(lines)
            if title and title.casefold() in line["text"].casefold()
        ),
        None,
    )
    if matching_title is None:
        title_candidates = [
            (index, line)
            for index, line in enumerate(lines[:30])
            if len(line["text"]) >= 8
            and len(line["text"]) <= 240
            and not _NON_AUTHOR.search(line["text"])
        ]
        if not title_candidates:
            return ""
        largest_size = max(line["size"] for _, line in title_candidates)
        matching_title = next(
            (
                index for index, line in title_candidates
                if line["size"] >= largest_size * 0.9
            ),
            title_candidates[0][0],
        )
    start = matching_title + 1
    for line in lines[start:start + 12]:
        text = line["text"]
        if _NON_AUTHOR.search(text) or len(text) > 240:
            continue
        words = re.findall(r"[A-Z][A-Za-z'’-]+|[A-Z]\.", text)
        if 2 <= len(words) <= 16 and len(text.split()) <= 24:
            return text
    return ""


def _extract_abstract(lines: list[dict[str, Any]]) -> str:
    start = None
    content: list[str] = []
    for index, line in enumerate(lines):
        match = re.match(r"(?i)^\s*abstract\b\s*[:.–—-]?\s*(.*)$", line["text"])
        if match:
            start = index
            if match.group(1):
                content.append(match.group(1))
            continue
        if start is not None:
            if _ABSTRACT_END.search(line["text"]):
                break
            content.append(line["text"])
    return _clean(" ".join(content))[:10_000]


def _extract_keywords(lines: list[dict[str, Any]], pdf_keywords: str) -> list[str]:
    raw = pdf_keywords
    if not raw:
        for line in lines:
            match = re.match(r"(?i)^\s*(?:keywords?|index terms)\s*[:—-]\s*(.+)$", line["text"])
            if match:
                raw = match.group(1)
                break
    return list(dict.fromkeys(
        keyword.strip(" .;")
        for keyword in re.split(r"[,;]", raw)
        if keyword.strip(" .;")
    ))


def extract_pdf_metadata(file_bytes: bytes) -> dict[str, Any]:
    """Extract bibliographic fields and original PDF properties where available."""
    document = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        pdf_info = document.metadata or {}
        lines = _page_lines(document[0]) if len(document) else []
        first_page_text = document[0].get_text("text") if len(document) else ""
        pdf_title = _clean(pdf_info.get("title"))
        title = pdf_title or _guess_title(lines)
        authors = _clean(pdf_info.get("author")) or _guess_authors(lines, title)
        abstract = _extract_abstract(lines)
        doi_match = _DOI.search(first_page_text)
        abstract_line = next(
            (index for index, line in enumerate(lines) if re.match(r"(?i)^\s*abstract\b", line["text"])),
            len(lines),
        )
        publication_year = next(
            (
                int(match.group())
                for line in lines[:abstract_line + 1]
                if (match := _YEAR.search(line["text"]))
            ),
            None,
        )
        return {
            "title": title,
            "authors": authors,
            "year": publication_year,
            "abstract": abstract,
            "doi": doi_match.group().rstrip(".,;:") if doi_match else None,
            "keywords": _extract_keywords(lines, _clean(pdf_info.get("keywords"))),
            "subject": _clean(pdf_info.get("subject")),
            "pdf_metadata": {
                key: value
                for key in ("format", "creator", "producer", "creationDate", "modDate", "trapped")
                if isinstance((value := pdf_info.get(key)), (str, int, float, bool))
                and value
            },
        }
    finally:
        document.close()
