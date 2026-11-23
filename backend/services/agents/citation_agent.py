from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Literal
from urllib.parse import quote

import aiohttp

from database.supabase_client import get_supabase
from services.agents.json_utils import parse_llm_json
from services.doc_structure.pipeline import extract_structure, persist_structure
from services.doc_structure.references import (
    ReferenceEntry,
    find_reference_section,
    parse_reference_entries,
)
from services.llm_service import call_llm
from services.rag_service import semantic_search
from services.source_gathering import (
    search_arxiv,
    search_crossref,
    search_openalex,
    search_semantic_scholar,
)
from services import tracking
from services.text_utils import split_sentences

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

_SOURCE_CACHE_TTL_SECONDS = 30 * 60
_SOURCE_CACHE_MAX_ENTRIES = 500
_source_cache: dict[str, tuple[float, Any]] = {}


def _cache_lookup(key: str) -> tuple[bool, Any]:
    cached = _source_cache.get(key)
    if cached is None:
        return False, None
    expires_at, value = cached
    if expires_at <= time.time():
        _source_cache.pop(key, None)
        return False, None
    return True, value


def _cache_store(key: str, value: Any) -> None:
    if key not in _source_cache and len(_source_cache) >= _SOURCE_CACHE_MAX_ENTRIES:
        now = time.time()
        for expired_key in [k for k, (expires, _) in _source_cache.items() if expires <= now]:
            _source_cache.pop(expired_key, None)
        if len(_source_cache) >= _SOURCE_CACHE_MAX_ENTRIES:
            oldest_key = min(_source_cache, key=lambda cache_key: _source_cache[cache_key][0])
            _source_cache.pop(oldest_key, None)
    _source_cache[key] = (time.time() + _SOURCE_CACHE_TTL_SECONDS, value)


# --------------------------------------------------------------------------- #
# Constants and regular expressions
# --------------------------------------------------------------------------- #

CITATION_STYLES = {"apa", "mla", "ieee", "chicago"}
_CLAIM_VERDICTS = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "source_not_found",
}

_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_TABLE = str.maketrans(_SUPERSCRIPTS, "0123456789")

# A capitalised name (case-sensitive even though the surrounding patterns are IGNORECASE).
_NAME = r"(?-i:[A-Z])[A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
_YEAR = r"(?:1[5-9]|20)\d{2}[a-z]?"
_NUMBER_LIST = r"\d{1,3}(?:\s*[,;\-–—]\s*\d{1,3})*"
_DOI = r"10\.\d{4,9}/[A-Z0-9._;()/:-]+"
# Content of "(Smith, 2020)", "(see Smith & Lee, 2019; Jones, 2018)" - needs a capitalised name.
_PAREN_AUTHOR_YEAR = (
    rf"(?=[^()]*{_NAME})[^()]{{1,160}}?,?\s+{_YEAR}[^()]{{0,60}}"
)
_NARRATIVE = (
    rf"\b{_NAME}(?:\s+(?:&|and)\s+{_NAME})?(?:\s+et\s+al\.?)?\s+\({_YEAR}\)"
)

_CITATION_MARKER = re.compile(
    rf"\[(?:CR-[A-F0-9]{{12}}|{_NUMBER_LIST})\]"
    r"|\[\^\d+\]"
    rf"|[{_SUPERSCRIPTS}]"
    rf"|\({_PAREN_AUTHOR_YEAR}\)"
    rf"|{_NARRATIVE}"
    rf"|{_DOI}",
    re.IGNORECASE,
)
_INLINE_CITATION_RE = re.compile(
    r"\[(?P<source_id>CR-[A-F0-9]{12})\]"
    rf"|\[(?P<numbers>{_NUMBER_LIST})\]"
    r"|\[\^(?P<footnote>\d+)\]"
    rf"|(?P<superscript>[{_SUPERSCRIPTS}]+)"
    rf"|\((?P<author>{_PAREN_AUTHOR_YEAR})\)"
    rf"|(?P<narrative>{_NARRATIVE})"
    rf"|(?P<doi>{_DOI})",
    re.IGNORECASE,
)
_ADJACENT_CITATION = re.compile(
    rf"[ \t]*(\[(?:CR-[A-F0-9]{{12}}|{_NUMBER_LIST})\]|\[\^\d+\]|[{_SUPERSCRIPTS}]+"
    rf"|\({_PAREN_AUTHOR_YEAR}\)|{_DOI})",
    re.IGNORECASE,
)
_EVIDENCE_CLAIM_CUE = re.compile(
    r"\b(?:"
    r"according to|prior research|previous studies|research (?:finds|shows|suggests|reports)|"
    r"stud(?:y|ies) (?:found|finds|reported|reports|showed|shows|suggested|suggests|"
    r"demonstrated|indicated|concluded)|"
    r"participants|respondents|sample of|surveyed|experiment(?:al)?|trial|"
    r"significant(?:ly)?|correlat(?:ed|ion)|associated with|increased|decreased|"
    r"improved|reduced|effect of|impact of|evidence (?:shows|suggests|indicates)|"
    r"defined as|refers to|was found to|were found to|has been shown"
    r")\b",
    re.IGNORECASE,
)
_AUTHOR_INTRODUCTION = re.compile(
    r"^\s*(?:this|the present)\s+(?:study|paper|research|article)\s+"
    r"(?:aims?|seeks?|intends?|proposes?|introduces?|examines?|investigates?|"
    r"uses?|employs?|is organized|is structured)\b",
    re.IGNORECASE,
)
_REFERENCE_PREFIX = re.compile(r"^(?:\[\d+\]|\d+\.|\(\d+\))\s*")
_REFERENCE_HEADING = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?"
    r"(?:references|reference list|bibliography|works cited|literature cited)\s*:?\s*$"
)
_NAME_WORDS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+")
_IGNORED_SURNAMES = {"unknown", "none", "n", "anonymous"}
_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_ARXIV_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _clean_ref_string(raw: str) -> str:
    cleaned = raw.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    cleaned = re.sub(r"(https?://\S*?)\n([^\s\n]+)", r"\1\2", cleaned)
    cleaned = re.sub(r"(10\.\d{4,9}/\S*?)\n([^\s\n]+)", r"\1\2", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_match(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _year_int(value: object) -> int | None:
    match = re.search(r"\d{4}", str(value or ""))
    return int(match.group(0)) if match else None


def _clean_doi(value: object) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = doi.rstrip(".,;:")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip(".,;:")
    return doi


def _title_similarity(first: object, second: object) -> float:
    a, b = _normalize_match(first), _normalize_match(second)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _titles_match(first: object, second: object, threshold: float = 0.88) -> bool:
    a, b = _normalize_match(first), _normalize_match(second)
    if not a or not b:
        return False
    if SequenceMatcher(None, a, b).ratio() >= threshold:
        return True
    shorter, longer = sorted((a, b), key=len)
    # Tolerate a dropped subtitle ("Title: a subtitle" vs "Title").
    return len(shorter) >= 25 and longer.startswith(shorter)


def _crossref_headers() -> dict[str, str]:
    mailto = os.getenv("CROSSREF_MAILTO", "research@genresearch.dev")
    return {"User-Agent": f"GenResearch/0.2.0 (mailto:{mailto})"}


# --------------------------------------------------------------------------- #
# Author handling
# --------------------------------------------------------------------------- #

def _split_authors(text: object) -> list[str]:
    """Split one author string into individual authors (APA, IEEE, Vancouver, plain)."""
    cleaned = re.sub(r"\bet\s+al\.?", "", str(text or ""), flags=re.IGNORECASE).strip(" ,;")
    if not cleaned:
        return []
    parts = re.split(r"\s*;\s*|(?<=\.),\s+|\s*&\s*|\s+and\s+", cleaned)
    authors = []
    for part in parts:
        part = re.sub(r"^\s*and\s+", "", part).strip(" ,;")
        if part:
            authors.append(part)
    expanded: list[str] = []
    for author in authors:
        pieces = [piece.strip() for piece in author.split(",") if piece.strip()]
        # "J. Smith, K. Jones" -> two authors; "Smith, John" stays one author.
        if len(pieces) > 1 and all(len(piece.split()) >= 2 for piece in pieces):
            expanded.extend(pieces)
        else:
            expanded.append(author)
    return expanded


def _surname_from_author(author: str) -> str | None:
    if "," in author:
        words = _NAME_WORDS.findall(author.split(",", 1)[0])
        return words[-1].strip("'’-") if words else None
    words = _NAME_WORDS.findall(author)
    # "Smith JK" (Vancouver): drop trailing initials.
    while len(words) > 1 and re.fullmatch(r"[A-Z]{1,3}", words[-1]):
        words.pop()
    return words[-1].strip("'’-") if words else None


def _author_surnames(value: object) -> list[str]:
    items = list(value) if isinstance(value, (list, tuple, set)) else [value]
    surnames: list[str] = []
    for item in items:
        for author in _split_authors(item):
            surname = _surname_from_author(author)
            if surname and len(surname) > 1 and surname.casefold() not in _IGNORED_SURNAMES:
                surnames.append(surname.casefold())
    return surnames


def _authors_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(author).strip() for author in value if str(author or "").strip()]
    return _split_authors(value)


def _authors_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_authors_list(value))
    return str(value or "").strip()


def _citation_surnames(text: str) -> list[str]:
    """Surnames from the author part of an in-text citation like 'Smith & Lee'."""
    text = re.sub(r"\bet\s+al\.?", " ", text, flags=re.IGNORECASE)
    surnames: list[str] = []
    for part in re.split(r"\s*(?:&|,|;|\band\b)\s*", text):
        words = _NAME_WORDS.findall(part)
        if not words:
            continue
        word = words[-1].strip("'’-")
        if len(word) > 1 and word[0].isupper():
            surnames.append(word.casefold())
    return surnames


# --------------------------------------------------------------------------- #
# Crossref / DOI / arXiv lookups
# --------------------------------------------------------------------------- #

def _crossref_metadata(message: dict[str, Any]) -> dict[str, Any]:
    dates = message.get("published-print") or message.get("issued") or message.get("published-online") or {}
    date_parts = dates.get("date-parts") or []
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    authors = []
    for author in message.get("author") or []:
        name = " ".join(part for part in (author.get("given"), author.get("family")) if part)
        name = name or str(author.get("name") or "").strip()
        if name:
            authors.append(name)
    titles = message.get("title") or []
    containers = message.get("container-title") or []
    update_items = list(message.get("updated-by") or []) + list(message.get("update-to") or [])
    relation = message.get("relation") or {}
    is_retracted = (
        any("retract" in str(update.get("type", "")).casefold() for update in update_items)
        or any("retract" in str(kind).casefold() for kind in relation)
        or bool(titles and str(titles[0]).strip().casefold().startswith("retracted"))
    )
    abstract = re.sub(
        r"\s+",
        " ",
        re.sub(r"<[^>]+>", " ", html.unescape(str(message.get("abstract") or ""))),
    ).strip()
    return {
        "title": titles[0] if titles else None,
        "authors": authors,
        "year": year,
        "venue": containers[0] if containers else None,
        "doi": message.get("DOI"),
        "url": message.get("URL"),
        "citation_count": message.get("is-referenced-by-count"),
        "is_retracted": is_retracted,
        "retraction_status": "retracted" if is_retracted else "not_indicated",
        "abstract_excerpt": abstract[:1500],
    }


async def _lookup_crossref_doi(doi: str) -> dict[str, Any] | None:
    clean_doi = _clean_doi(doi)
    if not clean_doi:
        return None
    cache_key = f"doi:{clean_doi.casefold()}"
    cached, value = _cache_lookup(cache_key)
    if cached:
        return dict(value) if value else None
    url = "https://api.crossref.org/works/" + quote(clean_doi, safe="/")
    timeout = aiohttp.ClientTimeout(total=8)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=_crossref_headers()) as response:
                if response.status == 404:
                    _cache_store(cache_key, None)
                    return None
                if response.status != 200:
                    raise RuntimeError(f"Crossref DOI lookup returned HTTP {response.status}.")
                data = await response.json(content_type=None)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise RuntimeError("Crossref DOI lookup returned invalid metadata.")
        metadata = _crossref_metadata(message)
        _cache_store(cache_key, metadata)
        return dict(metadata)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
        logger.info("citation_doi_lookup_failed", extra={"error": str(error)})
        raise RuntimeError("Crossref DOI metadata lookup is unavailable.") from error


async def _doi_is_registered(doi: str) -> bool | None:
    """Ask the DOI proxy whether a DOI exists at all (covers DataCite, arXiv DOIs, etc.)."""
    clean_doi = _clean_doi(doi)
    if not clean_doi:
        return False
    cache_key = f"doi-registered:{clean_doi.casefold()}"
    cached, value = _cache_lookup(cache_key)
    if cached:
        return value
    timeout = aiohttp.ClientTimeout(total=8)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                "https://doi.org/api/handle/" + quote(clean_doi, safe="/"),
                headers=_crossref_headers(),
            ) as response:
                if response.status == 404:
                    _cache_store(cache_key, False)
                    return False
                if response.status != 200:
                    return None
                data = await response.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return None
    code = data.get("responseCode") if isinstance(data, dict) else None
    if code == 1:
        _cache_store(cache_key, True)
        return True
    if code == 100:
        _cache_store(cache_key, False)
        return False
    return None


async def _classify_unresolved_doi(doi: str) -> tuple[str, bool | None]:
    """Crossref has no record: decide between 'does not exist' and 'registered elsewhere'."""
    registered = await _doi_is_registered(doi)
    if registered is False:
        return "doi_not_found", False
    return "manual_review", registered


def _scholarly_candidate_metadata(candidate: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "title": str(candidate.get("title") or ""),
        "authors": _authors_list(candidate.get("authors")),
        "year": candidate.get("year"),
        "venue": candidate.get("venue"),
        "doi": candidate.get("doi"),
        "url": candidate.get("url"),
        "citation_count": candidate.get("citation_count"),
        "is_retracted": False,
        "retraction_status": "not_checked",
        "abstract_excerpt": candidate.get("abstract_snippet") or "",
        "metadata_source": source,
    }


def _candidate_matches_reference(
    ref: ReferenceEntry,
    candidate_title: object,
    candidate_authors: object,
    threshold: float = 0.92,
) -> bool:
    if _title_similarity(ref.title, candidate_title) < threshold:
        return False
    ref_authors = set(_author_surnames(ref.authors))
    found_authors = set(_author_surnames(candidate_authors))
    if ref_authors and found_authors and not ref_authors & found_authors:
        return False
    return True


async def _lookup_crossref_title(ref: ReferenceEntry) -> dict[str, Any] | None:
    if not ref.title:
        return None
    cache_key = "title:" + _normalize_match(ref.title)
    cached, value = _cache_lookup(cache_key)
    if cached:
        return dict(value) if value else None

    crossref_ok = True
    try:
        response = await search_crossref(ref.title, limit=3)
        if response.get("status") not in {None, "success"}:
            crossref_ok = False
    except Exception as error:
        logger.info("citation_title_lookup_failed", extra={"error": str(error)})
        response = {"status": "error", "results": []}
        crossref_ok = False
    for candidate in response.get("results") or []:
        if not _candidate_matches_reference(ref, candidate.get("title"), candidate.get("authors")):
            continue
        metadata = _scholarly_candidate_metadata(candidate, "crossref")
        _cache_store(cache_key, metadata)
        return dict(metadata)

    responses = await asyncio.gather(
        search_semantic_scholar(ref.title, limit=3),
        search_openalex(ref.title, limit=3),
        search_arxiv(ref.title, limit=3),
        return_exceptions=True,
    )
    successful = [
        item for item in responses
        if not isinstance(item, Exception) and item.get("status") in {None, "success"}
    ]
    if not successful and not crossref_ok:
        raise RuntimeError("Crossref and fallback scholarly metadata providers are unavailable.")
    for item in successful:
        for candidate in item.get("results") or []:
            if not _candidate_matches_reference(ref, candidate.get("title"), candidate.get("authors")):
                continue
            metadata = _scholarly_candidate_metadata(
                candidate, candidate.get("source_api", "scholarly_api")
            )
            _cache_store(cache_key, metadata)
            return dict(metadata)
    if crossref_ok:
        # Never cache a miss that was caused by a provider outage.
        _cache_store(cache_key, None)
    return None


def _normalize_arxiv_id(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _arxiv_base_id(value: object) -> str:
    return re.sub(r"v\d+$", "", _normalize_arxiv_id(value), flags=re.IGNORECASE)


async def _fetch_arxiv_entry(arxiv_id: str) -> dict[str, Any] | None:
    """Direct arXiv ID lookup (authoritative; a text search is not)."""
    url = (
        "https://export.arxiv.org/api/query?id_list="
        + quote(arxiv_id, safe="/")
        + "&max_results=1"
    )
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=_crossref_headers()) as response:
                if response.status != 200:
                    raise RuntimeError(f"arXiv lookup returned HTTP {response.status}.")
                body = await response.text()
        root = ET.fromstring(body)
    except (aiohttp.ClientError, asyncio.TimeoutError, ET.ParseError) as error:
        logger.info("citation_arxiv_lookup_failed", extra={"error": str(error)})
        raise RuntimeError("arXiv metadata lookup is unavailable.") from error
    entry = root.find(f"{_ARXIV_ATOM}entry")
    if entry is None:
        return None
    entry_url = (entry.findtext(f"{_ARXIV_ATOM}id") or "").strip()
    title = re.sub(r"\s+", " ", entry.findtext(f"{_ARXIV_ATOM}title") or "").strip()
    if "/api/errors" in entry_url or title.casefold() == "error":
        return None
    published = (entry.findtext(f"{_ARXIV_ATOM}published") or "").strip()
    authors = [
        re.sub(r"\s+", " ", author.findtext(f"{_ARXIV_ATOM}name") or "").strip()
        for author in entry.findall(f"{_ARXIV_ATOM}author")
    ]
    summary = re.sub(r"\s+", " ", entry.findtext(f"{_ARXIV_ATOM}summary") or "").strip()
    return {
        "title": title,
        "authors": [author for author in authors if author],
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "venue": (entry.findtext(f"{_ARXIV_NS}journal_ref") or "").strip() or None,
        "doi": (entry.findtext(f"{_ARXIV_NS}doi") or "").strip() or None,
        "url": entry_url or None,
        "abstract_excerpt": summary[:1500],
    }


async def _lookup_arxiv_id(ref: ReferenceEntry) -> dict[str, Any] | None:
    arxiv_id = _arxiv_base_id(ref.arxiv_id)
    if not arxiv_id:
        return None
    cache_key = f"arxiv:{arxiv_id.casefold()}"
    cached, value = _cache_lookup(cache_key)
    if cached:
        return dict(value) if value else None
    entry = await _fetch_arxiv_entry(arxiv_id)
    if entry is None:
        _cache_store(cache_key, None)
        return None
    metadata: dict[str, Any] = {
        "title": entry["title"],
        "authors": entry["authors"],
        "year": entry["year"],
        "venue": entry["venue"],
        "doi": entry["doi"],
        "url": entry["url"],
        "citation_count": None,
        "is_retracted": False,
        "retraction_status": "not_checked",
        "abstract_excerpt": entry["abstract_excerpt"],
        "metadata_source": "arxiv",
        "published_version": None,
        "published_version_status": "not_checked",
    }
    if entry["title"]:
        crossref_ok = True
        try:
            published_response = await search_crossref(entry["title"], limit=3)
            if published_response.get("status") not in {None, "success"}:
                crossref_ok = False
        except Exception as error:
            logger.info("citation_published_version_lookup_failed", extra={"error": str(error)})
            published_response = {"results": []}
            crossref_ok = False
        for published in published_response.get("results") or []:
            if _title_similarity(entry["title"], published.get("title")) < 0.94:
                continue
            if not published.get("doi"):
                continue
            published_year = _year_int(published.get("year"))
            if entry["year"] and published_year and published_year < entry["year"]:
                continue
            metadata["published_version"] = {
                "title": published.get("title"),
                "doi": published.get("doi"),
                "year": published.get("year"),
                "url": published.get("url"),
            }
            metadata["published_version_status"] = "possible_published_version_found"
            break
        else:
            if crossref_ok:
                metadata["published_version_status"] = "no_crossref_match_found"
    _cache_store(cache_key, metadata)
    return dict(metadata)


# --------------------------------------------------------------------------- #
# Reference verification and formatting
# --------------------------------------------------------------------------- #

def _format_reference_entry(ref: ReferenceEntry, idx: int, style: str) -> str:
    """Keep the original bibliography text; append only a DOI parsed from that text."""
    cleaned_body = _REFERENCE_PREFIX.sub("", _clean_ref_string(ref.raw)).strip()
    doi_suffix = ""
    if ref.doi and ref.doi.casefold() not in cleaned_body.casefold():
        doi_suffix = f" https://doi.org/{ref.doi}"
    if style.lower() == "ieee":
        return f"[{idx}] {cleaned_body}{doi_suffix}".strip()
    return f"{idx}. {cleaned_body}{doi_suffix}".strip()


async def _verify_reference(ref: ReferenceEntry) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "authors": ref.authors,
        "title": ref.title,
        "year": ref.year,
        "venue": ref.venue,
        "doi": ref.doi,
        "parse_confidence": ref.parse_confidence,
        "verification_status": "not_checked",
        "metadata_source": "uploaded_reference_text",
        "doi_resolves": None,
        "metadata_matches": None,
        "retraction_status": "unknown",
        "citation_count": None,
        "abstract_excerpt": "",
        "published_version": None,
        "published_version_status": "not_checked",
    }
    try:
        if ref.doi:
            canonical = await _lookup_crossref_doi(ref.doi)
        elif ref.arxiv_id:
            canonical = await _lookup_arxiv_id(ref)
        else:
            canonical = await _lookup_crossref_title(ref)
    except Exception as error:
        logger.info("citation_reference_lookup_failed", extra={"error": str(error)})
        parsed.update({
            "verification_status": "manual_review",
            "resolution_error": (
                str(error) if isinstance(error, RuntimeError)
                else "Reference metadata lookup is unavailable."
            ),
        })
        return parsed

    if canonical is None:
        if ref.doi:
            status, resolves = await _classify_unresolved_doi(ref.doi)
            parsed["verification_status"] = status
            parsed["doi_resolves"] = resolves
        elif ref.arxiv_id:
            parsed["verification_status"] = "arxiv_not_found"
        else:
            parsed["verification_status"] = "manual_review"
        return parsed

    title_match = (
        not ref.title
        or not canonical.get("title")
        or _titles_match(ref.title, canonical.get("title"))
    )
    extracted_authors = set(_author_surnames(ref.authors))
    canonical_authors = set(_author_surnames(canonical.get("authors")))
    author_match = not extracted_authors or not canonical_authors or bool(
        extracted_authors & canonical_authors
    )
    ref_year = _year_int(ref.year)
    canonical_year = _year_int(canonical.get("year"))
    # Online-first vs print year commonly differs by one.
    year_match = ref_year is None or canonical_year is None or abs(ref_year - canonical_year) <= 1
    metadata_matches = title_match and author_match and year_match

    parsed["doi_resolves"] = True if ref.doi else None
    parsed["metadata_matches"] = metadata_matches
    if not metadata_matches:
        # Canonical data belongs to a different work; do not attach it.
        parsed["verification_status"] = "metadata_mismatch"
        return parsed

    canonical_authors_list = _authors_list(canonical.get("authors"))
    parsed.update({
        "authors": ", ".join(canonical_authors_list) or ref.authors,
        "title": canonical.get("title") or ref.title,
        "year": canonical.get("year") or ref.year,
        "venue": canonical.get("venue") or ref.venue,
        "doi": canonical.get("doi") or ref.doi,
        "metadata_source": canonical.get("metadata_source") or "crossref",
        "retraction_status": canonical.get("retraction_status", "not_checked"),
        "citation_count": canonical.get("citation_count"),
        "abstract_excerpt": canonical.get("abstract_excerpt") or "",
        "published_version": canonical.get("published_version"),
        "published_version_status": canonical.get("published_version_status", "not_checked"),
        "verification_status": "retracted" if canonical.get("is_retracted") else "verified",
    })
    return parsed


def _reference_metadata_without_lookup(ref: ReferenceEntry) -> dict[str, Any]:
    return {
        "authors": ref.authors,
        "title": ref.title,
        "year": ref.year,
        "venue": ref.venue,
        "doi": ref.doi,
        "parse_confidence": ref.parse_confidence,
        "verification_status": "not_checked",
        "metadata_source": "uploaded_reference_text",
        "doi_resolves": None,
        "metadata_matches": None,
        "retraction_status": "not_checked",
        "citation_count": None,
        "abstract_excerpt": "",
        "published_version": None,
        "published_version_status": "not_checked",
    }


def _reference_key(ref: ReferenceEntry) -> str:
    if ref.doi:
        return "doi:" + _normalize_match(_clean_doi(ref.doi))
    normalized = _normalize_match(ref.title or ref.raw)
    if normalized:
        return "title:" + normalized
    # Never empty and never shared between two unparsed entries.
    return f"ref:{ref.ref_number}:{id(ref)}"


def _deduplicate_references(references: list[ReferenceEntry]) -> list[ReferenceEntry]:
    found: set[str] = set()
    unique = []
    for reference in references:
        key = _reference_key(reference)
        if key in found:
            continue
        found.add(key)
        unique.append(reference)
    return unique


def _escape_bibtex(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _bibtex_authors(value: object) -> str:
    authors = _authors_list(value)
    cleaned = [_escape_bibtex(author) for author in authors if author]
    if not isinstance(value, (list, tuple)) and re.search(r"\bet\s+al\b", str(value or ""), re.I):
        cleaned.append("others")
    return " and ".join(cleaned)


def _end_with_period(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text[-1] in ".?!" else text + "."


def _format_structured_reference(reference: dict[str, Any], index: int, style: str) -> str:
    style = str(style or "apa").casefold()
    title = str(reference.get("title") or "").strip()
    authors = _authors_text(reference.get("authors"))
    venue = str(reference.get("venue") or reference.get("journal") or "").strip()
    year = str(reference.get("year") or "").strip()
    volume = str(reference.get("volume") or "").strip()
    issue = str(reference.get("issue") or "").strip()
    pages = str(reference.get("pages") or "").strip()
    doi = _clean_doi(reference.get("doi"))
    url = str(reference.get("url") or "").strip()
    if not title and not authors:
        return ""
    location = f"https://doi.org/{doi}" if doi else url

    if style == "ieee":
        lead = f"[{index}] " + (f'{authors}, ' if authors else "")
        tail_parts = [
            venue,
            f"vol. {volume}" if volume else "",
            f"no. {issue}" if issue else "",
            f"pp. {pages}" if pages else "",
            year,
        ]
        tail = ", ".join(part for part in tail_parts if part)
        if title and tail:
            text = f'{lead}"{title}," {tail}.'
        elif title:
            text = f'{lead}"{_end_with_period(title)}"'
        else:
            text = _end_with_period(f"{lead}{tail}".rstrip(", "))
        return f"{text} {location}".strip()

    if style == "mla":
        pieces = []
        if authors:
            pieces.append(_end_with_period(authors))
        if title:
            pieces.append(f'"{_end_with_period(title)}"' if authors else _end_with_period(title))
        container_parts = [
            venue,
            f"vol. {volume}" if volume else "",
            f"no. {issue}" if issue else "",
            year,
            f"pp. {pages}" if pages else "",
        ]
        container = ", ".join(part for part in container_parts if part)
        if container:
            pieces.append(_end_with_period(container))
        if location:
            pieces.append(location)
        return " ".join(pieces).strip()

    if style == "chicago":
        pieces = []
        if authors:
            pieces.append(_end_with_period(authors))
        if title:
            pieces.append(f'"{_end_with_period(title)}"' if authors else _end_with_period(title))
        source = venue
        if volume:
            source += f" {volume}"
        if issue:
            source += f", no. {issue}"
        if year:
            source += f" ({year})"
        if pages:
            source += f": {pages}"
        source = source.strip(" ,")
        if source:
            pieces.append(_end_with_period(source))
        if location:
            pieces.append(location)
        return " ".join(pieces).strip()

    # APA (default)
    date_part = f"({year or 'n.d.'})."
    pieces = []
    if authors:
        pieces.append(f"{_end_with_period(authors)} {date_part}")
        if title:
            pieces.append(_end_with_period(title))
    else:
        pieces.append(f"{_end_with_period(title)} {date_part}")
    source = venue
    if volume:
        source += f", {volume}" if source else volume
    if issue:
        source += f"({issue})"
    if pages:
        source += f", {pages}" if source else pages
    if source:
        pieces.append(_end_with_period(source))
    if location:
        pieces.append(location)
    return " ".join(pieces).strip()


def format_bibtex(references: list[dict[str, Any]]) -> str:
    entries = []
    for index, reference in enumerate(references, 1):
        fields = []
        used: set[str] = set()
        for key in (
            "title", "authors", "year", "venue", "journal",
            "volume", "issue", "pages", "doi", "url",
        ):
            value = reference.get(key)
            if value in (None, "") or value == []:
                continue
            bib_key = {"authors": "author", "venue": "journal"}.get(key, key)
            if bib_key in used:
                continue
            if key == "authors":
                text = _bibtex_authors(value)
            elif key == "doi":
                text = re.sub(r"[\s{}]", "", _clean_doi(value))
            elif key == "url":
                text = re.sub(r"[\s{}]", "", str(value))
            else:
                text = _escape_bibtex(value)
            if not text:
                continue
            used.add(bib_key)
            fields.append(f"  {bib_key} = {{{text}}}")
        if fields:
            entry_type = "article" if reference.get("venue") or reference.get("journal") else "misc"
            entries.append(f"@{entry_type}{{reference{index},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries)


def format_accepted_references(
    references: list[dict[str, Any]],
    style: str = "apa",
) -> dict[str, Any]:
    """Format user-accepted source metadata without asking a model to write citations."""
    selected_style = style.casefold()
    if selected_style not in CITATION_STYLES:
        raise ValueError(f"Unsupported citation style: {style}")
    allowed = {
        "title", "authors", "year", "venue", "journal", "volume",
        "issue", "pages", "doi", "url",
    }
    cleaned = [
        {
            key: value
            for key, value in reference.items()
            if key in allowed and value not in (None, "") and value != []
        }
        for reference in references
    ]
    formatted = [
        _format_structured_reference(reference, index, selected_style)
        for index, reference in enumerate(cleaned, start=1)
    ]
    return {
        "style": selected_style.upper(),
        "references": cleaned,
        "citations_text": "\n\n".join(item for item in formatted if item),
        "bibtex": format_bibtex(cleaned),
    }


def _reference_result(ref: ReferenceEntry, verification: dict[str, Any], index: int, style: str) -> dict[str, Any]:
    details = {
        "id": f"REF-{index:04d}",
        "ref_number": ref.ref_number,
        "raw": ref.raw,
        **verification,
    }
    if verification["verification_status"] == "verified":
        details["formatted"] = (
            _format_structured_reference(details, index, style)
            or _format_reference_entry(ref, index, style)
        )
    else:
        details["formatted"] = _format_reference_entry(ref, index, style)
    return details


# --------------------------------------------------------------------------- #
# In-text citation parsing and matching
# --------------------------------------------------------------------------- #

def _claim_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _expand_number_list(text: str) -> list[int]:
    numbers: set[int] = set()
    for part in re.split(r"[,;]", text):
        bounds = re.fullmatch(r"\s*(\d+)\s*[\-–—]\s*(\d+)\s*", part)
        if bounds:
            start, end = (int(value) for value in bounds.groups())
            if end >= start:
                numbers.update(range(start, min(end, start + 50) + 1))
            continue
        single = re.fullmatch(r"\s*(\d+)\s*", part)
        if single:
            numbers.add(int(single.group(1)))
    return sorted(numbers)


def _author_year_keys(citation_text: str) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = []
    for part in re.split(r"\s*;\s*", citation_text):
        year_match = re.search(r"\b((?:1[5-9]|20)\d{2})[a-z]?\b", part, re.IGNORECASE)
        if not year_match:
            continue
        surnames = _citation_surnames(part[:year_match.start()])
        # "(in March 2020)" is a date, not a citation.
        if not surnames or all(surname in _MONTH_NAMES for surname in surnames):
            continue
        keys.append({
            "kind": "author_year",
            "year": int(year_match.group(1)),
            "surnames": surnames,
        })
    return keys


def _citation_keys(text: str) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for match in _INLINE_CITATION_RE.finditer(text):
        if match.group("source_id"):
            citations.append({"kind": "source_id", "value": match.group("source_id").upper()})
        elif match.group("numbers"):
            citations.extend(
                {"kind": "number", "value": number}
                for number in _expand_number_list(match.group("numbers"))
            )
        elif match.group("footnote"):
            citations.append({"kind": "number", "value": int(match.group("footnote"))})
        elif match.group("superscript"):
            digits = match.group("superscript").translate(_SUPERSCRIPT_TABLE)
            if digits.isdigit():
                citations.append({"kind": "number", "value": int(digits)})
        elif match.group("author") or match.group("narrative"):
            citations.extend(_author_year_keys(match.group("author") or match.group("narrative")))
        elif match.group("doi"):
            citations.append({"kind": "doi", "value": _clean_doi(match.group("doi")).casefold()})
    return citations


def _match_citation_references(
    citations: list[dict[str, Any]],
    references: list[ReferenceEntry],
) -> list[ReferenceEntry]:
    numbered: dict[int, ReferenceEntry] = {}
    for position, reference in enumerate(references, start=1):
        number = _safe_int(reference.ref_number)
        numbered.setdefault(number if number is not None else position, reference)

    matches: list[ReferenceEntry] = []
    for citation in citations:
        kind = citation["kind"]
        if kind == "number":
            reference = numbered.get(citation["value"])
            if reference is not None:
                matches.append(reference)
        elif kind == "doi":
            for reference in references:
                if reference.doi and _clean_doi(reference.doi).casefold() == citation["value"]:
                    matches.append(reference)
        elif kind == "author_year":
            wanted = set(citation.get("surnames") or [])
            for reference in references:
                raw_text = _normalize_match(reference.raw)
                reference_year = _year_int(reference.year)
                year_ok = (
                    reference_year == citation.get("year")
                    if reference_year is not None
                    else str(citation.get("year")) in raw_text
                )
                if not year_ok:
                    continue
                reference_surnames = set(_author_surnames(reference.authors))
                if reference_surnames:
                    name_ok = bool(reference_surnames & wanted)
                else:
                    name_ok = any(surname in raw_text.split() for surname in wanted)
                if name_ok:
                    matches.append(reference)
    return _deduplicate_references(matches)


def _claim_citation_keys(claim: str, document_body: str) -> list[dict[str, Any]]:
    citations = _citation_keys(claim)
    if citations:
        return citations
    position = document_body.find(claim)
    if position < 0:
        return []
    start = position + len(claim)
    adjacent = _ADJACENT_CITATION.match(document_body[start:start + 100])
    return _citation_keys(adjacent.group(1)) if adjacent else []


def _document_body_and_references(text: str) -> tuple[str, list[ReferenceEntry]]:
    _, reference_text = find_reference_section([text])
    if not reference_text:
        return text, []
    headings = list(_REFERENCE_HEADING.finditer(text))
    if headings:
        body = text[:headings[-1].start()].rstrip()
    else:
        marker = reference_text.strip()[:80]
        position = text.rfind(marker) if marker else -1
        if position <= 0:
            return text, []
        body = text[:position].rstrip()
    bundle = parse_reference_entries(reference_text)
    return body, list(bundle.entries)


def _normalise_quote_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _contains_draft_text(draft: str, candidate: str) -> bool:
    return _normalise_quote_text(candidate) in _normalise_quote_text(draft)


async def _extract_claims(draft_text: str) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sentence in split_sentences(draft_text):
        text = sentence.strip()
        if len(text.split()) < 4 or _AUTHOR_INTRODUCTION.search(text):
            continue
        if not _CITATION_MARKER.search(text) and not _EVIDENCE_CLAIM_CUE.search(text):
            continue
        key = _normalise_quote_text(text)
        if key and key not in seen:
            seen.add(key)
            extracted.append({"text": text})
    return extracted


# --------------------------------------------------------------------------- #
# Candidate sources and claim judging
# --------------------------------------------------------------------------- #

def _candidate_source(hit: dict[str, Any]) -> dict[str, Any]:
    metadata = hit.get("metadata") or {}
    paper_id = str(hit.get("paper_id") or "")
    return {
        "id": paper_id,
        "paper_id": paper_id,
        "title": hit.get("title") or metadata.get("title") or "Untitled source",
        "authors": metadata.get("authors"),
        "year": metadata.get("year"),
        "doi": metadata.get("doi"),
        "venue": metadata.get("journal") or metadata.get("venue"),
        "citation_count": metadata.get("citation_count"),
        "passage": str(hit.get("text") or ""),
        "page": hit.get("page"),
        "source_type": "user_library",
    }


def _external_candidate(source: dict[str, Any]) -> dict[str, Any]:
    doi = str(source.get("doi") or "")
    title = str(source.get("title") or "")
    source_id = "EXT-" + hashlib.sha256((doi or title).encode("utf-8")).hexdigest()[:12].upper()
    return {
        "id": source_id,
        "paper_id": None,
        "title": title,
        "authors": source.get("authors"),
        "year": source.get("year"),
        "doi": doi or None,
        "venue": source.get("venue"),
        "citation_count": source.get("citation_count"),
        "passage": str(source.get("abstract_snippet") or ""),
        "page": None,
        "source_type": source.get("source_api", "crossref"),
        "url": source.get("url"),
        "published_version": source.get("published_version"),
        "published_version_status": source.get("published_version_status", "not_checked"),
        "preprint_version": (
            {
                "title": source.get("title"),
                "year": source.get("year"),
                "url": source.get("url"),
            }
            if source.get("source_api") == "arxiv"
            else None
        ),
    }


async def _verify_candidate_metadata(source: dict[str, Any]) -> dict[str, Any]:
    published = source.get("published_version") or {}
    if source.get("doi"):
        doi = source["doi"]
        title = source.get("title")
    elif published.get("doi"):
        doi = published["doi"]
        title = published.get("title") or source.get("title")
    else:
        return {
            "metadata_verification": "manual_review",
            "retraction_status": "unknown",
        }
    canonical = await _lookup_crossref_doi(str(doi))
    if canonical is None:
        status, _ = await _classify_unresolved_doi(str(doi))
        return {
            "metadata_verification": status,
            "retraction_status": "unknown",
        }
    title_match = _titles_match(title, canonical.get("title"))
    expected_authors = set(_author_surnames(source.get("authors")))
    canonical_authors = set(_author_surnames(canonical.get("authors")))
    authors_match = not expected_authors or not canonical_authors or bool(
        expected_authors & canonical_authors
    )
    if not title_match or not authors_match:
        status = "metadata_mismatch"
    elif canonical["is_retracted"]:
        status = "retracted"
    else:
        status = "verified"
    return {
        "metadata_verification": status,
        "metadata_source": "crossref",
        "canonical_title": canonical.get("title"),
        "canonical_authors": canonical.get("authors"),
        "canonical_year": canonical.get("year"),
        "canonical_doi": canonical.get("doi"),
        "retraction_status": canonical["retraction_status"],
        "citation_count": canonical.get("citation_count"),
    }


def _provider_results(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, Exception) or not isinstance(response, dict):
        return []
    return list(response.get("results") or [])


async def _search_external_candidates(claim: str) -> list[dict[str, Any]]:
    cache_key = "search:" + _normalize_match(claim)
    cached, value = _cache_lookup(cache_key)
    if cached:
        return [dict(candidate) for candidate in value]
    responses = await asyncio.gather(
        search_semantic_scholar(claim, limit=2),
        search_openalex(claim, limit=2),
        search_crossref(claim, limit=2),
        search_arxiv(claim, limit=2),
        return_exceptions=True,
    )
    successful_providers = [
        response for response in responses
        if not isinstance(response, Exception)
        and response.get("status") in {None, "success"}
    ]
    if not successful_providers:
        failures = [
            str(response) if isinstance(response, Exception)
            else str(response.get("status") or "unknown status")
            for response in responses
        ]
        raise RuntimeError(
            "All scholarly search providers failed: " + "; ".join(failures)
        )
    crossref_sources = _provider_results(responses[2])
    arxiv_sources = _provider_results(responses[3])
    for preprint in arxiv_sources:
        preprint_year = _year_int(preprint.get("year"))
        for published in crossref_sources:
            title = str(published.get("title") or "")
            if _title_similarity(preprint.get("title"), title) < 0.94 or not published.get("doi"):
                continue
            published_year = _year_int(published.get("year"))
            if preprint_year and published_year and published_year < preprint_year:
                continue
            preprint["published_version"] = {
                "title": title,
                "doi": published["doi"],
                "year": published.get("year"),
                "url": published.get("url"),
            }
            preprint["published_version_status"] = "possible_published_version_found"
            break
        else:
            preprint["published_version_status"] = "no_crossref_match_found"

    candidates: list[dict[str, Any]] = []
    for response in responses:
        if isinstance(response, Exception):
            logger.info("citation_external_source_search_failed", extra={"error": str(response)})
            continue
        for source in _provider_results(response):
            if not source.get("abstract_snippet"):
                continue
            candidate = _external_candidate(source)
            normalized_title = _normalize_match(candidate.get("title"))
            normalized_doi = _normalize_match(candidate.get("doi"))
            existing = next(
                (
                    item for item in candidates
                    if (
                        normalized_doi
                        and normalized_doi == _normalize_match(item.get("doi"))
                    ) or (
                        normalized_title
                        and _title_similarity(normalized_title, item.get("title")) >= 0.94
                    )
                ),
                None,
            )
            if existing:
                if candidate.get("preprint_version"):
                    existing["preprint_version"] = candidate["preprint_version"]
                    existing["published_version"] = candidate.get("published_version")
                    existing["published_version_status"] = candidate.get("published_version_status")
                continue
            if normalized_title or normalized_doi:
                candidates.append(candidate)
    candidates = candidates[:5]
    _cache_store(cache_key, candidates)
    return [dict(candidate) for candidate in candidates]


def _no_judgment(note: str) -> dict[str, Any]:
    return {
        "verdict": "source_not_found",
        "evidence_excerpt": "",
        "source_id": None,
        "source": None,
        "confidence": 0.0,
        "note": note,
        "citation_context": "unknown",
    }


async def _judge_claim(claim: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return _no_judgment("No candidate source passage was found.")
    usable = [item for item in candidates if item.get("passage")]
    if not usable:
        return _no_judgment(
            "Candidate metadata was found, but no source passage was available to verify support."
        )
    # Local labels keep several passages from the same paper distinguishable.
    excerpts = [
        {
            "source_id": f"S{position}",
            "title": item.get("title"),
            "passage": item["passage"][:1500],
        }
        for position, item in enumerate(usable, start=1)
    ]
    by_label = {excerpt["source_id"]: item for excerpt, item in zip(excerpts, usable)}
    try:
        response = await call_llm(
            prompt=(
                "Assess whether one of these source passages supports or contradicts the exact claim. "
                "Return JSON only with keys verdict (supported|partially_supported|unsupported|contradicted), "
                "source_id, evidence_quote (verbatim exact quote from that passage), confidence (0 to 1), "
                "citation_context (background|method|contrast|other), and note (one concise sentence). "
                "Never infer facts not stated in the passage.\n"
                f"CLAIM:\n{claim}\nPASSAGES:\n{json.dumps(excerpts, ensure_ascii=False)}"
            ),
            agent_role="verification",
            system="You conservatively verify academic claims against supplied source text. Output no reasoning.",
            temperature=0.0,
            max_tokens=500,
        )
    except Exception:
        logger.warning("citation_claim_judgment_failed", exc_info=True)
        return _no_judgment("The verification model was unavailable; manual review is required.")
    parsed = parse_llm_json(response)
    if not isinstance(parsed, dict):
        parsed = {}
    chosen = by_label.get(str(parsed.get("source_id") or "").strip())
    quote = str(parsed.get("evidence_quote") or "").strip()
    verdict = parsed.get("verdict")
    quote_is_verbatim = bool(
        chosen
        and len(quote.split()) >= 4
        and _normalise_quote_text(quote) in _normalise_quote_text(chosen.get("passage", ""))
    )
    if verdict not in _CLAIM_VERDICTS - {"source_not_found"} or not quote_is_verbatim:
        return _no_judgment("No quote-verified support was returned; manual review is required.")
    try:
        confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "verdict": verdict,
        "evidence_excerpt": quote,
        "source_id": chosen["id"],
        "source": chosen,
        "confidence": confidence,
        "note": str(parsed.get("note") or "")[:300],
        "citation_context": (
            parsed.get("citation_context")
            if parsed.get("citation_context") in {"background", "method", "contrast", "other"}
            else "unknown"
        ),
    }


# --------------------------------------------------------------------------- #
# Draft audit
# --------------------------------------------------------------------------- #

def _audit_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = (
        "verified",
        "evidence_required",
        "incorrect_reference",
        "source_mismatch",
        "needs_review",
    )
    status_counts = {
        status: sum(result.get("status") == status for result in results)
        for status in statuses
    }
    partially_supported = sum(
        result.get("verdict") == "partially_supported" for result in results
    )
    total_claims = len(results)

    if not total_claims:
        assessment = (
            "No citation-sensitive claims were detected by the local rules. "
            "This does not establish that the document contains no claims needing evidence."
        )
    elif status_counts["verified"] == total_claims:
        assessment = (
            "All detected claims were supported by quote-checked full-text passages. "
            "Review the source context before relying on these findings."
        )
    else:
        assessment = (
            "Manual review is required for one or more detected claims. "
            "See the claim-level findings and evidence notes below."
        )

    return {
        "total_claims": total_claims,
        "verified_claims": status_counts["verified"],
        "partially_supported_claims": partially_supported,
        "uncited_claims": sum(bool(result.get("uncited")) for result in results),
        "incorrect_references": status_counts["incorrect_reference"],
        "source_mismatches": status_counts["source_mismatch"],
        "needs_review": status_counts["needs_review"],
        "status_counts": status_counts,
        "assessment": assessment,
    }


def _empty_audit_result() -> dict[str, Any]:
    return {
        "claims": [],
        "uncited_claim_count": 0,
        "incorrect_reference_count": 0,
        "source_mismatch_count": 0,
        "needs_review_count": 0,
        "source_quality": [],
        "quality_hints": [],
        "reading_list": [],
        "references": [],
        "summary": _audit_summary([]),
    }


def _new_claim_result(claim: str, citation_keys: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "claim_id": _claim_id(claim),
        "claim": claim,
        "status": "needs_review",
        "verdict": "needs_review",
        "evidence_excerpt": "",
        "source": None,
        "confidence": 0.0,
        "action": "manual_review",
        "citation_context": "unknown",
        "note": "",
        "uncited": not citation_keys,
        "citation_keys": citation_keys,
        "references": [],
        "candidate_sources": [],
    }


def _aggregate_supported_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.get("status") != "verified":
            continue
        source = result.get("source") or {}
        key = str(source.get("paper_id") or source.get("id") or "")
        if not key:
            continue
        entry = aggregated.get(key)
        if entry is None:
            year = _year_int(source.get("year"))
            entry = {"source": source, "claim_count": 0, "years": [year] if year else []}
            aggregated[key] = entry
        entry["claim_count"] += 1
    return list(aggregated.values())


def _source_quality_hints(sources: list[dict[str, Any]]) -> list[str]:
    if not sources:
        return []
    hints = []
    total_claims = sum(source["claim_count"] for source in sources)
    author_usage: dict[str, int] = {}
    for source in sources:
        authors = _author_surnames((source.get("source") or {}).get("authors"))
        if authors:
            author_usage[authors[0]] = (
                author_usage.get(authors[0], 0) + source["claim_count"]
            )
    if total_claims and author_usage and max(author_usage.values()) / total_claims > 0.5:
        hints.append("More than half of supported claims rely on one author's work; diversify the evidence.")
    current_year = datetime.now().year
    old_sources = [
        year for source in sources for year in source.get("years", [])
        if isinstance(year, int) and current_year - year >= 15
    ]
    if old_sources:
        hints.append(f"{len(old_sources)} cited source(s) are at least 15 years old; check whether newer evidence is available.")
    return hints


async def audit_draft_claims(
    user_id: str,
    paper_ids: list[str],
    draft_text: str,
) -> dict[str, Any]:
    body, references = _document_body_and_references(draft_text)
    extraction_started = time.perf_counter()
    claims = await _extract_claims(body)
    tracking.log_audit_event(
        "citation.claim_extraction",
        model_used="local_heuristic",
        latency_ms=int((time.perf_counter() - extraction_started) * 1000),
    )
    if not claims:
        return _empty_audit_result()

    citations_by_claim = {
        claim["text"]: _claim_citation_keys(claim["text"], body) for claim in claims
    }
    cited_by_claim = {
        text: _match_citation_references(keys, references)
        for text, keys in citations_by_claim.items()
    }
    used_references = _deduplicate_references([
        reference
        for matched in cited_by_claim.values()
        for reference in matched
    ])
    resolution_started = time.perf_counter()
    verification_semaphore = asyncio.Semaphore(4)

    async def verify_bounded(reference: ReferenceEntry) -> dict[str, Any]:
        async with verification_semaphore:
            return await _verify_reference(reference)

    verification_results = await asyncio.gather(
        *(verify_bounded(reference) for reference in used_references)
    )
    verification_by_key = {
        _reference_key(reference): verification
        for reference, verification in zip(used_references, verification_results)
    }
    tracking.log_audit_event(
        "citation.reference_resolution",
        latency_ms=int((time.perf_counter() - resolution_started) * 1000),
    )
    papers = await asyncio.to_thread(_load_papers_sync, user_id, paper_ids)
    support_semaphore = asyncio.Semaphore(4)

    async def audit_claim(claim_data: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        claim = claim_data["text"]
        citation_keys = citations_by_claim[claim]
        matched_references = cited_by_claim[claim]

        if not citation_keys:
            result.update({
                "status": "evidence_required",
                "verdict": "no_citation",
                "action": "find_evidence",
                "note": "This claim has no detectable inline citation. Evidence search was not run.",
            })
            return result
        if not references:
            result["note"] = (
                "No bibliography section was detected, so the citation could not be matched "
                "to a source."
            )
            return result
        if not matched_references:
            result.update({
                "status": "incorrect_reference",
                "verdict": "reference_not_in_bibliography",
                "action": "review_reference",
                "note": "The inline citation could not be matched to a bibliography entry.",
            })
            return result

        reference_details = []
        statuses: list[str] = []
        for reference in matched_references:
            verification = verification_by_key[_reference_key(reference)]
            statuses.append(verification["verification_status"])
            reference_details.append({
                "ref_number": reference.ref_number,
                "raw": reference.raw,
                **verification,
            })
        result["references"] = reference_details

        if any(status in {"doi_not_found", "arxiv_not_found", "metadata_mismatch"} for status in statuses):
            result.update({
                "status": "incorrect_reference",
                "verdict": "metadata_mismatch",
                "action": "correct_reference",
                "note": "The DOI or bibliographic metadata does not match the resolved reference.",
            })
            return result
        if "retracted" in statuses:
            result.update({
                "status": "incorrect_reference",
                "verdict": "retracted_source",
                "action": "replace_source",
                "note": "A cited source is flagged as retracted or carries a retraction notice.",
            })
            return result
        if "verified" not in statuses:
            result["note"] = (
                "Reference metadata could not be resolved. Support checking was skipped."
            )
            return result

        matching_paper: dict[str, Any] | None = None
        for reference in matched_references:
            verification = verification_by_key[_reference_key(reference)]
            for paper in papers.values():
                similarity = max(
                    _title_similarity(reference.title, paper.get("title")),
                    _title_similarity(verification.get("title"), paper.get("title")),
                )
                if similarity >= 0.92:
                    matching_paper = paper
                    break
            if matching_paper:
                break
        if matching_paper:
            try:
                hits = await semantic_search(
                    user_id=user_id,
                    query=claim,
                    top_k=5,
                    paper_id=str(matching_paper["id"]),
                    exclude_references=True,
                )
            except Exception as error:
                logger.warning("citation_source_passage_lookup_failed", exc_info=True)
                result["note"] = f"Could not read the selected source text: {error}"
                return result
            candidates = [
                _candidate_source(hit)
                for hit in hits
                if str(hit.get("paper_id")) == str(matching_paper["id"]) and hit.get("text")
            ]
            if candidates:
                judgment = await _judge_claim(claim, candidates)
                result.update({
                    "verdict": judgment["verdict"],
                    "evidence_excerpt": judgment["evidence_excerpt"],
                    "source": judgment.get("source"),
                    "confidence": judgment["confidence"],
                    "citation_context": judgment["citation_context"],
                    "note": judgment["note"],
                })
                if judgment["verdict"] == "supported":
                    result.update({"status": "verified", "action": "keep"})
                elif judgment["verdict"] in {"unsupported", "contradicted"}:
                    result.update({
                        "status": "source_mismatch",
                        "action": "find_evidence",
                    })
                return result

        abstract_candidates = [
            {
                "id": "REF-" + hashlib.sha256(
                    _reference_key(reference).encode("utf-8")
                ).hexdigest()[:12].upper(),
                "title": reference.title or "Referenced paper",
                "passage": verification_by_key[_reference_key(reference)].get(
                    "abstract_excerpt", ""
                ),
                "source_type": "abstract",
            }
            for reference in matched_references
            if verification_by_key[_reference_key(reference)].get("abstract_excerpt")
        ]
        if abstract_candidates:
            judgment = await _judge_claim(claim, abstract_candidates)
            if judgment["verdict"] in {"supported", "partially_supported"}:
                result["verdict"] = "partially_supported"
                result["evidence_excerpt"] = judgment["evidence_excerpt"]
                result["confidence"] = min(judgment["confidence"], 0.6)
                result["note"] = (
                    "Only an abstract was available; full-text support cannot be established."
                )
            elif judgment["verdict"] == "contradicted":
                result["verdict"] = "needs_review"
                result["evidence_excerpt"] = judgment["evidence_excerpt"]
                result["note"] = (
                    "Only an abstract was available, and it appears to contradict the claim; "
                    "check the full text."
                )
            else:
                result["verdict"] = "needs_review"
                result["note"] = (
                    "Only an abstract was available; full-text support cannot be established."
                )
            result["evidence_level"] = "abstract"
            return result

        result["note"] = (
            "Reference metadata matches, but the source full text is unavailable. "
            "Support was not inferred from metadata."
        )
        result["verdict"] = "needs_review"
        return result

    async def audit_one(claim_data: dict[str, Any]) -> dict[str, Any]:
        result = _new_claim_result(claim_data["text"], citations_by_claim[claim_data["text"]])
        async with support_semaphore:
            try:
                return await audit_claim(claim_data, result)
            except Exception:
                logger.warning("citation_claim_audit_failed", exc_info=True)
                result.update({
                    "status": "needs_review",
                    "verdict": "needs_review",
                    "action": "manual_review",
                    "note": "This claim could not be audited automatically; review it manually.",
                })
                return result

    support_started = time.perf_counter()
    results = await asyncio.gather(*(audit_one(claim) for claim in claims))
    tracking.log_audit_event(
        "citation.support_check",
        model_used="configured_llm",
        latency_ms=int((time.perf_counter() - support_started) * 1000),
    )
    aggregated_sources = _aggregate_supported_sources(results)
    return {
        "claims": results,
        "uncited_claim_count": sum(1 for result in results if result["uncited"]),
        "incorrect_reference_count": sum(
            result["status"] == "incorrect_reference" for result in results
        ),
        "source_mismatch_count": sum(
            result["status"] == "source_mismatch" for result in results
        ),
        "needs_review_count": sum(
            result["status"] == "needs_review" for result in results
        ),
        "source_quality": [
            {
                "paper_id": entry["source"].get("paper_id"),
                "title": entry["source"].get("title"),
                "claim_count": entry["claim_count"],
                "years": entry["years"],
            }
            for entry in aggregated_sources
        ],
        "quality_hints": _source_quality_hints(aggregated_sources),
        "reading_list": [],
        "references": [
            {
                "ref_number": reference.ref_number,
                "raw": reference.raw,
                **verification_by_key[_reference_key(reference)],
            }
            for reference in used_references
        ],
        "summary": _audit_summary(results),
    }


# --------------------------------------------------------------------------- #
# Evidence search and single-claim checks
# --------------------------------------------------------------------------- #

def _support_score(verdict: str, confidence: float) -> int:
    confidence = min(1.0, max(0.0, confidence))
    if verdict == "supported":
        return round(80 + 20 * confidence)
    if verdict == "partially_supported":
        return round(50 + 30 * confidence)
    if verdict == "unsupported":
        return round(10 + 30 * confidence)
    if verdict == "contradicted":
        return round(10 * confidence)
    return 0


async def find_evidence_for_claim(
    user_id: str,
    paper_ids: list[str],
    claim: str,
) -> dict[str, Any]:
    """Search only after an explicit user request and return ranked, reviewable candidates."""
    search_started = time.perf_counter()
    allowed_papers = {str(paper_id) for paper_id in paper_ids}
    local_candidates: list[dict[str, Any]] = []
    if paper_ids:
        try:
            hits = await semantic_search(
                user_id=user_id,
                query=claim,
                top_k=40,
                exclude_references=True,
            )
        except Exception:
            logger.warning("citation_local_evidence_search_failed", exc_info=True)
            hits = []
        local_candidates = [
            _candidate_source(hit)
            for hit in hits
            if str(hit.get("paper_id") or "") in allowed_papers and hit.get("text")
        ][:3]
    search_warning = None
    try:
        external_candidates = await _search_external_candidates(claim)
    except Exception as error:
        logger.warning("citation_external_evidence_search_failed", exc_info=True)
        external_candidates = []
        search_warning = str(error)
    seen_ids = {candidate["id"] for candidate in local_candidates}
    candidates = (
        local_candidates
        + [candidate for candidate in external_candidates if candidate["id"] not in seen_ids]
    )[:5]

    semaphore = asyncio.Semaphore(3)

    async def rank_one(candidate: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            judgment = await _judge_claim(claim, [candidate])
            is_abstract = candidate.get("source_type") != "user_library"
            verdict = judgment["verdict"]
            if is_abstract and verdict == "supported":
                verdict = "partially_supported"
            if is_abstract:
                try:
                    metadata = await _verify_candidate_metadata(candidate)
                except Exception:
                    logger.info("citation_candidate_metadata_unavailable", exc_info=True)
                    metadata = {
                        "metadata_verification": "manual_review",
                        "retraction_status": "unknown",
                    }
            else:
                metadata = {
                    "metadata_verification": "from_user_library",
                    "metadata_source": "uploaded_library",
                    "retraction_status": "unknown",
                }
            note = (
                "Abstract-only evidence cannot establish full support."
                if is_abstract and verdict == "partially_supported"
                else judgment["note"]
            )
            score = _support_score(verdict, judgment["confidence"])
            if metadata.get("metadata_verification") == "retracted":
                score = 0
                note = (note + " This source is flagged as retracted.").strip()
            return {
                **candidate,
                **metadata,
                "verdict": verdict,
                "support_level": (
                    "supports" if verdict == "supported"
                    else "partly_supports" if verdict == "partially_supported"
                    else "contradicts" if verdict == "contradicted"
                    else "only_related"
                ),
                "support_score": score,
                "confidence": judgment["confidence"],
                "evidence_excerpt": judgment["evidence_excerpt"],
                "evidence_level": "full_text" if not is_abstract else "abstract",
                "note": note,
            }

    ranked = list(await asyncio.gather(*(rank_one(candidate) for candidate in candidates)))
    ranked.sort(
        key=lambda item: (
            item["support_score"],
            _safe_int(item.get("citation_count")) or 0,
            _year_int(item.get("year")) or 0,
        ),
        reverse=True,
    )
    tracking.log_audit_event(
        "citation.source_search",
        model_used="configured_llm",
        latency_ms=int((time.perf_counter() - search_started) * 1000),
    )
    return {"claim": claim, "candidates": ranked, "search_warning": search_warning}


async def check_text(
    user_id: str,
    text: str,
    citation: str | None = None,
    evidence_text: str | None = None,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Check user text against a supplied citation, or search when none is supplied."""
    selected_papers = paper_ids or []
    if not citation or not citation.strip():
        return {
            "mode": "search",
            **(await find_evidence_for_claim(user_id, selected_papers, text)),
        }

    parsed = parse_reference_entries(f"References\n{citation.strip()}")
    reference = parsed.entries[0] if parsed.entries else ReferenceEntry(raw=citation.strip())
    doi_match = re.search(_DOI, citation, re.IGNORECASE)
    if doi_match:
        reference.doi = _clean_doi(doi_match.group(0))
    arxiv_match = re.search(
        r"arxiv(?:\.org/(?:abs|pdf)/|\s*:\s*)"
        r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[a-z]{2})?/\d{7}(?:v\d+)?)",
        citation,
        re.IGNORECASE,
    )
    if arxiv_match:
        reference.arxiv_id = f"arXiv:{arxiv_match.group(1)}"
    resolution_started = time.perf_counter()
    metadata = await _verify_reference(reference)
    tracking.log_audit_event(
        "citation.reference_resolution",
        latency_ms=int((time.perf_counter() - resolution_started) * 1000),
    )

    status = metadata["verification_status"]
    if status in {"doi_not_found", "arxiv_not_found", "metadata_mismatch"}:
        return {
            "mode": "citation_check",
            "status": "incorrect_reference",
            "verdict": status,
            "metadata": metadata,
            "evidence_excerpt": "",
            "note": "The citation could not be verified against its DOI/arXiv/title metadata.",
        }
    if status == "retracted":
        return {
            "mode": "citation_check",
            "status": "incorrect_reference",
            "verdict": "retracted_source",
            "metadata": metadata,
            "evidence_excerpt": "",
            "note": "The cited source is flagged as retracted or carries a retraction notice.",
        }

    query = reference.title or reference.doi or reference.arxiv_id or citation
    try:
        sources = await _search_external_candidates(query)
    except Exception as error:
        logger.warning("citation_check_source_search_failed", exc_info=True)
        sources = []
        metadata.setdefault("resolution_error", str(error))
    canonical_doi = _clean_doi(metadata.get("doi") or reference.doi).casefold()
    canonical_title = metadata.get("title") or reference.title
    if canonical_doi:
        sources = [
            source for source in sources
            if _clean_doi(source.get("doi")).casefold() == canonical_doi
            or _title_similarity(source.get("title"), canonical_title) >= 0.92
        ]
    if canonical_title:
        sources = [
            source for source in sources
            if _titles_match(source.get("title"), canonical_title)
        ]
    if not sources and metadata.get("abstract_excerpt"):
        sources = [{
            "id": "REF-" + hashlib.sha256(
                (canonical_doi or _normalize_match(canonical_title) or citation).encode("utf-8")
            ).hexdigest()[:12].upper(),
            "paper_id": None,
            "title": canonical_title or "",
            "authors": metadata.get("authors"),
            "year": metadata.get("year"),
            "doi": metadata.get("doi"),
            "venue": metadata.get("venue"),
            "citation_count": metadata.get("citation_count"),
            "passage": metadata["abstract_excerpt"],
            "page": None,
            "source_type": metadata.get("metadata_source") or "crossref",
            "url": None,
        }]
    if not sources:
        return {
            "mode": "citation_check",
            "status": "needs_review",
            "verdict": "needs_review",
            "metadata": metadata,
            "evidence_excerpt": "",
            "note": "Citation metadata resolved, but no readable source passage was available.",
        }

    matched = sources[0]
    if evidence_text and not _contains_draft_text(matched.get("passage", ""), evidence_text):
        alternatives = await find_evidence_for_claim(user_id, selected_papers, text)
        return {
            "mode": "citation_check",
            "status": "needs_review",
            "verdict": "evidence_not_found_in_abstract",
            "metadata": metadata,
            "source": matched,
            "evidence_excerpt": "",
            "note": (
                "The pasted text is not in the retrieved abstract. Full text was unavailable, "
                "so the quote cannot be confirmed or ruled out."
            ),
            "evidence_level": "abstract",
            "suggested_sources": alternatives["candidates"],
            "search_warning": alternatives.get("search_warning"),
        }
    judgment = await _judge_claim(text, [matched])
    supported_abstract = judgment["verdict"] in {"supported", "partially_supported"}
    verdict = "partially_supported" if supported_abstract else "needs_review"
    alternatives = (
        await find_evidence_for_claim(user_id, selected_papers, text)
        if not supported_abstract
        else {"candidates": [], "search_warning": None}
    )
    if supported_abstract:
        detail = judgment["note"]
    elif judgment["verdict"] == "contradicted":
        detail = "The abstract appears to contradict the claim; check the full text."
    else:
        detail = "The abstract does not establish support."
    return {
        "mode": "citation_check",
        "status": "needs_review",
        "verdict": verdict,
        "metadata": metadata,
        "source": matched,
        "evidence_excerpt": judgment["evidence_excerpt"],
        "note": (
            "Only the abstract was available; full-text support cannot be confirmed. " + detail
        ).strip(),
        "evidence_level": "abstract",
        "suggested_sources": alternatives["candidates"],
        "search_warning": alternatives.get("search_warning"),
    }


async def run_citation_workflow(
    mode: Literal["audit", "check"],
    user_id: str,
    paper_ids: list[str],
    text: str,
    citation: str | None = None,
    evidence_text: str | None = None,
) -> dict[str, Any]:
    """Route audit and write-check entry points through the shared citation services."""
    if mode == "audit":
        return await audit_draft_claims(user_id, paper_ids, text)
    if mode == "check":
        return await check_text(
            user_id,
            text,
            citation=citation,
            evidence_text=evidence_text,
            paper_ids=paper_ids,
        )
    raise ValueError(f"Unsupported citation workflow mode: {mode}")


# --------------------------------------------------------------------------- #
# Storage and bibliography extraction
# --------------------------------------------------------------------------- #

def _load_papers_sync(user_id: str, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not paper_ids:
        return {}
    response = (
        get_supabase()
        .table("papers")
        .select("id, title, storage_path")
        .eq("user_id", user_id)
        .in_("id", paper_ids)
        .execute()
    )
    return {str(row["id"]): row for row in (response.data or [])}


def _load_stored_reference_entries_sync(
    user_id: str,
    paper_id: str,
) -> list[ReferenceEntry] | None:
    """Return references from current document structure, or None when stale/missing."""
    sb = get_supabase()
    structures = (
        sb.table("paper_structure")
        .select("status, extractor_version")
        .eq("paper_id", paper_id)
        .eq("user_id", user_id)
        .execute()
    ).data or []
    if not structures or str(structures[0].get("extractor_version")) != "2":
        return None
    if structures[0].get("status") not in {"ready", "partial"}:
        return None

    rows = (
        sb.table("paper_references")
        .select(
            "ref_number, raw, authors, title, year, venue, doi, arxiv_id, parse_confidence"
        )
        .eq("paper_id", paper_id)
        .eq("user_id", user_id)
        .order("ref_number", desc=False)
        .execute()
    ).data or []
    return [
        ReferenceEntry(
            ref_number=row.get("ref_number"),
            raw=str(row.get("raw") or ""),
            authors=row.get("authors"),
            title=row.get("title"),
            year=row.get("year"),
            venue=row.get("venue"),
            doi=row.get("doi"),
            arxiv_id=row.get("arxiv_id"),
            parse_confidence=float(row.get("parse_confidence") or 0),
        )
        for row in rows
        if row.get("raw")
    ]


async def run_citation_extraction(
    user_id: str,
    paper_ids: list[str],
    style: str = "apa",
    draft_text: str | None = None,
) -> dict[str, Any]:
    """Extract bibliography text deterministically and optionally audit a draft."""
    selected_style = style.lower()
    if selected_style not in CITATION_STYLES:
        raise ValueError(f"Unsupported citation style: {style}")

    paper_records = await asyncio.to_thread(_load_papers_sync, user_id, paper_ids)

    async def extract_paper(paper_id: str) -> dict[str, Any]:
        paper = paper_records.get(paper_id)
        if paper is None:
            return {
                "paper_id": paper_id,
                "title": "Unavailable source",
                "citations_text": "Paper not found for this user.",
                "references": [],
                "count": 0,
                "chunks_used": 0,
                "status": "not_found",
            }
        title = paper.get("title") or "Untitled Paper"
        storage_path = paper.get("storage_path")
        references: list[ReferenceEntry] = []
        extraction_warning = ""
        extraction_method = "stored_document_structure"
        try:
            stored_references = await asyncio.to_thread(
                _load_stored_reference_entries_sync,
                user_id,
                paper_id,
            )
        except Exception:
            logger.warning(
                "stored_citation_references_unavailable",
                extra={"paper_id": paper_id},
                exc_info=True,
            )
            stored_references = None

        if stored_references:
            references = stored_references
        elif storage_path and os.path.exists(storage_path):
            extraction_method = "local_pdf_text"
            structure = None
            try:
                def read_and_extract(path: str = storage_path):
                    with open(path, "rb") as file:
                        return extract_structure(file.read())

                structure = await asyncio.to_thread(read_and_extract)
                references = list(structure.references or [])
            except Exception:
                logger.warning(
                    "pdf_reference_extraction_failed",
                    extra={"paper_id": paper_id},
                    exc_info=True,
                )
                extraction_warning = (
                    "The document could not be parsed. Re-upload a text-readable PDF "
                    "or run OCR on a scanned PDF."
                )
            if references and structure is not None:
                # A storage failure must not discard references that were extracted.
                try:
                    await persist_structure(user_id, paper_id, structure)
                except Exception:
                    logger.warning(
                        "pdf_reference_persist_failed",
                        extra={"paper_id": paper_id},
                        exc_info=True,
                    )
            elif not references and not extraction_warning:
                if structure is not None and structure.references_start_page is not None:
                    extraction_warning = (
                        "A reference section was found, but its entries could not be "
                        "parsed. The bibliography may use an unsupported layout or format."
                    )
                else:
                    extraction_warning = (
                        "No reference section was detected in the document text. "
                        "Scanned PDFs may need OCR."
                    )
        elif stored_references is not None:
            extraction_warning = (
                "No references were found in the stored document structure, and the "
                "original file is unavailable for re-extraction."
            )
        else:
            extraction_method = "unavailable"
            extraction_warning = (
                "The original document file is unavailable, so local reference "
                "extraction could not run."
            )
        references = _deduplicate_references(references)
        structured = [
            _reference_result(
                reference,
                _reference_metadata_without_lookup(reference),
                index,
                selected_style,
            )
            for index, reference in enumerate(references, start=1)
        ]
        formatted = "\n\n".join(item["formatted"] for item in structured)
        status = (
            "completed" if references
            else "source_unavailable" if extraction_method == "unavailable"
            else "empty"
        )
        return {
            "paper_id": paper_id,
            "title": title,
            "citations_text": (
                f"Extracted {len(references)} references from {title} "
                f"({selected_style.upper()}):\n\n{formatted}"
                if formatted else "No references could be extracted from the uploaded source."
            ),
            "references": structured,
            "bibtex": format_bibtex(structured),
            "style": selected_style.upper(),
            "count": len(structured),
            "chunks_used": 0,
            "extraction_method": extraction_method,
            "metadata_verification": "not_performed",
            "warning": extraction_warning,
            "status": status,
        }

    all_citations = await asyncio.gather(*(extract_paper(paper_id) for paper_id in paper_ids))
    result: dict[str, Any] = {
        "agent": "citation",
        "results": all_citations,
        "style": selected_style.upper(),
        "total_papers": len(paper_ids),
    }
    if draft_text and draft_text.strip():
        result["draft_audit"] = await audit_draft_claims(user_id, paper_ids, draft_text)
    return result