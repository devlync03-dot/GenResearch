# ============================================================
# Service: Source Gathering
# Gap-driven academic source retrieval via REST APIs only.
#
# Priority order:
#   1. Semantic Scholar API  (structured, clean metadata)
#   2. CrossRef API          (DOI resolution, reliable)
#   3. OpenAlex API          (no key needed, comprehensive)
#   4. arXiv API             (preprints, CS-heavy)
#
# All 4 APIs run CONCURRENTLY per gap.
# All gaps run CONCURRENTLY (bounded by _GAP_CONCURRENCY).
#
# No Playwright / browser scraping — replaced by direct REST calls.
# ============================================================
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from html.parser import HTMLParser
from typing import Optional

import aiohttp

from config import settings

logger = logging.getLogger(__name__)

# Timeout for external API calls
API_TIMEOUT = aiohttp.ClientTimeout(total=20)
ABSTRACT_SNIPPET_LENGTH = 1500

# Max simultaneous gap queries to avoid hammering APIs
_GAP_CONCURRENCY = 4


class _AbstractTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _clean_abstract(abstract: str) -> str:
    parser = _AbstractTextParser()
    parser.feed(abstract or "")
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def _crossref_year(item: dict) -> int | None:
    for field in ("published-print", "issued", "published-online"):
        date_parts = (item.get(field) or {}).get("date-parts") or []
        if date_parts and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def _title_key(title: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title or "").casefold()).strip()


def _doi_key(doi: object) -> str:
    value = str(doi or "").strip().casefold()
    value = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", value)
    return value.rstrip(" .;,)")


def _source_keys(source: dict) -> set[str]:
    keys = set()
    title = _title_key(source.get("title"))
    doi = _doi_key(source.get("doi"))
    if title:
        keys.add(f"title:{title}")
    if doi:
        keys.add(f"doi:{doi}")
    return keys


def _source_metadata_score(source: dict) -> int:
    score = 0
    title = _title_key(source.get("title"))
    if title and title != "unknown":
        score += 1
    if source.get("authors") and str(source["authors"]).casefold() != "unknown":
        score += 1
    if source.get("year"):
        score += 1
    if _doi_key(source.get("doi")):
        score += 2
    if len(str(source.get("abstract_snippet") or "")) >= 200:
        score += 2
    if source.get("url"):
        score += 1
    if source.get("journal"):
        score += 2
    if source.get("volume") or source.get("issue") or source.get("pages"):
        score += 1
    return score


# ──────────────────────────────────────────────────────────
# Individual API search functions
# ──────────────────────────────────────────────────────────

async def search_semantic_scholar(
    query: str,
    limit: int = 5,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Search Semantic Scholar REST API. Returns structured source metadata."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,url,abstract,citationCount,externalIds,publicationVenue",
    }
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=API_TIMEOUT)

    try:
        async with session.get(url, params=params) as resp:
            if resp.status == 429:
                logger.warning("semantic_scholar_rate_limited")
                return {"status": "rate_limited", "results": []}
            if resp.status != 200:
                logger.warning("semantic_scholar_error", extra={"status": resp.status})
                return {"status": "error", "error": f"HTTP {resp.status}", "results": []}
            data = await resp.json()

        papers = data.get("data", [])
        results = []
        for p in papers:
            authors = ", ".join(a.get("name", "") for a in (p.get("authors") or []))
            doi = (p.get("externalIds") or {}).get("DOI", "")
            results.append({
                "title": p.get("title", "Unknown"),
                "authors": authors,
                "year": p.get("year"),
                "url": p.get("url", ""),
                "doi": doi,
                "abstract_snippet": (p.get("abstract") or "")[:ABSTRACT_SNIPPET_LENGTH],
                "citation_count": p.get("citationCount", 0),
                "journal": (p.get("publicationVenue") or {}).get("name", ""),
                "source_api": "semantic_scholar",
                "accessed_date": datetime.utcnow().isoformat(),
            })
        return {"status": "success", "results": results}

    except asyncio.TimeoutError:
        logger.warning("semantic_scholar_timeout")
        return {"status": "timeout", "results": []}
    except Exception as e:
        logger.warning("semantic_scholar_exception", extra={"error": str(e)})
        return {"status": "error", "error": str(e), "results": []}
    finally:
        if own_session:
            await session.close()


async def search_crossref(
    query: str,
    limit: int = 5,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Search CrossRef REST API for DOIs and scholarly works."""
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": limit,
        "select": "title,author,published-print,published-online,issued,DOI,URL,abstract,is-referenced-by-count,container-title,volume,issue,page,publisher,type",
    }
    headers = {"User-Agent": "GenResearch/0.2.0 (mailto:research@genresearch.dev)"}

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=API_TIMEOUT)

    try:
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                logger.warning("crossref_error", extra={"status": resp.status})
                return {"status": "error", "error": f"HTTP {resp.status}", "results": []}
            data = await resp.json()

        results = []
        for item in data.get("message", {}).get("items", []):
            titles = item.get("title") or []
            title = (titles[0] if titles else None) or "Unknown"
            author_names = []
            for a in item.get("author", []):
                if "given" in a and "family" in a:
                    author_names.append(f"{a['given']} {a['family']}")
                elif "family" in a:
                    author_names.append(a["family"])
                elif "name" in a:
                    author_names.append(a["name"])
            authors = ", ".join(author_names) if author_names else "Unknown"
            abstract = _clean_abstract(item.get("abstract") or "")
            results.append({
                "title": title,
                "authors": authors,
                "year": _crossref_year(item),
                "url": item.get("URL", ""),
                "doi": item.get("DOI", ""),
                "abstract_snippet": abstract[:ABSTRACT_SNIPPET_LENGTH],
                "citation_count": item.get("is-referenced-by-count", 0),
                "journal": (item.get("container-title") or [""])[0],
                "volume": item.get("volume", ""),
                "issue": item.get("issue", ""),
                "pages": item.get("page", ""),
                "publisher": item.get("publisher", ""),
                "publication_type": item.get("type", ""),
                "source_api": "crossref",
                "accessed_date": datetime.utcnow().isoformat(),
            })
        return {"status": "success", "results": results}

    except asyncio.TimeoutError:
        logger.warning("crossref_timeout")
        return {"status": "timeout", "results": []}
    except Exception as e:
        logger.warning("crossref_exception", extra={"error": str(e)})
        return {"status": "error", "error": str(e), "results": []}
    finally:
        if own_session:
            await session.close()


async def search_openalex(
    query: str,
    limit: int = 5,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Search OpenAlex REST API. Free, no API key required."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": limit,
        "select": "title,authorships,publication_year,doi,id,cited_by_count,abstract_inverted_index,primary_location,biblio,type",
    }
    headers = {"User-Agent": "GenResearch/0.2.0 (mailto:research@genresearch.dev)"}

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=API_TIMEOUT)

    try:
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                logger.warning("openalex_error", extra={"status": resp.status})
                return {"status": "error", "error": f"HTTP {resp.status}", "results": []}
            data = await resp.json()

        results = []
        for work in data.get("results", []):
            authors = ", ".join(
                (a.get("author") or {}).get("display_name", "")
                for a in (work.get("authorships") or [])[:5]
            )
            doi = work.get("doi", "") or ""
            if doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]

            # Reconstruct abstract from inverted index when available
            abstract = ""
            inv_index = work.get("abstract_inverted_index")
            if inv_index:
                try:
                    max_pos = max(pos for positions in inv_index.values() for pos in positions)
                    words = [""] * (max_pos + 1)
                    for word, positions in inv_index.items():
                        for pos in positions:
                            words[pos] = word
                    abstract = " ".join(w for w in words if w)
                except Exception:
                    abstract = ""

            results.append({
                "title": work.get("title", "Unknown") or "Unknown",
                "authors": authors,
                "year": work.get("publication_year"),
                "url": work.get("id", ""),
                "doi": doi,
                "abstract_snippet": abstract[:ABSTRACT_SNIPPET_LENGTH],
                "citation_count": work.get("cited_by_count", 0),
                "journal": (
                    (work.get("primary_location") or {}).get("source") or {}
                ).get("display_name", ""),
                "volume": (work.get("biblio") or {}).get("volume", ""),
                "issue": (work.get("biblio") or {}).get("issue", ""),
                "pages": "-".join(
                    str(page) for page in (
                        (work.get("biblio") or {}).get("first_page"),
                        (work.get("biblio") or {}).get("last_page"),
                    )
                    if page
                ),
                "publication_type": work.get("type", ""),
                "source_api": "openalex",
                "accessed_date": datetime.utcnow().isoformat(),
            })
        return {"status": "success", "results": results}

    except asyncio.TimeoutError:
        logger.warning("openalex_timeout")
        return {"status": "timeout", "results": []}
    except Exception as e:
        logger.warning("openalex_exception", extra={"error": str(e)})
        return {"status": "error", "error": str(e), "results": []}
    finally:
        if own_session:
            await session.close()


async def search_arxiv(
    query: str,
    limit: int = 5,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Search arXiv for preprints via their Atom API."""
    import xml.etree.ElementTree as ET

    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
    }

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=API_TIMEOUT)

    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning("arxiv_error", extra={"status": resp.status})
                return {"status": "error", "error": f"HTTP {resp.status}", "results": []}
            text = await resp.text()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(text)
        results = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]
            link = ""
            for lnk in entry.findall("atom:link", ns):
                if lnk.get("type") == "text/html":
                    link = lnk.get("href", "")
                    break
            year = None
            if published_el is not None and published_el.text:
                year = int(published_el.text[:4])
            results.append({
                "title": (title_el.text or "").strip().replace("\n", " ") if title_el is not None else "Unknown",
                "authors": ", ".join(authors),
                "year": year,
                "url": link,
                "doi": "",
                "abstract_snippet": ((summary_el.text or "").strip()[:ABSTRACT_SNIPPET_LENGTH]) if summary_el is not None else "",
                "source_api": "arxiv",
                "accessed_date": datetime.utcnow().isoformat(),
            })
        return {"status": "success", "results": results}

    except asyncio.TimeoutError:
        logger.warning("arxiv_timeout")
        return {"status": "timeout", "results": []}
    except Exception as e:
        logger.warning("arxiv_exception", extra={"error": str(e)})
        return {"status": "error", "error": str(e), "results": []}
    finally:
        if own_session:
            await session.close()


# ──────────────────────────────────────────────────────────
# Core gathering logic
# ──────────────────────────────────────────────────────────

async def _gather_one_gap(
    gap: dict,
    existing_keys: set[str],
    sources_per_gap: int,
    session: aiohttp.ClientSession,
    errors: list[str],
) -> list[dict]:
    """
    Fetch sources for a single gap by querying all APIs concurrently.
    Deduplicates and enriches DOI/title matches against a shared identity set.
    """
    query = gap.get("search_query") or gap.get("topic", "")
    if not query:
        return []

    # Fire all APIs concurrently — wall time = slowest API, not sum
    ss_task, cr_task, oa_task, ax_task = await asyncio.gather(
        search_semantic_scholar(query, limit=sources_per_gap, session=session),
        search_crossref(query, limit=sources_per_gap, session=session),
        search_openalex(query, limit=sources_per_gap, session=session),
        search_arxiv(query, limit=sources_per_gap, session=session),
        return_exceptions=True,
    )

    results_by_source: dict[str, dict] = {}
    source_key_to_id: dict[str, str] = {}

    for api_name, api_result in zip(
        ("semantic_scholar", "crossref", "openalex", "arxiv"),
        (ss_task, cr_task, oa_task, ax_task),
    ):
        if isinstance(api_result, Exception):
            errors.append(f"{api_name}: {api_result}")
            continue
        if api_result.get("status") != "success":
            errors.append(
                f"{api_name}: {api_result.get('error') or api_result.get('status') or 'request failed'}"
            )
        for source in api_result.get("results", []):
            keys = _source_keys(source)
            if not keys:
                continue
            if keys & existing_keys:
                continue
            match_id = next(
                (source_key_to_id[key] for key in keys if key in source_key_to_id),
                None,
            )
            if match_id is None:
                match_id = next(iter(keys))
                results_by_source[match_id] = {
                    **source,
                    "source_apis": [api_name],
                }
            else:
                merged = results_by_source[match_id]
                if api_name not in merged.get("source_apis", []):
                    merged.setdefault("source_apis", []).append(api_name)
                if _source_metadata_score(source) > _source_metadata_score(merged):
                    merged["source_api"] = api_name
                for field, value in source.items():
                    if field in {"source_api", "source_apis"}:
                        continue
                    if field == "abstract_snippet":
                        if len(str(value or "")) > len(str(merged.get(field) or "")):
                            merged[field] = value
                    elif value and (
                        not merged.get(field)
                        or str(merged[field]).casefold() in {"unknown", "none"}
                    ):
                        merged[field] = value
            for key in keys:
                source_key_to_id[key] = match_id

    gap_sources = list(results_by_source.values())
    gap_sources.sort(key=_source_metadata_score, reverse=True)
    selected = gap_sources[:sources_per_gap]
    for source in selected:
        source["gap_topic"] = gap.get("topic", "")
        if gap.get("source_category"):
            source["source_category"] = gap["source_category"]
        existing_keys.update(_source_keys(source))
    return selected


async def gather_sources_for_gaps(
    gaps: list[dict],
    existing_sources: list[dict] | None = None,
    sources_per_gap: int = 3,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Gap-driven source gathering. All gaps processed concurrently
    (bounded by _GAP_CONCURRENCY semaphore). All APIs queried in
    parallel per gap.

    Returns:
        (found_sources, unfilled_gaps, errors)
    """
    started_at = time.perf_counter()
    existing_keys: set[str] = {
        key
        for s in (existing_sources or [])
        for key in _source_keys(s)
    }

    all_found: list[dict] = []
    unfilled: list[dict] = []
    errors: list[str] = []

    sem = asyncio.Semaphore(_GAP_CONCURRENCY)

    async def bounded_gather(gap: dict, session: aiohttp.ClientSession) -> tuple[dict, list[dict]]:
        async with sem:
            sources = await _gather_one_gap(gap, existing_keys, sources_per_gap, session, errors)
            return gap, sources

    async with aiohttp.ClientSession(timeout=API_TIMEOUT) as session:
        tasks = [bounded_gather(gap, session) for gap in gaps if gap.get("search_query") or gap.get("topic")]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            errors.append(str(item))
            continue
        gap, sources = item
        if sources:
            all_found.extend(sources)
        else:
            unfilled.append(gap)

    # Gaps without a query string
    for gap in gaps:
        if not (gap.get("search_query") or gap.get("topic")):
            unfilled.append(gap)

    logger.info(
        "source_gathering_complete",
        extra={
            "gaps_searched": len(gaps),
            "sources_found": len(all_found),
            "unfilled_gaps": len(unfilled),
            "errors": len(errors),
            "request_count": sum(
                4 for gap in gaps if gap.get("search_query") or gap.get("topic")
            ),
            "duration_s": round(time.perf_counter() - started_at, 2),
        },
    )
    return all_found, unfilled, errors
