"""Audit drafted claims for evidence and visual artifacts the pipeline can verify."""
from __future__ import annotations

import re

from services.citation_formatting import CITATION_TAG
from services.text_utils import split_sentences


_QUANTITATIVE_CLAIM = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s*%|\s+(?:percent|percentage points?|"
    r"million|billion|thousand|times|fold|years?|months?|days?)\b)"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:to|versus|vs\.?|compared (?:to|with))"
    r"\s*\d[\d,]*(?:\.\d+)?\b"
    r"|\b(?:increased|decreased|grew|declined|rose|fell|doubled|halved)\b",
    re.IGNORECASE,
)
_EQUATION = re.compile(
    r"(?:\\begin\{(?:equation|align)\}|\\\(|\\\[|"
    r"\$[^$]*=[^$]*\$|\b[A-Za-z][A-Za-z0-9_{}^]*\s*=\s*[-+]?\d)"
)
_VISUAL_SECTIONS = {"results", "discussion", "literature review"}


def _citation_ids(sentence: str) -> set[str]:
    return {match.upper() for match in CITATION_TAG.findall(sentence)}


def _quantitative_gaps(
    sections: list[dict], registry: list[dict]
) -> list[dict]:
    registry_by_id = {
        str(source.get("id", "")).upper(): source
        for source in registry
        if source.get("id")
    }
    gaps = []

    for section in sections:
        section_name = str(section.get("section_name") or "Unknown section")
        for sentence in split_sentences(str(section.get("text") or "")):
            sentence = sentence.strip()
            without_citations = CITATION_TAG.sub("", sentence)
            if not _QUANTITATIVE_CLAIM.search(without_citations):
                continue

            cited_sources = [
                registry_by_id[source_id]
                for source_id in _citation_ids(sentence)
                if source_id in registry_by_id
            ]
            source_hint = (
                ", ".join(
                    str(source.get("id"))
                    for source in cited_sources
                )
                or "no registered source citation"
            )
            section_lower = section_name.strip().lower()
            priority = 1 if any(name in section_lower for name in _VISUAL_SECTIONS) else 2
            gaps.append({
                "section": section_name,
                "claim": sentence,
                "evidence_type_needed": (
                    "Verify the exact value, unit, year, source title, DOI/URL, "
                    f"retrieval date, and page/table location ({source_hint}); "
                    "only then consider a chart or table."
                ),
                "priority": priority,
            })

        for line in str(section.get("text") or "").splitlines():
            equation = line.strip()
            if not equation or not _EQUATION.search(equation):
                continue
            gaps.append({
                "section": section_name,
                "claim": equation,
                "evidence_type_needed": (
                    "Verify whether this equation is published or illustrative, "
                    "cite its origin, define every variable, and validate units "
                    "and a numeric example before use."
                ),
                "priority": 2,
            })

    return gaps


async def evidence_visuals_node(state: dict) -> dict:
    """Create an honest gap report without inventing or externally fetching evidence."""
    sections = state.get("generated_sections", [])
    registry = state.get("citation_registry", [])
    gaps = _quantitative_gaps(sections, registry)
    unresolved = [
        {
            "claim": gap["claim"],
            "reason": (
                "The draft pipeline does not retain an exact supporting passage "
                "or source location for this claim, so it cannot be marked "
                "VERIFIED or used to create a quantitative artifact."
            ),
        }
        for gap in gaps
    ]

    report = {
        "gap_report": gaps,
        "sources_added": [],
        "datasets": [],
        "equations": [],
        "figures": [],
        "tables": [],
        "patches": [],
        "unresolved": unresolved,
        "warnings": (
            [
                "No quantitative artifacts were generated: session source chunks "
                "do not retain page/table locations or extracted datapoints."
            ]
            if gaps
            else [
                "No quantitative claims or equations were identified for "
                "evidence-backed charts, tables, or equations."
            ]
        ),
    }

    return {
        "evidence_visual_report": report,
        "current_step": "evidence_visuals",
        "steps_log": [
            f"✓ Evidence & visuals audit: {len(gaps)} claim(s)/equation(s) "
            "need source-location verification; no unverified artifacts added"
        ],
    }
