# ============================================================
# Node: Citation/Claim Verification (Stage 11 — Branch C)
# MUST run on nemotron-3-nano (different from Draft Agent's model).
#
# Citation checks run once; incomplete or failing checks are flagged for review.
# ============================================================
from __future__ import annotations
import asyncio
import logging

from config import settings
from services.llm_service import call_llm
from services.agents.prompts.citation_verify import (
    CITATION_VERIFY_CHUNK_CHARS,
    CITATION_VERIFY_CHUNK_SYSTEM,
    CITATION_VERIFY_THRESHOLD,
    build_citation_verify_prompt,
)
from services.citation_formatting import CITATION_TAG
from services.agents.retry import (
    increment_attempt, build_flag_state_update,
)
from services.agents.json_utils import coerce_score, parse_llm_json

logger = logging.getLogger(__name__)
NODE_NAME = "citation_verification"
CITATION_VERIFY_CHUNK_MAX_TOKENS = 1000


def _split_verification_text(text: str, max_chars: int) -> list[str]:
    remaining = text.strip()
    if not remaining:
        return []
    chunks = []
    while len(remaining) > max_chars:
        boundary = remaining.rfind("\n\n", 0, max_chars + 1)
        if boundary < max_chars // 2:
            boundary = remaining.rfind(" ", 0, max_chars + 1)
        if boundary <= 0:
            boundary = max_chars
        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _verification_chunks(state: dict, draft: str) -> list[tuple[str, str]]:
    generated_sections = state.get("generated_sections", [])
    chunks = []
    if generated_sections:
        for section in generated_sections:
            name = str(section.get("section_name") or "Draft")
            for index, text in enumerate(
                _split_verification_text(
                    str(section.get("text") or ""),
                    CITATION_VERIFY_CHUNK_CHARS,
                ),
                start=1,
            ):
                location = name if index == 1 else f"{name}, part {index}"
                chunks.append((location, text))
    if not chunks:
        for index, text in enumerate(
            _split_verification_text(draft, CITATION_VERIFY_CHUNK_CHARS),
            start=1,
        ):
            chunks.append((f"Draft part {index}", text))
    return chunks


async def _verify_draft_chunks(
    chunks: list[tuple[str, str]],
    registry: list[dict],
) -> tuple[list[dict], list[float], bool]:
    semaphore = asyncio.Semaphore(max(1, settings.MODAL_MAX_CONCURRENT))

    async def verify_chunk(
        location: str,
        text: str,
    ) -> tuple[list[dict], float | None, bool]:
        prompt = build_citation_verify_prompt(
            text,
            registry,
            "",
            location=location,
        )
        async with semaphore:
            try:
                raw = await call_llm(
                    prompt=prompt,
                    agent_role="citation_verification",
                    system=CITATION_VERIFY_CHUNK_SYSTEM,
                    temperature=0.0,
                    max_tokens=CITATION_VERIFY_CHUNK_MAX_TOKENS,
                    context_limit=settings.MODAL_CONTEXT,
                    reasoning_effort="none",
                )
            except RuntimeError as error:
                logger.warning(
                    "citation_verification_chunk_request_failed",
                    extra={"location": location, "error": str(error)},
                )
                return [], None, True

        parsed = parse_llm_json(raw)
        score = coerce_score(
            parsed.get("coverage_score"), percentage=True
        ) if parsed else None
        if parsed is None or score is None:
            logger.warning(
                "citation_verification_chunk_parse_failed",
                extra={"location": location, "raw": raw[:300]},
            )
            return [], None, True

        chunk_issues = []
        for issue in parsed.get("unverified_claims", []):
            if isinstance(issue, dict):
                issue.setdefault("location", location)
                chunk_issues.append(issue)
        for invalid in parsed.get("invalid_citations", []):
            if isinstance(invalid, dict):
                invalid["location"] = location
                chunk_issues.append({
                    "claim": f"[{invalid.get('id', 'unknown')}]",
                    "location": location,
                    "reason": invalid.get("reason", "Citation ID is not in the registry."),
                })
        return chunk_issues, max(0.0, min(1.0, score)), False

    chunk_results = await asyncio.gather(
        *(verify_chunk(location, text) for location, text in chunks)
    )
    issues = [
        issue
        for chunk_issues, _, _ in chunk_results
        for issue in chunk_issues
    ]
    scores = [
        score
        for _, score, _ in chunk_results
        if score is not None
    ]
    unknown = any(chunk_unknown for _, _, chunk_unknown in chunk_results)

    valid_ids = {
        str(source.get("id", "")).upper()
        for source in registry
        if source.get("id")
    }
    all_text = "\n".join(text for _, text in chunks)
    invalid_ids = {
        match.upper()
        for match in CITATION_TAG.findall(all_text)
        if match.upper() not in valid_ids
    }
    existing_invalid = {
        str(issue.get("claim", "")).strip("[]").upper()
        for issue in issues
        if isinstance(issue, dict) and str(issue.get("claim", "")).startswith("[")
    }
    for invalid_id in sorted(invalid_ids - existing_invalid):
        issues.append({
            "claim": f"[{invalid_id}]",
            "location": "Draft",
            "reason": "Citation ID is not in the citation registry.",
        })

    return issues, scores, unknown or not scores


async def citation_verify_node(state: dict) -> dict:
    draft = state.get("draft_text", "")
    registry = state.get("citation_registry", [])
    attempt_update = increment_attempt(state, NODE_NAME)

    chunks = _verification_chunks(state, draft)
    unverified_claims, scores, unknown = await _verify_draft_chunks(
        chunks,
        registry,
    )
    score = sum(scores) / len(scores) if scores and not unknown else None
    result = {
        "unverified_claims": unverified_claims,
        "invalid_citations": [
            issue for issue in unverified_claims
            if str(issue.get("claim", "")).startswith("[")
        ],
    }
    if unknown:
        result.update({
            "coverage_score": None,
            "unknown": True,
            "summary": (
                "One or more citation-verification chunks could not be evaluated; "
                "review is required."
            ),
            "unverified_claims": unverified_claims,
        })
        result["passed"] = False
    else:
        result["coverage_score"] = score
        result["passed"] = score >= CITATION_VERIFY_THRESHOLD
        result["summary"] = (
            f"Verified {len(chunks)} draft excerpt(s); "
            f"coverage is {score:.0%}."
        )

    generated_sections = state.get("generated_sections", [])
    failed_sections = set()
    for issue in unverified_claims:
        if not isinstance(issue, dict):
            continue
        claim = str(issue.get("claim", "")).strip().lower()
        location = str(issue.get("location", "")).lower()
        for section in generated_sections:
            name = str(section.get("section_name", ""))
            if name and (
                name.lower() in location
                or (claim and claim in str(section.get("text", "")).lower())
            ):
                failed_sections.add(name)

    feedback = "\n".join(
        str(issue.get("claim", ""))
        for issue in unverified_claims
        if isinstance(issue, dict)
    )
    if not feedback:
        feedback = result["summary"]
    result["review_sections"] = sorted(failed_sections)
    extra_state = (
        build_flag_state_update(NODE_NAME, feedback, state)
        if not result["passed"]
        else {}
    )

    score_label = f"{score:.0%}" if score is not None else "unknown"

    return {
        "citation_verification_result": result,
        "current_step": "citation_verification",
        **attempt_update,
        **extra_state,
        "steps_log": [
            f"✓ Citation verification: {score_label} coverage "
            f"({'PASSED' if result['passed'] else 'NEEDS REVIEW'})"
        ],
    }
