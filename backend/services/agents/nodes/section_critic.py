# ============================================================
# Node: Section Critic (Stage 12 — Branch D)
# Bounded section-level critique keeps long drafts within output limits.
# ============================================================
from __future__ import annotations

import asyncio
import logging

from config import settings
from services.llm_service import call_llm
from services.agents.prompts.section_critic import (
    SECTION_CRITIC_CHUNK_CHARS,
    SECTION_CRITIC_CHUNK_SYSTEM,
    SECTION_CRITIC_THRESHOLD,
    build_section_critic_chunk_prompt,
)
from services.agents.retry import (
    increment_attempt,
    build_flag_state_update,
)
from services.agents.json_utils import coerce_score, parse_llm_json

logger = logging.getLogger(__name__)
NODE_NAME = "section_critic"
SECTION_CRITIC_CHUNK_MAX_TOKENS = 900


def _split_critic_text(text: str, max_chars: int) -> list[str]:
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


def _critic_tasks(state: dict, draft: str) -> list[tuple[str, dict, int, int, str]]:
    outline = state.get("outline", {})
    outline_sections = {
        str(section.get("name", "")).casefold(): section
        for section in outline.get("sections", [])
        if isinstance(section, dict)
    }
    generated_sections = state.get("generated_sections", [])
    sections = [
        (str(section.get("section_name") or "Draft"), str(section.get("text") or ""))
        for section in generated_sections
        if isinstance(section, dict) and str(section.get("text") or "").strip()
    ]
    if not sections and draft.strip():
        sections = [("Draft", draft)]

    tasks = []
    for section_name, text in sections:
        parts = _split_critic_text(text, SECTION_CRITIC_CHUNK_CHARS)
        section_spec = outline_sections.get(section_name.casefold(), {})
        for part_index, part_text in enumerate(parts, start=1):
            tasks.append((
                section_name,
                section_spec,
                part_index,
                len(parts),
                build_section_critic_chunk_prompt(
                    section_name,
                    part_text,
                    section_spec,
                    "",
                    part_index,
                    len(parts),
                ),
            ))
    return tasks


async def _evaluate_critic_task(
    task: tuple[str, dict, int, int, str],
) -> tuple[str, dict | None, str | None]:
    section_name, _, part_index, _, prompt = task
    try:
        raw = await call_llm(
            prompt=prompt,
            agent_role="section_critic",
            system=SECTION_CRITIC_CHUNK_SYSTEM,
            temperature=0.0,
            max_tokens=SECTION_CRITIC_CHUNK_MAX_TOKENS,
            context_limit=settings.MODAL_CONTEXT,
            reasoning_effort="none",
        )
    except RuntimeError as error:
        return section_name, None, f"{section_name} excerpt {part_index}: {error}"

    result = parse_llm_json(raw)
    score = coerce_score(result.get("overall_score")) if result else None
    if score is None:
        return section_name, None, f"{section_name} excerpt {part_index}: invalid critique response"
    result["overall_score"] = max(0.0, min(10.0, score))
    return section_name, result, None


async def section_critic_node(state: dict) -> dict:
    draft = state.get("draft_text", "")
    attempt_update = increment_attempt(state, NODE_NAME)

    tasks = _critic_tasks(state, draft)
    outcomes = await asyncio.gather(
        *(_evaluate_critic_task(task) for task in tasks),
    ) if tasks else []

    section_parts: dict[str, list[dict]] = {}
    errors = []
    for section_name, result, error in outcomes:
        if error:
            errors.append(error)
        elif result is not None:
            section_parts.setdefault(section_name, []).append(result)

    sections = {}
    for section_name, parts in section_parts.items():
        scores = [part["overall_score"] for part in parts]
        section_score = sum(scores) / len(scores)
        sections[section_name] = {
            "average_score": section_score,
            "issues": [
                issue
                for part in parts
                for issue in part.get("issues", [])
            ],
            "missing_elements": [
                item
                for part in parts
                for item in part.get("missing_elements", [])
            ],
            "suggestions": [
                item
                for part in parts
                for item in part.get("suggestions", [])
            ],
            "critical_failures": [
                item
                for part in parts
                for item in part.get("critical_failures", [])
            ],
        }

    evaluated_scores = [
        section["average_score"] for section in sections.values()
    ]
    score = (
        sum(evaluated_scores) / len(evaluated_scores)
        if evaluated_scores else None
    )
    unknown = bool(errors) or score is None
    result = {
        "sections": sections,
        "overall_score": score,
        "unknown": unknown,
        "summary": (
            f"Reviewed {len(sections)} section(s) in {len(tasks)} bounded excerpt(s)."
            if not errors
            else (
                f"Reviewed {len(sections)} section(s); {len(errors)} excerpt(s) "
                "could not be evaluated."
            )
        ),
    }
    if unknown:
        result["passed"] = False
        if errors:
            result["failed_excerpts"] = errors
    else:
        result["passed"] = score >= SECTION_CRITIC_THRESHOLD

    feedback = result["summary"]
    if errors:
        feedback = "\n".join(errors)
    elif score is not None and score < SECTION_CRITIC_THRESHOLD:
        feedback = "\n".join(
            f"{name}: {section['issues'] or section['suggestions']}"
            for name, section in sections.items()
            if section["average_score"] < SECTION_CRITIC_THRESHOLD
        ) or result["summary"]
    extra_state = (
        build_flag_state_update(NODE_NAME, feedback, state)
        if not result["passed"]
        else {}
    )

    score_label = f"{score:g}" if score is not None else "unknown"
    return {
        "section_critic_result": result,
        "current_step": "section_critic",
        **attempt_update,
        **extra_state,
        "steps_log": [
            f"✓ Section critique: {score_label}/10 overall "
            f"({'PASSED' if result['passed'] else 'NEEDS REVIEW'})"
        ],
    }
