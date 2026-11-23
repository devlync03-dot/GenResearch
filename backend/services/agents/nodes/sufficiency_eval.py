# ============================================================
# Node: Sufficiency Evaluator (Stage 3)
# Rubric-based evaluation with retry support (max 2 attempts).
# Model: nemotron-3-nano
# ============================================================
from __future__ import annotations

import logging
import time

from services.llm_service import call_llm
from services.agents.prompts.sufficiency import (
    SUFFICIENCY_CHUNK_SYSTEM,
    SUFFICIENCY_MERGE_SYSTEM,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_REDUCE_GROUP_SIZE,
    build_sufficiency_merge_prompt,
    build_sufficiency_prompt_chunks,
)
from services.agents.retry import (
    increment_attempt,
    should_retry,
    build_retry_state_update,
    build_flag_state_update,
)
from services.agents.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

NODE_NAME = "sufficiency_eval"
MIN_SOURCES_FOR_SUFFICIENT = 3
SUFFICIENCY_CHUNK_MAX_TOKENS = 1200
SUFFICIENCY_MERGE_MAX_TOKENS = 1800


async def _evaluate_material_chunks(
    topic: str,
    answers: dict,
    user_sources: list[dict],
    retry_feedback: str,
) -> tuple[dict | None, bool]:
    prompts = build_sufficiency_prompt_chunks(
        topic=topic,
        answers=answers,
        available_sources=user_sources,
        retry_feedback=retry_feedback,
    )
    if len(prompts) == 1:
        raw = await call_llm(
            prompt=prompts[0],
            agent_role="sufficiency_evaluator",
            system=SUFFICIENCY_SYSTEM,
            temperature=0.0,
            max_tokens=SUFFICIENCY_MERGE_MAX_TOKENS,
            reasoning_effort="none",
        )
        report = parse_llm_json(raw)
        return report, report is None

    logger.info(
        "sufficiency_material_split",
        extra={"chunk_count": len(prompts)},
    )
    partial_reports = []
    for index, prompt in enumerate(prompts, start=1):
        raw = await call_llm(
            prompt=prompt,
            agent_role="sufficiency_evaluator",
            system=SUFFICIENCY_CHUNK_SYSTEM,
            temperature=0.0,
            max_tokens=SUFFICIENCY_CHUNK_MAX_TOKENS,
            reasoning_effort="none",
        )
        partial = parse_llm_json(raw)
        if partial is None:
            logger.warning(
                "sufficiency_chunk_json_parse_failed",
                extra={"chunk": index, "chunk_count": len(prompts), "raw": raw[:500]},
            )
            return None, True
        partial_reports.append(partial)

    while len(partial_reports) > 1:
        final_merge = len(partial_reports) <= SUFFICIENCY_REDUCE_GROUP_SIZE
        merged_reports = []
        for start in range(0, len(partial_reports), SUFFICIENCY_REDUCE_GROUP_SIZE):
            report_group = partial_reports[start:start + SUFFICIENCY_REDUCE_GROUP_SIZE]
            if len(report_group) == 1 and not final_merge:
                merged_reports.append(report_group[0])
                continue
            raw = await call_llm(
                prompt=build_sufficiency_merge_prompt(
                    topic=topic,
                    partial_reports=report_group,
                    retry_feedback=retry_feedback,
                ),
                agent_role="sufficiency_evaluator",
                system=SUFFICIENCY_MERGE_SYSTEM,
                temperature=0.0,
                max_tokens=SUFFICIENCY_MERGE_MAX_TOKENS,
                reasoning_effort="none",
            )
            merged = parse_llm_json(raw)
            if merged is None:
                logger.warning(
                    "sufficiency_merge_json_parse_failed",
                    extra={"raw": raw[:500]},
                )
                return None, True
            merged_reports.append(merged)
        partial_reports = merged_reports

    return partial_reports[0], False


async def sufficiency_eval_node(state: dict) -> dict:
    """
    Stage 3: Evaluate sufficiency of user's provided material.
    Per-section confidence scoring with structured rubric.
    Supports retry loop (max 2) with structured feedback.
    """
    topic = state["topic"]
    answers = state.get("questionnaire_answers", {})
    user_sources = state.get("user_provided_sources", [])
    retry_fb = state.get("retry_feedback", {}).get(NODE_NAME, "")

    # Track attempt
    attempt_update = increment_attempt(state, NODE_NAME)

    t0 = time.perf_counter()
    report, parse_failed = await _evaluate_material_chunks(
        topic=topic,
        answers=answers,
        user_sources=user_sources,
        retry_feedback=retry_fb,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "sufficiency_eval_llm_done",
        extra={"elapsed_s": round(elapsed, 2)},
    )
    if parse_failed:
        logger.warning("sufficiency_json_parse_failed")
        report = {
            "sections": {},
            "overall_assessment": "needs_more",
            "evaluation_unavailable": True,
            "summary": "Evaluation returned invalid JSON; continuing with source gathering.",
        }

    # The LLM can be too lenient about coverage when the supplied source set
    # is small. Keep the override in the report itself because downstream
    # routing reads sufficiency_report.overall_assessment.
    source_count_override = False
    if (
        report.get("overall_assessment") == "sufficient"
        and len(user_sources) < MIN_SOURCES_FOR_SUFFICIENT
    ):
        logger.warning(
            "sufficiency_override_insufficient_sources",
            extra={
                "user_source_count": len(user_sources),
                "min_required": MIN_SOURCES_FOR_SUFFICIENT,
            },
        )
        report["overall_assessment"] = "needs_more"
        report["source_count_override"] = True
        source_count_override = True
        report["summary"] = (
            f"Overridden: model judged material sufficient, but only "
            f"{len(user_sources)} source(s) were provided (minimum "
            f"{MIN_SOURCES_FOR_SUFFICIENT} required to skip the scrape-permission "
            f"step). Original assessment: {report.get('summary', '')}"
        )

    # A5 Fix: compute retry/flag locally
    passed = report.get("overall_assessment") == "sufficient"
    feedback = (
        "Sufficiency evaluation returned invalid JSON; used the needs-more fallback."
        if parse_failed
        else "Material is insufficient. Missing background or sources."
    )
    if parse_failed:
        decision = "flag"
    elif source_count_override:
        decision = "proceed"
    else:
        decision = should_retry(
            state,
            NODE_NAME,
            passed,
            feedback,
            attempts_done=state.get("retry_counts", {}).get(NODE_NAME, 0) + 1,
        )

    extra_state = {}
    if decision == "retry":
        extra_state = build_retry_state_update(NODE_NAME, feedback, state)
    elif decision == "flag":
        extra_state = build_flag_state_update(NODE_NAME, feedback, state)

    return {
        "sufficiency_report": report,
        "current_step": "sufficiency_eval",
        **attempt_update,
        **extra_state,
        "steps_log": [
            f"✓ Sufficiency evaluation: {report.get('overall_assessment', 'unknown')}"
        ],
    }
