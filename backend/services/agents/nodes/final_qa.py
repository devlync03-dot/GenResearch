# ============================================================
# Node: Final QA (Stage 13)
# Whole-paper check — only meaningful after both C and D complete.
# Model: nemotron-3-nano
#
# Final QA runs once; unknown or failing assessments are flagged for review.
# ============================================================
from __future__ import annotations
import logging
import math

from config import settings
from services.llm_service import call_llm
from services.agents.prompts.final_qa import (
    FINAL_QA_SYSTEM, FINAL_QA_THRESHOLD, build_final_qa_prompt,
)
from services.agents.retry import (
    increment_attempt,
    build_flag_state_update,
)
from services.agents.json_utils import coerce_score, parse_llm_json

logger = logging.getLogger(__name__)
NODE_NAME = "final_qa"


def _bounded_draft_excerpts(state: dict, max_chars: int) -> str:
    sections = [
        (str(section.get("section_name") or "Draft"), str(section.get("text") or "").strip())
        for section in state.get("generated_sections", [])
        if isinstance(section, dict) and str(section.get("text") or "").strip()
    ]
    if not sections:
        sections = [("Draft", str(state.get("draft_text", "")).strip())]
    if not sections or not sections[0][1]:
        return ""

    heading_cost = sum(len(name) + 10 for name, _ in sections)
    if max_chars <= heading_cost + 100:
        return ""
    text_budget = max(0, max_chars - heading_cost - 100)
    chars_per_section = text_budget // len(sections)
    if chars_per_section < 40:
        return ""
    excerpts = []
    for name, text in sections:
        if len(text) <= chars_per_section:
            excerpt = text
        else:
            start_chars = chars_per_section // 2
            end_chars = chars_per_section - start_chars
            ending = text[-end_chars:] if end_chars else ""
            excerpt = f"{text[:start_chars]}\n...\n{ending}"
        excerpts.append(f"## {name}\n{excerpt}")
    return (
        "[Bounded opening and closing excerpts from each section; "
        "use these with the full-section citation and critic results below.]\n\n"
        + "\n\n".join(excerpts)
    )


async def final_qa_node(state: dict) -> dict:
    cv_result = state.get("citation_verification_result", {})
    sc_result = state.get("section_critic_result", {})
    attempt_update = increment_attempt(state, NODE_NAME)

    output_tokens = min(1500, max(512, settings.MODAL_CONTEXT // 8))
    system_tokens = math.ceil(len(FINAL_QA_SYSTEM) / 3.5)
    draft_char_budget = max(
        0,
        min(
            14000,
            int(
                (
                    settings.MODAL_CONTEXT
                    - output_tokens
                    - system_tokens
                    - 700
                )
                * 3
            ),
        ),
    )
    bounded_draft = _bounded_draft_excerpts(state, draft_char_budget)
    prompt = build_final_qa_prompt(bounded_draft, cv_result, sc_result, "")
    request_error = None
    if not bounded_draft:
        request_error = "The configured context leaves no room for a draft excerpt."
        result = None
    else:
        try:
            raw = await call_llm(
                prompt=prompt, agent_role="final_qa",
                system=FINAL_QA_SYSTEM, temperature=0.2, max_tokens=output_tokens,
                context_limit=settings.MODAL_CONTEXT,
            )
            result = parse_llm_json(raw)
        except RuntimeError as error:
            logger.warning("final_qa_request_failed", extra={"error": str(error)})
            request_error = str(error)
            result = None

    score = coerce_score(result.get("overall_score")) if result else None
    unknown = result is None or score is None
    if result is None:
        result = {}
    if unknown:
        result.update({
            "overall_score": None,
            "unknown": True,
            "passed": False,
            "summary": (
                f"Final QA assessment unavailable: {request_error}"
                if request_error
                else "Final QA response was not parseable or had no valid score."
            ),
            "issues": result.get("issues") or [],
        })
    else:
        result["overall_score"] = score
        result["passed"] = score >= FINAL_QA_THRESHOLD

    feedback = "\n".join(str(i) for i in result.get("issues", []))
    if not feedback:
        feedback = str(
            result.get("summary")
            or f"Final QA score {score:g} is below the required threshold."
        )
    extra_state = (
        build_flag_state_update(NODE_NAME, feedback, state)
        if not result["passed"]
        else {}
    )

    score_label = f"{score:g}" if score is not None else "unknown"
    return {
        "final_qa_result": result,
        "current_step": "final_qa",
        **attempt_update,
        **extra_state,
        "steps_log": [
            f"✓ Final QA: {score_label}/10 "
            f"({'PASSED' if result['passed'] else 'NEEDS REVIEW'})"
        ],
    }
