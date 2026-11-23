from __future__ import annotations

from dataclasses import dataclass

from config import settings


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    max_concurrent: int
    num_ctx: int = 4096
    provider: str = "modal"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "modal": ModelSpec(
        "modal",
        settings.MODAL_MODEL,
        max_concurrent=settings.MODAL_MAX_CONCURRENT,
        num_ctx=settings.MODAL_CONTEXT,
        provider="modal",
    ),
}


ROLE_MODEL_CHAINS: dict[str, list[str]] = {
    "chat": ["modal"],
    "chat_map": ["modal"],
    "chat_reduce": ["modal"],
    "topic_analysis": ["modal"],
    "summarization": ["modal"],
    "summarization_final": ["modal"],
    "citation_extraction": ["modal"],
    "literature_review": ["modal"],
    "proposal_composition": ["modal"],
    "composer": ["modal"],
    "proposal_review": ["modal"],
    "draft": ["modal"],
    "citation_verification": ["modal"],
    "verification": ["modal"],
    "sufficiency_evaluator": ["modal"],
    "gap_report": ["modal"],
    "outline_plan": ["modal"],
    "source_quality_evaluator": ["modal"],
    "user_doc_quality_eval": ["modal"],
    "questionnaire": ["modal"],
    "section_critic": ["modal"],
    "final_qa": ["modal"],
}


def get_chain_for_role(agent_role: str) -> list[ModelSpec]:
    keys = ROLE_MODEL_CHAINS.get(agent_role)
    if not keys:
        raise ValueError(
            f"Unknown agent role '{agent_role}'. "
            f"Valid roles: {sorted(ROLE_MODEL_CHAINS.keys())}"
        )
    return [MODEL_REGISTRY[key] for key in keys]


def list_registry() -> dict[str, dict]:
    return {
        key: {
            "model_id": spec.model_id,
            "max_concurrent": spec.max_concurrent,
        }
        for key, spec in MODEL_REGISTRY.items()
    }
