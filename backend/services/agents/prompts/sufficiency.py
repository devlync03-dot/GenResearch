import json

# ============================================================
# Prompt: Sufficiency Evaluator
# ============================================================

MAX_SUFFICIENCY_CHUNK_CHARS = 6_000
SUFFICIENCY_REDUCE_GROUP_SIZE = 8

SUFFICIENCY_SYSTEM = """You are GenResearch Sufficiency Evaluator. You assess whether the user's \
provided material (answers, uploaded documents, references) is sufficient to produce a well-grounded \
research paper on their topic.

This is NOT a vibe check. You evaluate against a structured rubric:

1. **Source coverage**: Does every major section/argument have at least one supporting source?
2. **Background coverage**: Is there material addressing background/context, not just the core claim?
3. **Counter-arguments**: Is there anything addressing counter-arguments or complicating evidence?
4. **Recency**: Is source recency reasonable for the topic? (e.g., AI topics need recent sources)

You MUST output valid JSON — nothing else.

Output format:
{
  "sections": {
    "<section_name>": {
      "confidence": "well_supported | adequate | weak",
      "reasoning": "Why this rating",
      "sources_found": 2,
      "sources_needed": 3
    }
  },
  "overall_assessment": "sufficient | needs_more",
  "recency_adequate": true,
  "counter_arguments_present": false,
  "summary": "One paragraph summary of the assessment"
}"""


SUFFICIENCY_CHUNK_SYSTEM = f"""{SUFFICIENCY_SYSTEM}

This request contains only one part of the user's material. Assess only what is present in this part.
Do not treat information that may be in another part as absent. Keep section reasoning concise, and
return only the required JSON."""


SUFFICIENCY_MERGE_SYSTEM = f"""{SUFFICIENCY_SYSTEM}

You are combining partial evaluations of different portions of the same user's material. Treat the
reports as complementary parts, not independent full-document evaluations. Merge section findings,
avoid counting the same source twice, and make the overall assessment using the combined evidence.
Keep reasoning and summary concise. Return only the required JSON."""


def _split_text(text: str, max_chars: int) -> list[str]:
    remaining = text
    if not remaining.strip():
        return [remaining]

    parts = []
    while len(remaining) > max_chars:
        boundary = remaining.rfind(" ", 0, max_chars + 1)
        if boundary <= 0:
            boundary = max_chars
        else:
            boundary += 1
        parts.append(remaining[:boundary])
        remaining = remaining[boundary:]
    if remaining:
        parts.append(remaining)
    return parts


def build_sufficiency_prompt_chunks(
    topic: str,
    answers: dict,
    available_sources: list[dict],
    retry_feedback: str = "",
    max_chunk_chars: int = MAX_SUFFICIENCY_CHUNK_CHARS,
) -> list[str]:
    """Split long user material into bounded prompts without dropping any entries."""
    entries: list[tuple[str, str, str]] = []
    for key, value in (answers or {}).items():
        answer = str(value)
        answer_prefix = f"{key}"
        value_budget = max(1, max_chunk_chars - len(answer_prefix) - 40)
        parts = _split_text(answer, value_budget)
        for index, part in enumerate(parts, start=1):
            part_label = (
                f"{answer_prefix} (part {index}/{len(parts)})"
                if len(parts) > 1
                else answer_prefix
            )
            entries.append(("answer", part_label, part))

    for source in available_sources or []:
        title = str(source.get("title", "Unknown"))
        tag = str(source.get("tag", "unknown"))
        title_budget = max(1, max_chunk_chars - len(tag) - 48)
        parts = _split_text(title, title_budget)
        for index, part in enumerate(parts, start=1):
            part_title = (
                f"{part} (part {index}/{len(parts)})"
                if len(parts) > 1
                else part
            )
            entries.append(("source", part_title, tag))

    chunks: list[tuple[dict[str, str], list[dict[str, str]], int]] = []
    chunk_answers: dict[str, str] = {}
    chunk_sources: list[dict[str, str]] = []
    chunk_size = 0

    for kind, label, value in entries:
        entry_size = len(label) + len(value) + 32
        if chunk_size and chunk_size + entry_size > max_chunk_chars:
            chunks.append((chunk_answers, chunk_sources, chunk_size))
            chunk_answers = {}
            chunk_sources = []
            chunk_size = 0

        if kind == "answer":
            chunk_answers[label] = value
        else:
            chunk_sources.append({"title": label, "tag": value})
        chunk_size += entry_size

    if chunk_size or not chunks:
        chunks.append((chunk_answers, chunk_sources, chunk_size))

    total = len(chunks)
    return [
        build_sufficiency_prompt(
            topic=topic,
            answers=chunk_answer,
            available_sources=chunk_source,
            retry_feedback=retry_feedback,
            chunk_index=index,
            chunk_count=total,
        )
        for index, (chunk_answer, chunk_source, _) in enumerate(chunks, start=1)
    ]


def build_sufficiency_prompt(
    topic: str,
    answers: dict,
    available_sources: list[dict],
    retry_feedback: str = "",
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> str:
    answers_text = "\n".join(
        f"- {k}: {v}" for k, v in answers.items()
    ) if answers else "No questionnaire answers provided."

    sources_text = "\n".join(
        f"- {s.get('title', 'Unknown')} (type: {s.get('tag', 'unknown')})"
        for s in available_sources
    ) if available_sources else "No sources available yet."

    feedback_section = ""
    if retry_feedback:
        feedback_section = f"""

IMPORTANT — PREVIOUS EVALUATION FEEDBACK:
This is your second attempt. The previous evaluation was returned because:
{retry_feedback}
Address this feedback in your new evaluation."""

    chunk_section = (
        f"\nMaterial part {chunk_index} of {chunk_count}. "
        "Consider only the answers and sources shown in this part."
        if chunk_index is not None and chunk_count is not None
        else ""
    )

    return f"""Evaluate the sufficiency of the user's material for the following research topic.

Research Topic: "{topic}"

User's Answers:
{answers_text}

Available Sources:
{sources_text}
{feedback_section}
{chunk_section}

Produce your sufficiency evaluation JSON now:"""


def build_sufficiency_merge_prompt(
    topic: str,
    partial_reports: list[dict],
    retry_feedback: str = "",
) -> str:
    feedback = (
        f"\nPrevious retry feedback: {retry_feedback}"
        if retry_feedback
        else ""
    )
    return f"""Combine these partial sufficiency evaluations for the complete research material.

Research Topic: "{topic}"
Partial reports:
{json.dumps(partial_reports, ensure_ascii=False)}
{feedback}

Return one concise final sufficiency evaluation in the required JSON format:"""
