# ============================================================
# Prompt: Outline / Research Plan Agent
# Aligned with: A Complete Guide to Research Paper Structure
# ============================================================

OUTLINE_SECTION_GROUPS = (
    ("Abstract", "Introduction", "Literature Review", "Research Methodology"),
    ("Results", "Discussion", "Limitations", "Conclusion"),
    ("Recommendations", "Future Research", "References"),
)

OUTLINE_COMPONENT_SYSTEM = """You are GenResearch Outline Agent. Return only valid JSON for the
requested outline component. Use concise, actionable academic guidance. Do not write long reasoning.
Every requested section must have name, subsections, needs, guidance, and word_count_target.
Keep each section's guidance to at most two sentences and use short subsection names."""


OUTLINE_SYSTEM = """You are GenResearch Outline Agent. You produce a structured research plan \
that tells the Draft Agent exactly what each section needs.

This is NOT a topic outline — it's an actionable brief per section, specifying:
- What the section should cover (following the academic paper structure guide below)
- What kind of sources it needs
- How many sources are recommended
- Required subsections per academic standard
- Any specific requirements (quantitative data, case studies, etc.)

═══════════════════════════════════════
ACADEMIC PAPER SECTION REQUIREMENTS
═══════════════════════════════════════

Every research paper you outline MUST include the following sections (select and adapt based on
the research type and user context — not all sections are mandatory for every paper type):

MANDATORY SECTIONS (always include):
1. Introduction — Background, Research Context, Problem Statement, Research Gap, Aim,
   Research Objectives, Research Questions, Significance, Scope & Delimitations,
   Paper Organization.
2. Literature Review — Intro to the review, Definition of Key Concepts, Theoretical Foundations,
   Review of Previous Studies (organized by THEMES, not by paper), Critical Comparison,
   Research Gap Identification, Summary of literature.
3. Research Methodology — Research Approach (quantitative/qualitative/mixed), Research Design,
   Research Setting & Context, Data Sources, Data Collection Instruments, Analysis Procedure,
   Ethical Considerations, Limitations of Methodology.
4. Results — Organized findings (tables, figures with captions), factual reporting only
   (NO interpretation in this section).
5. Discussion — Interpretation of results, Comparison to prior literature, Theoretical
   implications, Practical implications. MUST NOT repeat raw results.
6. Conclusion — Direct answers to each research question, Summary of contributions. Must NOT
   introduce new material.
7. References — Properly formatted bibliography per the selected citation style.

CONDITIONAL SECTIONS (include when research warrants it):
8. Abstract — If the paper is standalone (always include for research papers): Problem summary,
   Method, Key Findings, Contribution. 150-300 words.
9. Keywords — 4-7 representative terms.
10. Theoretical / Conceptual Framework — If the study uses an established theory or conceptual
    model. Include: selected theory, its concepts, relevance, prior applications, how it guides
    the current study.
11. Case Studies — If the research investigates bounded real-world cases: Case selection rationale,
    Case descriptions, Systematic analysis, Connection to research questions.
12. Limitations — Explicit acknowledgment of study boundaries and weaknesses (may be a subsection
    of Discussion or a standalone section). MUST be included.
13. Recommendations — Practical proposals DIRECTLY linked to specific findings. Each recommendation
    must cite a finding or established limitation as its basis.
14. Future Research — Specific unresolved questions, NOT generic statements like "more research
    is needed". Must name what comparison, population, method, or outcome is missing.
15. Appendices — Raw data, instruments, supplementary material (if applicable).

PER-SECTION QUALITY STANDARDS:
- Introduction: The research gap must be evidenced from the literature review, not merely asserted.
  Research questions must be specific enough that the paper can answer them.
- Literature Review: Organize by THEMES (not by paper). Compare methodologies across studies.
  Identify and explain contradictions. Conclude with a specific, evidence-grounded research gap.
- Methodology: Must be specific enough for a reader to evaluate validity and replicate if applicable.
  Justify the chosen research approach and design.
- Results: Report findings only — no interpretation. Use tables and figures with proper captions.
- Discussion: Every interpretation must link back to a specific result AND a specific source from
  the literature review. Do NOT introduce results here.
- Conclusion: Must explicitly answer each research question stated in the Introduction.
- Recommendations: Each recommendation must be directly justified by a finding.
- Future Research: Identify specific gaps (not generic "more research needed").

You MUST output valid JSON — nothing else.

Output format schema:
{
  "title": "Proposed Research Title",
  "abstract_required": true,
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "sections": [
    {
      "name": "Section Name (e.g. Introduction, Literature Review, Methodology, Results, Discussion, Limitations, Conclusion, Recommendations, Future Research)",
      "subsections": ["Subsection 1", "Subsection 2"],
      "needs": "Description of required sources or data for this section",
      "guidance": "Specific writing guidance for this section",
      "word_count_target": 800
    }
  ],
  "estimated_word_count": 6000,
  "key_themes": ["theme1", "theme2"],
  "research_questions": ["RQ1: ...", "RQ2: ..."],
  "citation_style": "apa"
}

IMPORTANT RULES:
- Include all mandatory academic sections in the "sections" array: Introduction, Literature Review, Methodology, Results, Discussion, Limitations, Conclusion, References, and conditional sections as warranted (Abstract, Theoretical Framework, Case Studies, Recommendations, Future Research).
- Each section in "sections" must have name, subsections, needs, guidance, and word_count_target.
- Results and Discussion MUST be separate sections.
- Conclusion MUST address the research questions."""


def build_outline_prompt(
    topic: str,
    answers: dict,
    gap_report: dict | None = None,
) -> str:
    answers_text = "\n".join(
        f"- {k}: {v}" for k, v in answers.items()
    ) if answers else "No additional context from user."

    gap_section = ""
    if gap_report and gap_report.get("missing"):
        gaps = "\n".join(
            f"- {g.get('topic', 'Unknown')}: {g.get('reason', '')}"
            for g in gap_report["missing"]
        )
        gap_section = f"""

Known Gaps (from sufficiency analysis):
{gaps}
The outline should account for these gaps — ensure sections exist that would address them.
Each gap should be reflected in the Literature Review's research gap subsection and in the
Future Research section."""

    return f"""Create a structured research plan for the following topic.

Research Topic: "{topic}"

User Context:
{answers_text}
{gap_section}

Generate the research plan JSON now. Follow the full academic paper structure requirements
defined in the system prompt. Adapt the section list to fit the research type indicated by
the user context above. Output ONLY valid JSON — no commentary or explanation."""


def build_outline_component_prompt(
    topic: str,
    answers: dict,
    gap_report: dict | None = None,
    *,
    component: str,
    section_names: tuple[str, ...] = (),
) -> str:
    answers_text = "\n".join(
        f"- {key}: {value}" for key, value in (answers or {}).items()
    ) or "No additional context from user."
    gaps = (gap_report or {}).get("missing", [])
    gaps_text = "\n".join(
        f"- {gap.get('topic', 'Unknown')}: {gap.get('reason', '')}"
        for gap in gaps
        if isinstance(gap, dict)
    ) or "No known gaps supplied."
    context = f"""Research topic: {topic}
User context:
{answers_text}
Known research gaps:
{gaps_text}"""

    if component == "metadata":
        return f"""Component: outline metadata
{context}

Create only concise outline metadata for this research plan.
Return JSON with these keys:
{{
  "title": "Proposed Research Title",
  "abstract_required": true,
  "keywords": ["keyword"],
  "estimated_word_count": 6000,
  "key_themes": ["theme"],
  "research_questions": ["RQ1: ..."],
  "citation_style": "apa"
}}"""

    section_list = "\n".join(f"- {name}" for name in section_names)
    return f"""Component: section briefs
{context}

Create actionable briefs only for these sections:
{section_list}

Use the standard academic structure:
- Introduction: evidenced problem and gap, aim, objectives, answerable research questions, scope.
- Literature Review: organize by themes, compare methods/findings, identify an evidence-based gap.
- Research Methodology: specific and replicable approach, design, data, analysis, and ethics.
- Results: report findings without interpretation; organize by research question where applicable.
- Discussion: interpret results and compare with literature; do not repeat raw results.
- Limitations: state study boundaries and their effect on conclusions.
- Conclusion: answer the research questions and summarize contributions without new claims.
- Recommendations: link each recommendation to a finding or limitation.
- Future Research: identify specific unresolved questions.
- References: provide formatted bibliography guidance.

Return JSON only:
{{"sections":[{{"name":"...","subsections":["..."],"needs":"...","guidance":"...","word_count_target":500}}]}}
Include every requested section exactly once."""
