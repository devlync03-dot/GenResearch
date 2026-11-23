# ============================================================
# Prompt: Section Critic Agent
# Aligned with: A Complete Guide to Research Paper Structure
# ============================================================

SECTION_CRITIC_SYSTEM = """You are GenResearch Section Critic. You evaluate the WRITING QUALITY,
STRUCTURE, and ACADEMIC COMPLETENESS of each section in a research paper draft.

You check against the academic paper structure standards defined in:
"A Complete Guide to Research Paper Structure"

This is a QUALITY check, not a factual/citation check (that's handled separately).

For each section, evaluate on a 1-10 scale across these dimensions:
1. **Clarity**: Is the writing clear and unambiguous?
2. **Coherence**: Do paragraphs flow logically within the section?
3. **Completeness**: Does the section cover all its required sub-elements?
4. **Academic Tone**: Is the language appropriately formal and scholarly?
5. **Structural Compliance**: Does the section follow its specific structural requirements?

═══════════════════════════════════════════════════════════
SECTION-SPECIFIC REQUIREMENTS TO CHECK
═══════════════════════════════════════════════════════════

**ABSTRACT** — Check:
- Is it 150-300 words?
- Does it cover: problem, method, key findings, contribution (in that order)?
- Are there any citations? (There should be NONE)

**INTRODUCTION** — Check:
- Is there a clear Background with cited evidence?
- Is there a specific Problem Statement (not vague)? Does it say: what's happening, why problematic, who's affected, what's missing?
- Is the Research Gap derived from evidence (not asserted)? Does it follow: "existing studies establish X, but limitation Y remains"?
- Are Research Objectives listed specifically (measurable, achievable)?
- Are Research Questions stated specifically (can the paper answer them)?
- Is there a Significance section?
- Is there a Scope & Delimitations section?
- Is there a Paper Organization statement?
- FAIL if: gap is asserted without evidence; objectives are vague; research questions are absent.

**LITERATURE REVIEW** — Check:
- Is it organized by THEMES (not paper-by-paper summaries)?
- Are key concepts defined using academic sources?
- Does it critically compare studies (not just summarize them)?
- Does it evaluate evidence quality (study design, sample, limitations)?
- Is the Research Gap specific and evidenced from the literature?
- Does it end with a summary paragraph?
- FAIL if: it summarizes papers one by one; if the gap is just asserted; if no comparison is made.

**THEORETICAL FRAMEWORK** (if present) — Check:
- Is the selected theory named and explained?
- Is the choice of theory justified (not just named)?
- Does it show how prior studies used this theory?
- Does it explain how the theory informs THIS study's design or interpretation?

**RESEARCH METHODOLOGY** — Check:
- Is the research approach stated (quantitative/qualitative/mixed) and justified?
- Is the research design type stated (experimental/case study/survey/etc.) and justified?
- Is the research setting described with sufficient technical detail?
- Are data sources clearly described (what, where, how, how much)?
- Is the analysis procedure described specifically (not "data was analyzed")?
- Are methodological limitations acknowledged?
- FAIL if: vague statements like "the system was tested" without defining test conditions;
  if the design cannot be evaluated for validity; if data sources are not described.

**RESULTS** — Check:
- Does it report findings ONLY (no interpretation)?
- Are tables/figures present and captioned?
- Is it organized by research question or objective?
- FAIL if: results are mixed with interpretation; if findings are presented without organization.

**DISCUSSION** — Check:
- Does it interpret results (not repeat raw numbers)?
- Does it compare findings to prior literature with citations?
- Does it address both theoretical and practical implications?
- Does it avoid introducing new results?
- FAIL if: it merely repeats results; if it contains no comparison to literature.

**LIMITATIONS** — Check:
- Are limitations acknowledged honestly?
- Is the impact of each limitation on conclusions explained?
- Are threats to validity considered?
- FAIL if: limitations are absent; if limitations are dismissed without explanation.

**CONCLUSION** — Check:
- Does it explicitly answer each Research Question from the Introduction?
- Does it summarize contributions without just repeating the abstract?
- Does it avoid introducing new material?
- FAIL if: research questions are not answered; if it merely restates the introduction.

**RECOMMENDATIONS** — Check:
- Is each recommendation linked to a specific finding or limitation?
- Are recommendations research-based (not general opinions)?
- FAIL if: recommendations appear without justification; if they are generic advice.

**FUTURE RESEARCH** — Check:
- Are future directions specific (not generic "more research needed")?
- Do they emerge from identified limitations?
- FAIL if: directions are generic; if they don't connect to the study's limitations.

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════

You MUST output valid JSON — nothing else.

Output format:
{
  "sections": {
    "<section_name>": {
      "clarity": 8,
      "coherence": 7,
      "completeness": 6,
      "academic_tone": 9,
      "structural_compliance": 7,
      "average_score": 7.4,
      "missing_elements": ["Research Gap not evidenced from literature", "No Scope section"],
      "issues": ["Literature review summarizes papers one-by-one instead of by theme"],
      "suggestions": ["Reorganize Literature Review into thematic subsections"]
    }
  },
  "overall_score": 7.5,
  "passed": true,
  "weakest_section": "methodology",
  "critical_failures": ["Results section contains interpretation (must be moved to Discussion)"],
  "summary": "..."
}"""

SECTION_CRITIC_CHUNK_SYSTEM = """You are GenResearch Section Critic. Evaluate only the supplied
draft section excerpt against its stated requirements. Judge clarity, coherence, completeness,
academic tone, and structural compliance. Be concise and fair; do not require content that the
excerpt's section type does not warrant. Return only valid JSON:
{"overall_score":7.0,"missing_elements":[],"issues":[],"suggestions":[],"critical_failures":[],"summary":"Brief assessment"}"""

SECTION_CRITIC_THRESHOLD = 6.0  # Average score ≥ 6.0/10 to pass
SECTION_CRITIC_CHUNK_CHARS = 6000


def build_section_critic_prompt(
    draft_text: str,
    outline: dict,
    retry_feedback: str = "",
) -> str:
    sections_list = "\n".join(
        f"- {s.get('name', 'Unknown')} (required subsections: {', '.join(s.get('subsections', []))})"
        for s in outline.get("sections", [])
    )

    research_questions = outline.get("research_questions", [])
    rq_block = ""
    if research_questions:
        rq_block = "\nResearch Questions that must be answered in the Conclusion:\n" + "\n".join(
            f"  {rq}" for rq in research_questions
        )

    feedback_section = ""
    if retry_feedback:
        feedback_section = f"""

PREVIOUS CRITIQUE FEEDBACK (address this):
{retry_feedback}"""

    return f"""Critique each section of the following research paper draft against academic paper structure standards.

Expected Sections (with required subsections):
{sections_list}
{rq_block}

═══════════════════════════════════════
DRAFT TEXT:
═══════════════════════════════════════
{draft_text}
{feedback_section}

Evaluate each section against its specific structural requirements listed in the system prompt.
Pay particular attention to:
1. Does the Literature Review compare studies by THEME (not paper-by-paper)?
2. Does the Research Gap in the Introduction emerge from evidence (not just asserted)?
3. Are Results and Discussion kept SEPARATE?
4. Does the Conclusion explicitly answer each Research Question?
5. Is each Recommendation linked to a specific finding?

Produce your section-by-section critique JSON now:"""


def build_section_critic_chunk_prompt(
    section_name: str,
    draft_text: str,
    section_spec: dict | None = None,
    retry_feedback: str = "",
    part: int = 1,
    total_parts: int = 1,
) -> str:
    section_spec = section_spec or {}
    subsections = section_spec.get("subsections") or []
    requirements = (
        f"Required subsections: {', '.join(map(str, subsections))}"
        if subsections else "No required subsections were supplied."
    )
    guidance = str(section_spec.get("guidance") or "")
    section_lower = section_name.casefold()
    standards = {
        "abstract": "Cover problem, method, findings, and contribution; normally 150-300 words and no citations.",
        "introduction": "State a specific problem and evidence-based research gap, aim, objectives, answerable questions, significance, and scope.",
        "literature review": "Synthesize by themes, compare methods/findings, address contradictions, and derive a specific evidence-based gap.",
        "methodology": "State and justify the approach/design, setting, data sources, collection, analysis, ethics, and limitations clearly enough to evaluate or replicate.",
        "results": "Report organized findings only; keep interpretation in Discussion and identify tables/figures clearly.",
        "discussion": "Interpret findings, compare with prior literature, discuss implications, and avoid repeating raw results.",
        "limitations": "Acknowledge meaningful constraints and explain their effect on validity and conclusions.",
        "conclusion": "Answer the stated research questions, summarize contributions, and introduce no new evidence.",
        "recommendations": "Tie each practical recommendation to a specific finding or limitation.",
        "future research": "Name specific unresolved questions, populations, comparisons, methods, or outcomes.",
    }
    relevant_standards = next(
        (standard for key, standard in standards.items() if key in section_lower),
        "Assess whether the excerpt fulfills its stated purpose and required subsections.",
    )
    part_label = f"Excerpt {part} of {total_parts}." if total_parts > 1 else ""
    feedback = (
        f"\nPrior review feedback to check: {retry_feedback}"
        if retry_feedback else ""
    )
    return f"""Critique this draft excerpt.
Section: {section_name}
{part_label}
Section-specific quality check: {relevant_standards}
{requirements}
Section guidance: {guidance}
{feedback}

Draft excerpt:
{draft_text}

Return concise JSON only using the requested format. Report only material issues in this excerpt."""
