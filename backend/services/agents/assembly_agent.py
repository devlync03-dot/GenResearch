"""Assemble verified proposal sections into a DOCX and human checklist."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from models.schemas import ProposalOutput
from services.docx_export import _clean_markdown


HUMAN_INPUT_TRIGGERS = (
    "primary data",
    "interview",
    "survey",
    "irb",
    "ethics approval",
    "human participants",
)


def human_input_checklist(output: ProposalOutput) -> list[str]:
    """Identify methodology decisions that cannot be safely invented."""
    checklist = list(output.human_input_needed)
    methodology = next(
        (
            section.text.lower()
            for section in output.sections
            if section.section_name.strip().lower() == "methodology"
        ),
        "",
    )
    for trigger in HUMAN_INPUT_TRIGGERS:
        if trigger in methodology:
            item = f"Confirm the methodology detail involving {trigger}."
            if item not in checklist:
                checklist.append(item)
    if any(result.verdict != "supported" for section in output.sections for result in section.verification):
        checklist.append("Review claims marked partial or unsupported by evidence verification.")
    return checklist


def _set_font(style, size: int, bold: bool = False) -> None:
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def _configure_document(document: Document) -> None:
    for section in document.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    normal = document.styles["Normal"]
    _set_font(normal, 12)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Inches(0.3)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)

    title = document.styles["Title"]
    _set_font(title, 16, bold=True)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(18)

    for level in range(1, 5):
        style = document.styles[f"Heading {level}"]
        _set_font(style, 16 if level == 1 else 14, bold=True)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        style.paragraph_format.first_line_indent = Inches(0)
        style.paragraph_format.line_spacing = 1.0
        style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True

    for style_name in ("List Bullet", "List Number"):
        style = document.styles[style_name]
        _set_font(style, 12)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        style.paragraph_format.first_line_indent = Inches(0)
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.space_after = Pt(3)


def _add_section_body(document: Document, text: str) -> None:
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            body = _clean_markdown(" ".join(line.strip() for line in paragraph_lines))
            if body:
                paragraph = document.add_paragraph(body)
                paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            paragraph_lines.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = min(len(heading.group(1)), 4)
            document.add_heading(_clean_markdown(heading.group(2)), level=level)
        else:
            paragraph_lines.append(stripped)
    flush_paragraph()


def _add_reference(document: Document, text: str):
    paragraph = document.add_paragraph()
    for part in re.split(r"(\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("*") and part.endswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.left_indent = Inches(0.5)
    paragraph.paragraph_format.first_line_indent = Inches(-0.5)
    paragraph.paragraph_format.keep_together = True
    return paragraph


def assemble_docx(output: ProposalOutput, path: str | Path) -> str:
    """Write proposal sections, figures, references, and checklist to DOCX."""
    document = Document()
    _configure_document(document)
    document.add_paragraph(output.topic, style="Title")
    for section in output.sections:
        document.add_heading(section.section_name.replace("_", " ").title(), level=1)
        _add_section_body(document, section.text.strip())
    for figure in output.figures:
        if Path(figure).exists():
            document.add_picture(figure, width=Inches(5.5))
    document.add_heading("References", level=1)
    for reference in output.references:
        _add_reference(document, reference)
    document.add_heading("Items Requiring Human Input", level=1)
    for item in human_input_checklist(output) or ["No additional manual input was detected."]:
        document.add_paragraph(item, style="List Bullet")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    return str(destination)