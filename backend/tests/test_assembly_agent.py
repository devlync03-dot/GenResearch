from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from models.schemas import GeneratedSection, ProposalOutput
from services.agents.assembly_agent import assemble_docx, human_input_checklist


def test_human_input_checklist_flags_methodology_decisions():
    output = ProposalOutput(
        topic="Study",
        sections=[GeneratedSection(section_name="methodology", text="The study uses interviews with participants.")],
    )
    assert any("interview" in item for item in human_input_checklist(output))


def test_docx_uses_professional_typography_margins_and_subheadings(tmp_path):
    output = ProposalOutput(
        topic="A Professional Research Proposal",
        sections=[
            GeneratedSection(
                section_name="Literature Review",
                text=(
                    "## Evidence quality\n\n"
                    "This plain paragraph contains **no bold emphasis** and should be "
                    "readable at the full body-text size.\n\n"
                    "### Comparative findings\n"
                    "Studies are compared using the available evidence."
                ),
            )
        ],
        references=["Smith, J. (2024). Evidence quality. *Research Journal*, *12*, 1-10."],
    )
    path = tmp_path / "professional.docx"
    assemble_docx(output, path)
    document = Document(path)

    assert document.sections[0].left_margin == Inches(1)
    assert document.sections[0].right_margin == Inches(1)
    assert document.styles["Normal"].font.name == "Times New Roman"
    assert document.styles["Normal"].font.size == Pt(12)
    assert document.styles["Heading 1"].font.size == Pt(16)
    assert document.styles["Heading 2"].font.size == Pt(14)

    subheading = next(p for p in document.paragraphs if p.text == "Evidence quality")
    body = next(p for p in document.paragraphs if p.text.startswith("This plain paragraph"))
    reference = next(p for p in document.paragraphs if p.text.startswith("Smith, J."))
    assert subheading.style.name == "Heading 2"
    assert body.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY
    assert all(not run.bold for run in body.runs)
    assert reference.style.name == "Normal"
    assert reference.paragraph_format.first_line_indent == Inches(-0.5)
    assert any(run.italic for run in reference.runs)