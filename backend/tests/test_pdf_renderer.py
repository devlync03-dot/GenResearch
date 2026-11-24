from services.pdf_renderer import APAPDFRenderer, render_markdown_to_pdf


def test_heading_has_one_body_line_of_space_above_and_below():
    pdf = APAPDFRenderer(title="Spacing test", double_spaced=False)
    start_y = pdf.get_y()

    pdf.write_heading("Introduction", level=2)

    assert pdf.get_y() - start_y >= 3 * pdf.line_h - 1e-6


def test_pdf_renderer_handles_markdown_lists_and_citations():
    registry = [
        {
            "id": "CR-001",
            "title": "Sample Paper",
            "authors": "Doe, J.",
            "year": 2024,
            "url": "https://example.org/sample",
        }
    ]
    pdf = render_markdown_to_pdf(
        "## Summary\n\n- First item\n- Second item\n\n"
        "Prior work—published in 2024—supports this claim [CR-001]. "
        "‘Quoted text’… Greek: α.",
        title="Unicode – smoke test",
        citation_registry=registry,
    )

    assert pdf.startswith(b"%PDF-")


def test_pdf_renderer_separates_inline_and_consecutive_headings(monkeypatch):
    headings = []
    original_write_heading = APAPDFRenderer.write_heading

    def record_heading(self, text, level):
        headings.append(text)
        original_write_heading(self, text, level)

    monkeypatch.setattr(APAPDFRenderer, "write_heading", record_heading)

    pdf = render_markdown_to_pdf(
        "Text before. ## Evidence-based gap\n"
        "## Aim and objectives\n"
        "Text after."
    )

    assert pdf.startswith(b"%PDF-")
    assert headings == ["Evidence-based gap", "Aim and objectives"]
