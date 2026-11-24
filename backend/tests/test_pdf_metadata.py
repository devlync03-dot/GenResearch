import fitz

from services.pdf_metadata import extract_pdf_metadata


def _build_pdf(metadata: dict[str, str] | None = None) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A Study of Research Metadata", fontsize=20)
    page.insert_text((72, 105), "Jane Doe and John Smith", fontsize=12)
    page.insert_text((72, 135), "2024", fontsize=11)
    page.insert_text((72, 165), "Abstract", fontsize=14)
    page.insert_textbox(
        fitz.Rect(72, 180, 520, 245),
        "This paper examines metadata extraction in research articles.",
        fontsize=11,
    )
    page.insert_text((72, 270), "Keywords: metadata, indexing", fontsize=10)
    page.insert_text((72, 300), "https://doi.org/10.1234/example.paper", fontsize=10)
    document.set_metadata(metadata or {})
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_pdf_metadata_uses_embedded_document_info_and_first_page_text():
    metadata = extract_pdf_metadata(_build_pdf({
        "title": "Metadata Title",
        "author": "Jane Doe; John Smith",
        "subject": "PDF indexing",
        "keywords": "document, indexing",
        "creator": "Test Publisher",
    }))

    assert metadata["title"] == "Metadata Title"
    assert metadata["authors"] == "Jane Doe; John Smith"
    assert metadata["year"] == 2024
    assert metadata["abstract"] == "This paper examines metadata extraction in research articles."
    assert metadata["doi"] == "10.1234/example.paper"
    assert metadata["keywords"] == ["document", "indexing"]
    assert metadata["subject"] == "PDF indexing"
    assert metadata["pdf_metadata"]["creator"] == "Test Publisher"


def test_extract_pdf_metadata_falls_back_to_first_page_bibliographic_text():
    metadata = extract_pdf_metadata(_build_pdf())

    assert metadata["title"] == "A Study of Research Metadata"
    assert metadata["authors"] == "Jane Doe and John Smith"
    assert metadata["year"] == 2024
    assert metadata["keywords"] == ["metadata", "indexing"]
