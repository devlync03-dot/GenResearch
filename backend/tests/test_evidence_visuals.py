import asyncio

from services.agents.nodes.evidence_visuals import evidence_visuals_node
from services.report_store import build_completion_guide_text


def test_evidence_visuals_reports_numeric_claims_and_refuses_unverified_artifacts():
    result = asyncio.run(
        evidence_visuals_node(
            {
                "generated_sections": [
                    {
                        "section_name": "Results",
                        "text": "Employment increased by 12% in 2024 [CR-001].",
                    },
                    {
                        "section_name": "Results and Discussion",
                        "text": "The estimate rose from 2.5 to 5.0.",
                    },
                ],
                "citation_registry": [
                    {
                        "id": "CR-001",
                        "title": "Employment study",
                        "doi": "10.1234/example",
                        "evidence_level": "full_text",
                    }
                ],
            }
        )
    )

    report = result["evidence_visual_report"]
    assert [gap["section"] for gap in report["gap_report"]] == [
        "Results",
        "Results and Discussion",
    ]
    assert [gap["priority"] for gap in report["gap_report"]] == [1, 1]
    assert len(report["unresolved"]) == 2
    assert report["sources_added"] == []
    assert report["datasets"] == []
    assert report["equations"] == []
    assert report["figures"] == []
    assert report["tables"] == []
    assert report["patches"] == []
    assert "page/table locations" in report["warnings"][0]


def test_evidence_visuals_flags_equation_without_claiming_validation():
    result = asyncio.run(
        evidence_visuals_node(
            {
                "generated_sections": [
                    {"section_name": "Methodology", "text": "$y = 2x$"},
                ],
                "citation_registry": [],
            }
        )
    )

    report = result["evidence_visual_report"]
    assert report["gap_report"][0]["claim"] == "$y = 2x$"
    assert "published or illustrative" in report["gap_report"][0]["evidence_type_needed"]
    assert report["equations"] == []


def test_evidence_visuals_does_not_flag_citation_ids_as_numeric_claims():
    result = asyncio.run(
        evidence_visuals_node(
            {
                "generated_sections": [
                    {"section_name": "Introduction", "text": "Prior work [CR-001] is relevant."},
                ],
                "citation_registry": [{"id": "CR-001", "title": "Paper"}],
            }
        )
    )

    report = result["evidence_visual_report"]
    assert report["gap_report"] == []
    assert report["unresolved"] == []


def test_completion_guide_includes_evidence_visual_gap_and_warning():
    guide = build_completion_guide_text(
        {
            "topic": "Test",
            "evidence_visual_report": {
                "gap_report": [{
                    "section": "Results",
                    "claim": "Employment increased by 12%.",
                    "priority": 1,
                }],
                "warnings": ["No verified chart was generated."],
            },
        }
    )

    assert "## 4. Evidence & Visuals Audit" in guide
    assert "Results" in guide
    assert "Employment increased by 12%." in guide
    assert "No verified chart was generated." in guide
