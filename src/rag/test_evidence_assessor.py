
"""Unit tests for the evidence sufficiency assessor."""

import json
from unittest.mock import Mock

import pytest

from src.rag.context_builder import ContextPackage, ContextSource
from src.rag.evidence_assessor import EvidenceSufficiencyAssessor


@pytest.fixture
def context_package() -> ContextPackage:
    """Create a sample context package containing retrieved document evidence."""

    source = ContextSource(
        citation_id="S1",
        chunk_id="chunk-001",
        document_id="doc-001",
        source="employee_policy.pdf",
        page_numbers=[2],
        heading_path=["Leave Policy"],
        text="Employees receive 20 days of annual leave.",
        retrieval_score=0.85,
    )

    return ContextPackage(
        context_text="[S1] Employees receive 20 days of annual leave.",
        sources=[source],
        scope_type="personal",
        scope_id="user-001",
        included_chunks=1,
        omitted_chunk_ids=[],
    )


def create_assessor(response: str = "") -> tuple[EvidenceSufficiencyAssessor, Mock]:
    """Create an assessor with a mocked LLM generator."""

    generator = Mock()
    generator.generate.return_value = response

    assessor = EvidenceSufficiencyAssessor(
        llm_generator=generator,
        confidence_threshold=0.70,
    )

    return assessor, generator


def test_sufficient_evidence_is_accepted(context_package: ContextPackage) -> None:
    """Accept evidence when the model reports sufficient coverage and confidence."""

    response = json.dumps({
        "sufficient": True,
        "confidence": 0.90,
        "reason": "The document directly answers the question.",
        "covered_aspects": ["Annual leave entitlement"],
        "missing_aspects": [],
    })

    assessor, generator = create_assessor(response)

    result = assessor.assess(
        "How many annual leave days do employees receive?",
        context_package,
    )

    assert result.sufficient is True
    assert result.confidence == 0.90
    assert result.covered_aspects == ["Annual leave entitlement"]
    generator.generate.assert_called_once()


def test_insufficient_evidence_is_rejected(context_package: ContextPackage) -> None:
    """Reject evidence when the model identifies missing information."""

    response = json.dumps({
        "sufficient": False,
        "confidence": 0.95,
        "reason": "The context does not mention carry-forward rules.",
        "covered_aspects": ["Annual leave entitlement"],
        "missing_aspects": ["Leave carry-forward policy"],
    })

    assessor, _ = create_assessor(response)

    result = assessor.assess(
        "How many leave days are available and can they be carried forward?",
        context_package,
    )

    assert result.sufficient is False
    assert result.missing_aspects == ["Leave carry-forward policy"]


def test_low_confidence_is_rejected(context_package: ContextPackage) -> None:
    """Reject a positive assessment when confidence is below the threshold."""

    response = json.dumps({
        "sufficient": True,
        "confidence": 0.50,
        "reason": "The evidence may answer the question.",
        "covered_aspects": ["Annual leave entitlement"],
        "missing_aspects": [],
    })

    assessor, _ = create_assessor(response)

    result = assessor.assess("What is the leave entitlement?", context_package)

    assert result.sufficient is False


def test_no_covered_aspects_is_rejected(context_package: ContextPackage) -> None:
    """Reject a positive assessment without any identified supporting aspects."""

    response = json.dumps({
        "sufficient": True,
        "confidence": 0.95,
        "reason": "The evidence appears relevant.",
        "covered_aspects": [],
        "missing_aspects": [],
    })

    assessor, _ = create_assessor(response)

    result = assessor.assess("What is the leave entitlement?", context_package)

    assert result.sufficient is False


def test_empty_context_does_not_call_llm() -> None:
    """Return insufficient evidence immediately when no context is available."""

    empty_context = ContextPackage(
        context_text="",
        sources=[],
        scope_type="personal",
        scope_id="user-001",
        included_chunks=0,
        omitted_chunk_ids=[],
    )

    assessor, generator = create_assessor()

    result = assessor.assess("What is the leave entitlement?", empty_context)

    assert result.sufficient is False
    assert result.confidence == 0.0
    generator.generate.assert_not_called()


def test_empty_question_is_rejected(context_package: ContextPackage) -> None:
    """Reject an empty question before calling the LLM."""

    assessor, generator = create_assessor()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        assessor.assess("   ", context_package)

    generator.generate.assert_not_called()


def test_malformed_json_fails_closed(context_package: ContextPackage) -> None:
    """Treat malformed model output as insufficient evidence."""

    assessor, _ = create_assessor("This is not valid JSON.")

    result = assessor.assess("What is the leave entitlement?", context_package)

    assert result.sufficient is False
    assert result.confidence == 0.0


def test_missing_json_field_fails_closed(context_package: ContextPackage) -> None:
    """Reject model output that does not contain every required field."""

    response = json.dumps({
        "sufficient": True,
        "confidence": 0.95,
        "reason": "The document answers the question.",
        "covered_aspects": ["Annual leave entitlement"],
    })

    assessor, _ = create_assessor(response)

    result = assessor.assess("What is the leave entitlement?", context_package)

    assert result.sufficient is False


def test_invalid_confidence_fails_closed(context_package: ContextPackage) -> None:
    """Reject confidence values outside the valid range."""

    response = json.dumps({
        "sufficient": True,
        "confidence": 1.5,
        "reason": "The document answers the question.",
        "covered_aspects": ["Annual leave entitlement"],
        "missing_aspects": [],
    })

    assessor, _ = create_assessor(response)

    result = assessor.assess("What is the leave entitlement?", context_package)

    assert result.sufficient is False


def test_invalid_boolean_fails_closed(context_package: ContextPackage) -> None:
    """Reject string values instead of a JSON boolean for sufficiency."""

    response = json.dumps({
        "sufficient": "true",
        "confidence": 0.95,
        "reason": "The document answers the question.",
        "covered_aspects": ["Annual leave entitlement"],
        "missing_aspects": [],
    })

    assessor, _ = create_assessor(response)

    result = assessor.assess("What is the leave entitlement?", context_package)

    assert result.sufficient is False


def test_markdown_json_fence_is_supported(context_package: ContextPackage) -> None:
    """Accept valid JSON even when the model wraps it in Markdown code fences."""

    response = """```json
{
    "sufficient": true,
    "confidence": 0.90,
    "reason": "The document directly answers the question.",
    "covered_aspects": ["Annual leave entitlement"],
    "missing_aspects": []
}
```"""

    assessor, _ = create_assessor(response)

    result = assessor.assess("What is the leave entitlement?", context_package)

    assert result.sufficient is True


def test_invalid_threshold_is_rejected() -> None:
    """Reject confidence thresholds outside the range from zero to one."""

    with pytest.raises(ValueError, match="confidence_threshold"):
        EvidenceSufficiencyAssessor(
            llm_generator=Mock(),
            confidence_threshold=1.5,
        )


def test_unexpected_json_field_fails_closed(context_package: ContextPackage) -> None:
    """Reject JSON containing unexpected fields."""

    response = json.dumps({
        "sufficient": True,
        "confidence": 0.90,
        "reason": "The document answers the question.",
        "covered_aspects": ["Annual leave entitlement"],
        "missing_aspects": [],
        "extra_field": "unexpected",
    })

    assessor, _ = create_assessor(response)

    result = assessor.assess("What is the leave entitlement?", context_package)

    assert result.sufficient is False
