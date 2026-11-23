
"""Tests for citation extraction, validation, and management."""

import pytest

from src.rag.citation_manager import CitationManager
from src.rag.models import SourceReference


@pytest.fixture
def manager() -> CitationManager:
    """Provide a CitationManager instance."""

    return CitationManager()


@pytest.fixture
def sources() -> list[SourceReference]:
    """Provide representative document source references."""

    return [
        SourceReference(
            citation_id="S1",
            chunk_id="chunk-001",
            document_id="doc-001",
            source="HR_Policy.pdf",
            page_numbers=[2],
            heading_path=["HR", "Leave Policy"],
        ),
        SourceReference(
            citation_id="S2",
            chunk_id="chunk-002",
            document_id="doc-002",
            source="Employee_Handbook.pdf",
            page_numbers=[4],
            heading_path=["Employee Rules"],
        ),
    ]


def test_extract_single_citation(manager: CitationManager) -> None:
    """Extract one citation from generated text."""

    result = manager.extract_citations(
        "Employees receive 20 days of leave. [S1]"
    )

    assert result == ["S1"]


def test_extract_multiple_citations(manager: CitationManager) -> None:
    """Extract multiple citation identifiers."""

    result = manager.extract_citations(
        "Leave entitlement is 20 days [S1]. Approval requires a manager [S2]."
    )

    assert result == ["S1", "S2"]


def test_duplicate_citations_are_removed(
    manager: CitationManager,
) -> None:
    """Return each citation identifier only once."""

    result = manager.extract_citations(
        "The policy states 20 days [S1]. This is confirmed again [S1]."
    )

    assert result == ["S1"]


def test_no_citations_found(manager: CitationManager) -> None:
    """Return an empty list when no citations are present."""

    result = manager.extract_citations(
        "The available documents do not contain sufficient information."
    )

    assert result == []


def test_valid_citations(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Accept citations that reference supplied sources."""

    result = manager.validate(
        "Employees receive 20 days of annual leave [S1].",
        sources,
    )

    assert result.valid is True
    assert result.found_citations == ["S1"]
    assert result.unknown_citations == []
    assert result.unused_citations == ["S2"]
    assert result.missing_citations is False


def test_multiple_valid_citations(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Accept multiple valid citations."""

    result = manager.validate(
        "Leave is 20 days [S1]. Manager approval is required [S2].",
        sources,
    )

    assert result.valid is True
    assert result.found_citations == ["S1", "S2"]
    assert result.unknown_citations == []
    assert result.unused_citations == []


def test_unknown_citation_is_detected(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Identify citation identifiers that were not supplied."""

    result = manager.validate(
        "Employees receive 30 days of leave [S99].",
        sources,
    )

    assert result.valid is False
    assert result.unknown_citations == ["S99"]


def test_missing_citations_are_detected(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Identify answers without citation markers."""

    result = manager.validate(
        "Employees receive 20 days of annual leave.",
        sources,
    )

    assert result.valid is True
    assert result.missing_citations is True
    assert result.found_citations == []


def test_duplicate_source_ids_are_rejected(
    manager: CitationManager,
) -> None:
    """Reject ambiguous source lists containing duplicate citation IDs."""

    duplicate_sources = [
        SourceReference(
            citation_id="S1",
            chunk_id="chunk-001",
            document_id="doc-001",
            source="HR_Policy.pdf",
        ),
        SourceReference(
            citation_id="S1",
            chunk_id="chunk-002",
            document_id="doc-002",
            source="Employee_Handbook.pdf",
        ),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        manager.validate("Annual leave is 20 days [S1].", duplicate_sources)


def test_filter_valid_sources(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Return only sources actually cited by the answer."""

    result = manager.filter_valid_sources(
        "Employees receive 20 days of annual leave [S1].",
        sources,
    )

    assert len(result) == 1
    assert result[0].citation_id == "S1"


def test_filter_sources_rejects_unknown_citations(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Reject source filtering when unknown citations exist."""

    with pytest.raises(ValueError, match="unknown citation"):
        manager.filter_valid_sources(
            "Employees receive 20 days of leave [S99].",
            sources,
        )


def test_remove_invalid_citations(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Remove citation markers that do not reference supplied sources."""

    answer = "Annual leave is 20 days [S1]. Additional leave is 10 days [S99]."

    result = manager.remove_invalid_citations(answer, sources)

    assert "[S1]" in result
    assert "[S99]" not in result


def test_remove_invalid_citations_preserves_valid_answer(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Preserve answers when every citation is valid."""

    answer = "Annual leave is 20 days [S1]."

    result = manager.remove_invalid_citations(answer, sources)

    assert result == answer


def test_invalid_answer_type_is_rejected(
    manager: CitationManager,
    sources: list[SourceReference],
) -> None:
    """Reject non-string answer values."""

    with pytest.raises(TypeError):
        manager.validate(123, sources)


def test_invalid_sources_type_is_rejected(
    manager: CitationManager,
) -> None:
    """Reject a source collection that is not a list."""

    with pytest.raises(TypeError):
        manager.validate("Answer [S1].", "invalid")


def test_invalid_source_item_is_rejected(
    manager: CitationManager,
) -> None:
    """Reject source entries that are not SourceReference objects."""

    with pytest.raises(TypeError):
        manager.validate("Answer [S1].", ["invalid"])
