
"""Tests for the scope-safe RAG context builder."""

import pytest

from src.rag.context_builder import ContextBuilder


@pytest.fixture
def builder() -> ContextBuilder:
    """Provide a ContextBuilder instance for testing."""

    return ContextBuilder(max_context_chars=12000)


@pytest.fixture
def retrieval_results() -> list[dict]:
    """Provide representative results from the hybrid retriever."""

    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "text": "The leave policy provides 20 days of annual leave.",
            "source": "HR_Policy.pdf",
            "scope_type": "personal",
            "scope_id": "user-101",
            "rrf_score": 0.045,
            "metadata": {
                "document_id": "doc-001",
                "source": "HR_Policy.pdf",
                "page_numbers": [2],
                "heading_path": ["Human Resources", "Leave Policy"],
            },
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-002",
            "text": "Employees must submit leave requests to their manager.",
            "source": "Employee_Handbook.pdf",
            "scope_type": "personal",
            "scope_id": "user-101",
            "rrf_score": 0.038,
            "metadata": {
                "document_id": "doc-002",
                "source": "Employee_Handbook.pdf",
                "page_numbers": "[4, 5]",
                "heading_path": '["Employee Rules", "Leave Requests"]',
            },
        },
    ]


def test_build_context_success(
    builder: ContextBuilder,
    retrieval_results: list[dict],
) -> None:
    """Build context successfully from valid retrieval results."""

    package = builder.build(
        retrieval_results,
        scope_type="personal",
        scope_id="user-101",
    )

    assert package.included_chunks == 2
    assert package.scope_type == "personal"
    assert package.scope_id == "user-101"
    assert "[S1]" in package.context_text
    assert "[S2]" in package.context_text
    assert "HR_Policy.pdf" in package.context_text
    assert "Employee_Handbook.pdf" in package.context_text


def test_citation_source_mapping(
    builder: ContextBuilder,
    retrieval_results: list[dict],
) -> None:
    """Preserve citation IDs and their corresponding evidence."""

    package = builder.build(
        retrieval_results,
        scope_type="personal",
        scope_id="user-101",
    )

    first_source = package.sources[0]
    second_source = package.sources[1]

    assert first_source.citation_id == "S1"
    assert first_source.chunk_id == "chunk-001"
    assert first_source.document_id == "doc-001"

    assert second_source.citation_id == "S2"
    assert second_source.chunk_id == "chunk-002"
    assert second_source.document_id == "doc-002"


def test_page_and_heading_metadata_parsing(
    builder: ContextBuilder,
    retrieval_results: list[dict],
) -> None:
    """Parse both native lists and JSON-serialized metadata lists."""

    package = builder.build(
        retrieval_results,
        scope_type="personal",
        scope_id="user-101",
    )

    assert package.sources[0].page_numbers == [2]
    assert package.sources[0].heading_path == [
        "Human Resources",
        "Leave Policy",
    ]

    assert package.sources[1].page_numbers == [4, 5]
    assert package.sources[1].heading_path == [
        "Employee Rules",
        "Leave Requests",
    ]


def test_rejects_different_scope_type(
    builder: ContextBuilder,
    retrieval_results: list[dict],
) -> None:
    """Reject retrieval results from another scope type."""

    retrieval_results[0]["scope_type"] = "project"

    with pytest.raises(PermissionError):
        builder.build(
            retrieval_results,
            scope_type="personal",
            scope_id="user-101",
        )


def test_rejects_different_scope_id(
    builder: ContextBuilder,
    retrieval_results: list[dict],
) -> None:
    """Reject retrieval results belonging to another user."""

    retrieval_results[0]["scope_id"] = "user-202"

    with pytest.raises(PermissionError):
        builder.build(
            retrieval_results,
            scope_type="personal",
            scope_id="user-101",
        )


def test_rejects_missing_scope_metadata(
    builder: ContextBuilder,
    retrieval_results: list[dict],
) -> None:
    """Reject results that do not contain complete scope metadata."""

    del retrieval_results[0]["scope_id"]

    with pytest.raises(PermissionError):
        builder.build(
            retrieval_results,
            scope_type="personal",
            scope_id="user-101",
        )


def test_duplicate_chunks_are_removed(
    builder: ContextBuilder,
    retrieval_results: list[dict],
) -> None:
    """Include a retrieved chunk only once."""

    duplicate = retrieval_results[0].copy()
    retrieval_results.append(duplicate)

    package = builder.build(
        retrieval_results,
        scope_type="personal",
        scope_id="user-101",
    )

    assert package.included_chunks == 2
    assert len(package.sources) == 2


def test_empty_retrieval_results(builder: ContextBuilder) -> None:
    """Return an empty context when no evidence is retrieved."""

    package = builder.build(
        [],
        scope_type="personal",
        scope_id="user-101",
    )

    assert package.context_text == ""
    assert package.sources == []
    assert package.included_chunks == 0
    assert package.omitted_chunk_ids == []


def test_context_character_limit() -> None:
    """Exclude chunks that exceed the configured context budget."""

    small_builder = ContextBuilder(max_context_chars=100)

    results = [
        {
            "chunk_id": "chunk-large",
            "document_id": "doc-001",
            "text": "A" * 500,
            "scope_type": "personal",
            "scope_id": "user-101",
            "metadata": {},
        }
    ]

    package = small_builder.build(
        results,
        scope_type="personal",
        scope_id="user-101",
    )

    assert package.included_chunks == 0
    assert package.omitted_chunk_ids == ["chunk-large"]
    assert package.context_text == ""


def test_invalid_scope_type(builder: ContextBuilder) -> None:
    """Reject unsupported knowledge scope types."""

    with pytest.raises(ValueError):
        builder.build(
            [],
            scope_type="global",
            scope_id="user-101",
        )


def test_invalid_scope_id(builder: ContextBuilder) -> None:
    """Reject an empty scope identifier."""

    with pytest.raises(ValueError):
        builder.build(
            [],
            scope_type="personal",
            scope_id="",
        )


def test_invalid_retrieval_result_type(builder: ContextBuilder) -> None:
    """Reject retrieval results that are not supplied as a list."""

    with pytest.raises(TypeError):
        builder.build(
            "invalid",
            scope_type="personal",
            scope_id="user-101",
        )


def test_missing_chunk_text(
    builder: ContextBuilder,
    retrieval_results: list[dict],
) -> None:
    """Reject chunks without usable evidence text."""

    retrieval_results[0]["text"] = ""

    with pytest.raises(ValueError):
        builder.build(
            retrieval_results,
            scope_type="personal",
            scope_id="user-101",
        )


def test_project_scope_context(builder: ContextBuilder) -> None:
    """Build context successfully for an authorized project scope."""

    results = [
        {
            "chunk_id": "project-chunk-001",
            "document_id": "project-doc-001",
            "text": "The project deadline is December 15.",
            "scope_type": "project",
            "scope_id": "project-001",
            "metadata": {},
        }
    ]

    package = builder.build(
        results,
        scope_type="project",
        scope_id="project-001",
    )

    assert package.scope_type == "project"
    assert package.scope_id == "project-001"
    assert package.included_chunks == 1
    assert "December 15" in package.context_text
