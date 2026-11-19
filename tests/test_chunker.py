
"""Tests for structure-aware document chunking."""

import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.loaders import DocumentElement


def test_chunk_document_preserves_document_metadata():
    """Check that chunks retain their document and source metadata."""

    elements = [
        DocumentElement(
            text="Employees receive annual leave.",
            element_type="paragraph",
            source="leave_policy.txt",
            page_number=1,
        )
    ]

    chunks = chunk_document(elements, document_id="doc_001")

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc_001"
    assert chunks[0].source == "leave_policy.txt"
    assert chunks[0].page_numbers == (1,)
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "doc_001_chunk_0000"


def test_chunk_document_preserves_heading_hierarchy():
    """Check that heading levels are represented in the heading path."""

    elements = [
        DocumentElement(
            text="Leave Policy",
            element_type="heading",
            source="policy.txt",
            heading_level=1,
        ),
        DocumentElement(
            text="Approval Process",
            element_type="heading",
            source="policy.txt",
            heading_level=2,
        ),
        DocumentElement(
            text="The manager approves leave requests.",
            element_type="paragraph",
            source="policy.txt",
            page_number=2,
        ),
    ]

    chunks = chunk_document(elements, document_id="doc_002")

    assert len(chunks) == 1
    assert chunks[0].heading_path == (
        "Leave Policy",
        "Approval Process",
    )
    assert chunks[0].page_numbers == (2,)


def test_chunk_document_does_not_merge_different_sections():
    """Check that a new heading starts a separate chunk."""

    elements = [
        DocumentElement(
            text="Leave Policy",
            element_type="heading",
            source="policy.txt",
            heading_level=1,
        ),
        DocumentElement(
            text="Employees receive annual leave.",
            element_type="paragraph",
            source="policy.txt",
            page_number=1,
        ),
        DocumentElement(
            text="Approval Process",
            element_type="heading",
            source="policy.txt",
            heading_level=1,
        ),
        DocumentElement(
            text="Managers approve leave requests.",
            element_type="paragraph",
            source="policy.txt",
            page_number=2,
        ),
    ]

    chunks = chunk_document(elements, document_id="doc_003")

    assert len(chunks) == 2
    assert chunks[0].heading_path == ("Leave Policy",)
    assert chunks[1].heading_path == ("Approval Process",)


def test_chunk_document_splits_long_paragraphs():
    """Check that oversized paragraphs are divided into smaller chunks."""

    text = "This is a sentence about enterprise documents. " * 20

    elements = [
        DocumentElement(
            text=text,
            element_type="paragraph",
            source="long_document.txt",
            page_number=1,
        )
    ]

    chunks = chunk_document(
        elements,
        document_id="doc_004",
        chunk_size=100,
    )

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.document_id == "doc_004" for chunk in chunks)


def test_chunk_document_rejects_invalid_settings():
    """Checks that invalid document and chunk settings raise exceptions."""

    # A Zero chunk size is invalid.
    with pytest.raises(ValueError):
        chunk_document([], document_id="doc_005", chunk_size=0)

    # A negative chunk size is invalid.
    with pytest.raises(ValueError):
        chunk_document([], document_id="doc_005", chunk_size=-10)
    # An empty document ID is invalid.
    with pytest.raises(ValueError):
        chunk_document([], document_id="   ", chunk_size=100)
def test_long_text_has_no_overlapping_content():
    """Verify that splitting text does not repeat content between pieces."""

    from src.ingestion.chunker import _split_long_text

    original_text = (
        "Alpha sentence. Bravo sentence. "
        "Charlie sentence. Delta sentence."
    )

    pieces = _split_long_text(original_text, chunk_size=25)

    # Joining the pieces should reconstruct the original text.
    assert " ".join(pieces) == original_text
    assert len(pieces) > 1


def test_neighbor_links_are_created_within_same_section():
    """Verify that chunks in the same section link to their immediate neighbors."""

    elements = [
        DocumentElement(
            text="Leave Policy",
            element_type="heading",
            source="policy.txt",
            heading_level=1,
        ),
        DocumentElement(
            text=(
                "Alpha sentence. Bravo sentence. "
                "Charlie sentence. Delta sentence."
            ),
            element_type="paragraph",
            source="policy.txt",
            page_number=1,
        ),
    ]

    chunks = chunk_document(
        elements,
        document_id="policy",
        chunk_size=25,
    )

    assert len(chunks) > 2

    # The first chunk has no previous neighbor.
    assert chunks[0].previous_chunk_id is None

    # The last chunk has no next neighbor.
    assert chunks[-1].next_chunk_id is None

    # Every middle chunk links to the chunks immediately around it.
    for index in range(1, len(chunks)):
        assert chunks[index].previous_chunk_id == chunks[index - 1].chunk_id

    for index in range(len(chunks) - 1):
        assert chunks[index].next_chunk_id == chunks[index + 1].chunk_id


def test_neighbor_links_do_not_cross_section_boundaries():
    """Verify that chunks from different heading sections are not linked."""

    elements = [
        DocumentElement(
            text="Leave Policy",
            element_type="heading",
            source="policy.txt",
            heading_level=1,
        ),
        DocumentElement(
            text="Employees must submit a leave request.",
            element_type="paragraph",
            source="policy.txt",
            page_number=1,
        ),
        DocumentElement(
            text="Approval Process",
            element_type="heading",
            source="policy.txt",
            heading_level=2,
        ),
        DocumentElement(
            text="The manager reviews the request.",
            element_type="paragraph",
            source="policy.txt",
            page_number=2,
        ),
    ]

    chunks = chunk_document(
        elements,
        document_id="policy",
        chunk_size=500,
    )

    assert len(chunks) == 2

    # The two chunks belong to different heading paths.
    assert chunks[0].heading_path != chunks[1].heading_path

    # Neither chunk should link across the section boundary.
    assert chunks[0].next_chunk_id is None
    assert chunks[1].previous_chunk_id is None
