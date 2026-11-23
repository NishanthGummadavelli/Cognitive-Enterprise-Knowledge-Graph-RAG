
"""Tests for shared RAG request and response models."""

import pytest
from pydantic import ValidationError

from src.rag.models import (
    AnswerSection,
    EvidenceOrigin,
    RAGRequest,
    RAGResponse,
    RequestMode,
    ScopeType,
    SourceReference,
    SummaryStyle,
)


def test_document_qa_request() -> None:
    """Accept a valid document-grounded Q&A request."""

    request = RAGRequest(
        mode=RequestMode.DOCUMENT_QA,
        scope_type=ScopeType.PERSONAL,
        scope_id="user-101",
        question="What is the annual leave policy?",
    )

    assert request.mode == RequestMode.DOCUMENT_QA
    assert request.question == "What is the annual leave policy?"


def test_knowledge_augmented_request() -> None:
    """Accept a valid knowledge-augmented Q&A request."""

    request = RAGRequest(
        mode=RequestMode.KNOWLEDGE_AUGMENTED_QA,
        scope_type=ScopeType.PROJECT,
        scope_id="project-101",
        question="Explain the project architecture.",
    )

    assert request.mode == RequestMode.KNOWLEDGE_AUGMENTED_QA


def test_document_summary_request() -> None:
    """Accept a valid document summarization request."""

    request = RAGRequest(
        mode=RequestMode.DOCUMENT_SUMMARY,
        scope_type=ScopeType.PERSONAL,
        scope_id="user-101",
        document_ids=["doc-001", "doc-002"],
    )

    assert request.document_ids == ["doc-001", "doc-002"]
    assert request.summary_style == SummaryStyle.COMPREHENSIVE


def test_document_qa_requires_question() -> None:
    """Reject a Q&A request without a question."""

    with pytest.raises(ValidationError):
        RAGRequest(
            mode=RequestMode.DOCUMENT_QA,
            scope_type=ScopeType.PERSONAL,
            scope_id="user-101",
        )


def test_knowledge_augmented_qa_requires_question() -> None:
    """Reject a knowledge-augmented request without a question."""

    with pytest.raises(ValidationError):
        RAGRequest(
            mode=RequestMode.KNOWLEDGE_AUGMENTED_QA,
            scope_type=ScopeType.PERSONAL,
            scope_id="user-101",
        )


def test_summary_requires_document_ids() -> None:
    """Reject a summary request without document identifiers."""

    with pytest.raises(ValidationError):
        RAGRequest(
            mode=RequestMode.DOCUMENT_SUMMARY,
            scope_type=ScopeType.PERSONAL,
            scope_id="user-101",
        )


def test_summary_rejects_duplicate_document_ids() -> None:
    """Reject duplicate document identifiers."""

    with pytest.raises(ValidationError):
        RAGRequest(
            mode=RequestMode.DOCUMENT_SUMMARY,
            scope_type=ScopeType.PERSONAL,
            scope_id="user-101",
            document_ids=["doc-001", "doc-001"],
        )


def test_empty_scope_id_is_rejected() -> None:
    """Reject requests without a valid scope identifier."""

    with pytest.raises(ValidationError):
        RAGRequest(
            mode=RequestMode.DOCUMENT_QA,
            scope_type=ScopeType.PERSONAL,
            scope_id="",
            question="What is the policy?",
        )


def test_invalid_top_k_is_rejected() -> None:
    """Reject retrieval limits outside the supported range."""

    with pytest.raises(ValidationError):
        RAGRequest(
            mode=RequestMode.DOCUMENT_QA,
            scope_type=ScopeType.PERSONAL,
            scope_id="user-101",
            question="What is the policy?",
            top_k=100,
        )


def test_source_reference_creation() -> None:
    """Create a document source reference with citation metadata."""

    source = SourceReference(
        citation_id="S1",
        chunk_id="chunk-001",
        document_id="doc-001",
        source="HR_Policy.pdf",
        page_numbers=[2, 3],
        heading_path=["Human Resources", "Leave Policy"],
    )

    assert source.citation_id == "S1"
    assert source.page_numbers == [2, 3]


def test_document_answer_response() -> None:
    """Accept a document-grounded answer with a valid citation."""

    response = RAGResponse(
        mode=RequestMode.DOCUMENT_QA,
        answer="Employees receive 20 days of annual leave. [S1]",
        sections=[
            AnswerSection(
                heading="Annual Leave",
                content="Employees receive 20 days of annual leave.",
                origin=EvidenceOrigin.DOCUMENT,
                citations=["S1"],
            )
        ],
        sources=[
            SourceReference(
                citation_id="S1",
                chunk_id="chunk-001",
                document_id="doc-001",
                source="HR_Policy.pdf",
            )
        ],
    )

    assert response.general_knowledge_used is False
    assert response.sections[0].citations == ["S1"]


def test_knowledge_augmented_response() -> None:
    """Accept separate document and general-knowledge answer sections."""

    response = RAGResponse(
        mode=RequestMode.KNOWLEDGE_AUGMENTED_QA,
        answer="The company provides 20 days of leave. [S1]",
        sections=[
            AnswerSection(
                heading="From uploaded documents",
                content="The company provides 20 days of leave.",
                origin=EvidenceOrigin.DOCUMENT,
                citations=["S1"],
            ),
            AnswerSection(
                heading="Additional general information",
                content="Some organizations also allow leave carry-forward.",
                origin=EvidenceOrigin.GENERAL_KNOWLEDGE,
                citations=[],
            ),
        ],
        sources=[
            SourceReference(
                citation_id="S1",
                chunk_id="chunk-001",
                document_id="doc-001",
                source="HR_Policy.pdf",
            )
        ],
        general_knowledge_used=True,
    )

    assert response.general_knowledge_used is True
    assert len(response.sections) == 2


def test_general_knowledge_cannot_claim_document_citation() -> None:
    """Reject general-knowledge sections containing document citations."""

    with pytest.raises(ValidationError):
        RAGResponse(
            mode=RequestMode.KNOWLEDGE_AUGMENTED_QA,
            answer="Some organizations allow carry-forward. [S1]",
            sections=[
                AnswerSection(
                    heading="General information",
                    content="Some organizations allow carry-forward.",
                    origin=EvidenceOrigin.GENERAL_KNOWLEDGE,
                    citations=["S1"],
                )
            ],
            sources=[
                SourceReference(
                    citation_id="S1",
                    chunk_id="chunk-001",
                    document_id="doc-001",
                    source="HR_Policy.pdf",
                )
            ],
            general_knowledge_used=True,
        )


def test_unknown_citation_is_rejected() -> None:
    """Reject citations that do not reference an available source."""

    with pytest.raises(ValidationError):
        RAGResponse(
            mode=RequestMode.DOCUMENT_QA,
            answer="Employees receive 20 days of leave. [S99]",
            sections=[
                AnswerSection(
                    heading="Leave",
                    content="Employees receive 20 days of leave.",
                    origin=EvidenceOrigin.DOCUMENT,
                    citations=["S99"],
                )
            ],
        )


def test_strict_document_qa_rejects_general_knowledge() -> None:
    """Prevent general knowledge from entering strict document Q&A."""

    with pytest.raises(ValidationError):
        RAGResponse(
            mode=RequestMode.DOCUMENT_QA,
            answer="General information.",
            general_knowledge_used=True,
        )


def test_empty_document_id_is_rejected() -> None:
    """Reject blank document identifiers in summary requests."""

    with pytest.raises(ValidationError):
        RAGRequest(
            mode=RequestMode.DOCUMENT_SUMMARY,
            scope_type=ScopeType.PERSONAL,
            scope_id="user-101",
            document_ids=["doc-001", ""],
        )


def test_general_knowledge_mode_without_documents_is_valid() -> None:
    """Allow knowledge-augmented requests without assuming documents exist."""

    request = RAGRequest(
        mode=RequestMode.KNOWLEDGE_AUGMENTED_QA,
        scope_type=ScopeType.PERSONAL,
        scope_id="user-101",
        question="Explain artificial intelligence.",
    )

    assert request.document_ids is None
