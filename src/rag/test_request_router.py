
"""Test request routing across the three RAG capabilities."""

import pytest
from unittest.mock import MagicMock

from src.rag.models import (
    RAGRequest,
    RAGResponse,
    RequestMode,
    ScopeType,
    SummaryStyle,
)
from src.rag.request_router import RequestRouter


def create_response(mode: RequestMode) -> RAGResponse:
    """Create a minimal valid response for router tests."""

    return RAGResponse(
        mode=mode,
        answer="Test response.",
    )


@pytest.fixture
def mock_pipelines():
    """Create mocked instances of all three processing pipelines."""

    rag_pipeline = MagicMock()
    summarization_pipeline = MagicMock()
    knowledge_augmented_pipeline = MagicMock()

    rag_pipeline.answer.return_value = create_response(
        RequestMode.DOCUMENT_QA
    )

    summarization_pipeline.summarize.return_value = create_response(
        RequestMode.DOCUMENT_SUMMARY
    )

    knowledge_augmented_pipeline.answer.return_value = create_response(
        RequestMode.KNOWLEDGE_AUGMENTED_QA
    )

    return (
        rag_pipeline,
        summarization_pipeline,
        knowledge_augmented_pipeline,
    )


@pytest.fixture
def router(mock_pipelines):
    """Create a router using the mocked processing pipelines."""

    return RequestRouter(*mock_pipelines)


def test_document_qa_routing(router, mock_pipelines):
    """Verify that document Q&A requests reach the RAG pipeline."""

    rag_pipeline, summarization_pipeline, knowledge_pipeline = mock_pipelines

    request = RAGRequest(
        mode=RequestMode.DOCUMENT_QA,
        question="What is the company policy?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user_123",
        top_k=8,
    )

    response = router.process(request)

    assert response.mode == RequestMode.DOCUMENT_QA
    rag_pipeline.answer.assert_called_once_with(
        question="What is the company policy?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user_123",
        top_k=8,
    )

    summarization_pipeline.summarize.assert_not_called()
    knowledge_pipeline.answer.assert_not_called()


def test_document_summary_routing(router, mock_pipelines):
    """Verify that summary requests reach the summarization pipeline."""

    rag_pipeline, summarization_pipeline, knowledge_pipeline = mock_pipelines

    request = RAGRequest(
        mode=RequestMode.DOCUMENT_SUMMARY,
        document_ids=["doc_001", "doc_002"],
        scope_type=ScopeType.PROJECT,
        scope_id="project_123",
        summary_style=SummaryStyle.EXECUTIVE,
    )

    response = router.process(request)

    assert response.mode == RequestMode.DOCUMENT_SUMMARY
    summarization_pipeline.summarize.assert_called_once_with(
        document_ids=["doc_001", "doc_002"],
        scope_type=ScopeType.PROJECT,
        scope_id="project_123",
        summary_style=SummaryStyle.EXECUTIVE,
    )

    rag_pipeline.answer.assert_not_called()
    knowledge_pipeline.answer.assert_not_called()


def test_knowledge_augmented_qa_routing(router, mock_pipelines):
    """Verify that augmented Q&A reaches the correct pipeline."""

    rag_pipeline, summarization_pipeline, knowledge_pipeline = mock_pipelines

    request = RAGRequest(
        mode=RequestMode.KNOWLEDGE_AUGMENTED_QA,
        question="Explain this technical concept.",
        scope_type=ScopeType.PROJECT,
        scope_id="project_123",
        top_k=10,
    )

    response = router.process(request)

    assert response.mode == RequestMode.KNOWLEDGE_AUGMENTED_QA
    knowledge_pipeline.answer.assert_called_once_with(
        question="Explain this technical concept.",
        scope_type=ScopeType.PROJECT,
        scope_id="project_123",
        top_k=10,
    )

    rag_pipeline.answer.assert_not_called()
    summarization_pipeline.summarize.assert_not_called()


def test_router_rejects_non_rag_request(router):
    """Verify that the router rejects objects other than RAGRequest."""

    with pytest.raises(TypeError, match="RAGRequest"):
        router.process({"mode": "document_qa"})


def test_router_rejects_invalid_pipeline_response(mock_pipelines):
    """Verify that the router rejects an invalid pipeline response."""

    rag_pipeline, summarization_pipeline, knowledge_pipeline = mock_pipelines

    rag_pipeline.answer.return_value = "Invalid response"

    router = RequestRouter(
        rag_pipeline,
        summarization_pipeline,
        knowledge_pipeline,
    )

    request = RAGRequest(
        mode=RequestMode.DOCUMENT_QA,
        question="Test question",
        scope_type=ScopeType.PERSONAL,
        scope_id="user_123",
    )

    with pytest.raises(TypeError, match="RAGResponse"):
        router.process(request)


def test_router_does_not_call_other_pipelines(router, mock_pipelines):
    """Verify that each request invokes only its selected pipeline."""

    rag_pipeline, summarization_pipeline, knowledge_pipeline = mock_pipelines

    request = RAGRequest(
        mode=RequestMode.DOCUMENT_QA,
        question="Test question",
        scope_type=ScopeType.PERSONAL,
        scope_id="user_123",
    )

    router.process(request)

    assert rag_pipeline.answer.call_count == 1
    assert summarization_pipeline.summarize.call_count == 0
    assert knowledge_pipeline.answer.call_count == 0
