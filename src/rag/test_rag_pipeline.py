
"""Unit tests for the strict document-grounded RAG pipeline."""

import pytest

from src.rag.context_builder import ContextBuilder, ContextPackage, ContextSource
from src.rag.models import (
    EvidenceOrigin,
    RAGResponse,
    RequestMode,
    ScopeType,
    SourceReference,
)
from src.rag.prompt_builder import PromptPackage
from src.rag.rag_pipeline import RAGPipeline


class FakeRetriever:
    """Provide deterministic retrieval results for pipeline tests."""

    def __init__(self, results=None):
        """Initialize the fake retriever with optional results."""

        self.results = results or []
        self.calls = []

    def search(
        self,
        query,
        scope_type,
        scope_id,
        top_k=5,
        retrieval_top_k=10,
    ):
        """Record the search request and return configured results."""

        self.calls.append(
            {
                "query": query,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "top_k": top_k,
                "retrieval_top_k": retrieval_top_k,
            }
        )
        return self.results


class FakeContextBuilder:
    """Return a prepared context package without real retrieval metadata."""

    def __init__(self, context_package):
        """Store the context package used by the test."""

        self.context_package = context_package
        self.calls = []

    def build(self, retrieval_results, scope_type, scope_id):
        """Record the context-building request and return its package."""

        self.calls.append(
            {
                "retrieval_results": retrieval_results,
                "scope_type": scope_type,
                "scope_id": scope_id,
            }
        )
        return self.context_package


class FakePromptBuilder:
    """Capture prompt construction requests."""

    def __init__(self):
        """Initialize the prompt call history."""

        self.calls = []

    def build_document_qa(self, question, context_package):
        """Record the document question and return a fake prompt."""

        self.calls.append(
            {
                "question": question,
                "context_package": context_package,
            }
        )
        return PromptPackage(
            system_prompt="Test system prompt",
            user_prompt="Test user prompt",
        )


class FakeLLMGenerator:
    """Return predefined model answers without invoking Ollama."""

    def __init__(self, answer="The policy allows 20 days of annual leave. [S1]"):
        """Configure the model response."""

        self.answer = answer
        self.calls = []

    def generate(self, prompt_package):
        """Record the prompt and return the configured answer."""

        self.calls.append(prompt_package)
        return self.answer


class FakeCitationManager:
    """Use the real citation manager for pipeline validation."""

    def __new__(cls):
        """Return the real citation manager implementation."""

        from src.rag.citation_manager import CitationManager

        return CitationManager()


def make_context_package():
    """Create a valid context package containing one source."""

    source = ContextSource(
        citation_id="S1",
        chunk_id="chunk-1",
        document_id="document-1",
        source="Employee Handbook.pdf",
        page_numbers=[12],
        heading_path=["Leave Policy"],
        text="Employees receive 20 days of annual leave.",
        retrieval_score=0.85,
    )

    return ContextPackage(
        context_text=(
            "[S1] Employee Handbook.pdf, Page 12\n"
            "Employees receive 20 days of annual leave."
        ),
        sources=[source],
        scope_type="personal",
        scope_id="user-1",
        included_chunks=1,
        omitted_chunk_ids=[],
    )


def make_retrieval_result():
    """Create a minimal result with the requested scope."""

    return {
        "chunk_id": "chunk-1",
        "text": "Employees receive 20 days of annual leave.",
        "scope_type": "personal",
        "scope_id": "user-1",
        "metadata": {
            "document_id": "document-1",
            "source": "Employee Handbook.pdf",
            "page_numbers": [12],
            "heading_path": ["Leave Policy"],
        },
        "rrf_score": 0.85,
    }


def make_pipeline(
    answer="The policy allows 20 days of annual leave. [S1]",
    retrieval_results=None,
    context_package=None,
):
    """Construct a pipeline using deterministic fake dependencies."""

    retriever = FakeRetriever(
        results=(
            [make_retrieval_result()]
            if retrieval_results is None
            else retrieval_results
        )
    )
    context_builder = FakeContextBuilder(
        context_package or make_context_package()
    )
    prompt_builder = FakePromptBuilder()
    llm_generator = FakeLLMGenerator(answer=answer)

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_generator=llm_generator,
    )

    return (
        pipeline,
        retriever,
        context_builder,
        prompt_builder,
        llm_generator,
    )


def test_pipeline_returns_document_grounded_answer():
    """Verify that a valid answer includes its document citation."""

    pipeline, _, _, _, _ = make_pipeline()

    response = pipeline.answer(
        question="How many annual leave days are allowed?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert isinstance(response, RAGResponse)
    assert response.mode == RequestMode.DOCUMENT_QA
    assert response.general_knowledge_used is False
    assert response.insufficient_evidence is False
    assert response.answer.endswith("[S1]")
    assert len(response.sources) == 1
    assert response.sources[0].citation_id == "S1"


def test_pipeline_uses_requested_scope():
    """Verify that retrieval receives the requested knowledge scope."""

    pipeline, retriever, _, _, _ = make_pipeline()

    pipeline.answer(
        question="What is the leave policy?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
        top_k=7,
    )

    assert retriever.calls[0]["scope_type"] == "personal"
    assert retriever.calls[0]["scope_id"] == "user-1"
    assert retriever.calls[0]["top_k"] == 7
    assert retriever.calls[0]["retrieval_top_k"] == 14


def test_pipeline_builds_context_before_generation():
    """Verify that retrieved results are passed to context preparation."""

    pipeline, _, context_builder, _, llm_generator = make_pipeline()

    pipeline.answer(
        question="What is the leave policy?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert len(context_builder.calls) == 1
    assert len(llm_generator.calls) == 1
    assert context_builder.calls[0]["scope_id"] == "user-1"


def test_pipeline_builds_document_qa_prompt():
    """Verify that the strict document Q&A prompt is used."""

    pipeline, _, _, prompt_builder, _ = make_pipeline()

    pipeline.answer(
        question="How many leave days are allowed?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert len(prompt_builder.calls) == 1
    assert (
        prompt_builder.calls[0]["question"]
        == "How many leave days are allowed?"
    )



def test_pipeline_handles_empty_retrieval():
    """Verify that no model answer is generated without retrieved evidence."""

    retriever = FakeRetriever(results=[])
    context_builder = ContextBuilder()
    prompt_builder = FakePromptBuilder()
    llm_generator = FakeLLMGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_generator=llm_generator,
    )

    response = pipeline.answer(
        question="What is the leave policy?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert response.insufficient_evidence is True
    assert response.sources == []
    assert response.sections == []
    assert len(prompt_builder.calls) == 0
    assert len(llm_generator.calls) == 0



def test_pipeline_rejects_unknown_citations():
    """Verify that citations absent from the source list are rejected."""

    pipeline, _, _, _, _ = make_pipeline(
        answer="The policy allows 20 days of leave. [S9]"
    )

    with pytest.raises(RuntimeError, match="unknown citations"):
        pipeline.answer(
            question="What is the leave policy?",
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_pipeline_rejects_answers_without_citations():
    """Verify that substantive document answers require citations."""

    pipeline, _, _, _, _ = make_pipeline(
        answer="The policy allows 20 days of annual leave."
    )

    with pytest.raises(RuntimeError, match="contains no citations"):
        pipeline.answer(
            question="What is the leave policy?",
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_pipeline_accepts_insufficient_evidence_answer():
    """Verify that an explicit insufficient-evidence answer is accepted."""

    pipeline, _, _, _, _ = make_pipeline(
        answer=(
            "The available documents do not contain sufficient "
            "information to answer this question."
        )
    )

    response = pipeline.answer(
        question="What is the company's international transfer policy?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert response.insufficient_evidence is True
    assert response.sections == []


@pytest.mark.parametrize(
    "question,scope_type,scope_id,top_k,error",
    [
        ("", ScopeType.PERSONAL, "user-1", 5, ValueError),
        ("Question", "personal", "user-1", 5, TypeError),
        ("Question", ScopeType.PERSONAL, "", 5, ValueError),
        ("Question", ScopeType.PERSONAL, "user-1", 0, ValueError),
        ("Question", ScopeType.PERSONAL, "user-1", 51, ValueError),
    ],
)
def test_pipeline_rejects_invalid_inputs(
    question,
    scope_type,
    scope_id,
    top_k,
    error,
):
    """Verify that invalid request parameters are rejected."""

    pipeline, _, _, _, _ = make_pipeline()

    with pytest.raises(error):
        pipeline.answer(
            question=question,
            scope_type=scope_type,
            scope_id=scope_id,
            top_k=top_k,
        )


def test_pipeline_rejects_non_list_retrieval_results():
    """Verify that malformed retriever output is rejected."""

    pipeline, _, _, _, _ = make_pipeline()
    pipeline.retriever.results = None

    with pytest.raises(TypeError, match="must return a list"):
        pipeline.answer(
            question="What is the leave policy?",
            scope_type=ScopeType.PERSONAL,
            scope_id="user-1",
        )


def test_pipeline_does_not_use_general_knowledge():
    """Verify that strict document Q&A never marks general knowledge used."""

    pipeline, _, _, _, _ = make_pipeline()

    response = pipeline.answer(
        question="What is the leave policy?",
        scope_type=ScopeType.PERSONAL,
        scope_id="user-1",
    )

    assert response.general_knowledge_used is False
    assert all(
        section.origin == EvidenceOrigin.DOCUMENT
        for section in response.sections
    )


def test_pipeline_rejects_missing_retriever():
    """Verify that a retriever is required during initialization."""

    with pytest.raises(ValueError, match="retriever must be provided"):
        RAGPipeline(retriever=None)
