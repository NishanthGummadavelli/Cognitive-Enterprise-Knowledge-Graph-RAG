
"""Unit tests for the knowledge-augmented RAG pipeline."""

import pytest

from src.rag.context_builder import ContextPackage, ContextSource
from src.rag.evidence_assessor import EvidenceAssessment
from src.rag.knowledge_augmented_pipeline import KnowledgeAugmentedPipeline
from src.rag.models import EvidenceOrigin, RequestMode, ScopeType
from src.rag.prompt_builder import PromptPackage


class FakeRetriever:
    """Provide deterministic retrieval results for pipeline tests."""

    def __init__(self, results=None):
        """Initialize the fake retriever with configured results."""

        self.results = results if results is not None else []
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

        self.calls.append({
            "query": query,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "top_k": top_k,
            "retrieval_top_k": retrieval_top_k,
        })

        return self.results


class FakeContextBuilder:
    """Return a prepared context package for pipeline tests."""

    def __init__(self, context_package):
        """Store the context package used by the test."""

        self.context_package = context_package
        self.calls = []

    def build(self, retrieval_results, scope_type, scope_id):
        """Record context construction and return the configured package."""

        self.calls.append({
            "retrieval_results": retrieval_results,
            "scope_type": scope_type,
            "scope_id": scope_id,
        })

        return self.context_package


class FakePromptBuilder:
    """Capture the type of prompt requested by the pipeline."""

    def __init__(self):
        """Initialize prompt construction history."""

        self.document_calls = []
        self.augmented_calls = []

    def build_document_qa(self, question, context_package):
        """Capture a strict document Q&A prompt request."""

        self.document_calls.append({
            "question": question,
            "context_package": context_package,
        })

        return PromptPackage(
            system_prompt="Strict document prompt",
            user_prompt="Document question",
        )

    def build_knowledge_augmented_qa(self, question, context_package):
        """Capture a knowledge-augmented prompt request."""

        self.augmented_calls.append({
            "question": question,
            "context_package": context_package,
        })

        return PromptPackage(
            system_prompt="Knowledge-augmented prompt",
            user_prompt="Augmented question",
        )


class FakeLLMGenerator:
    """Return configured answers without invoking Ollama."""

    def __init__(self, answer):
        """Configure the generated answer."""

        self.answer = answer
        self.calls = []

    def generate(self, prompt_package):
        """Record the prompt and return the configured answer."""

        self.calls.append(prompt_package)
        return self.answer


class FakeEvidenceAssessor:
    """Return a predetermined evidence assessment."""

    def __init__(self, sufficient=True):
        """Configure whether the document evidence is sufficient."""

        self.sufficient = sufficient
        self.calls = []

    def assess(self, question, context_package):
        """Record the assessment request and return its configured result."""

        self.calls.append({
            "question": question,
            "context_package": context_package,
        })

        return EvidenceAssessment(
            sufficient=self.sufficient,
            confidence=0.90,
            reason="Test evidence assessment.",
            covered_aspects=(
                ["Annual leave entitlement"] if self.sufficient else []
            ),
            missing_aspects=(
                [] if self.sufficient else ["Additional policy details"]
            ),
        )


def make_context_package(with_source=True):
    """Create a context package with an optional document source."""

    sources = []

    context_text = ""

    if with_source:
        sources.append(
            ContextSource(
                citation_id="S1",
                chunk_id="chunk-1",
                document_id="document-1",
                source="Employee Handbook.pdf",
                page_numbers=[12],
                heading_path=["Leave Policy"],
                text="Employees receive 20 days of annual leave.",
                retrieval_score=0.85,
            )
        )

        context_text = (
            "[S1] Employee Handbook.pdf, Page 12\n"
            "Employees receive 20 days of annual leave."
        )

    return ContextPackage(
        context_text=context_text,
        sources=sources,
        scope_type="personal",
        scope_id="user-1",
        included_chunks=len(sources),
        omitted_chunk_ids=[],
    )


def make_retrieval_result():
    """Create a minimal retrieval result in the requested scope."""

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
    assessment_sufficient=True,
    answer="The policy allows 20 days of annual leave. [S1]",
    context_package=None,
    retrieval_results=None,
):
    """Construct a pipeline with deterministic fake dependencies."""

    if context_package is None:
        context_package = make_context_package()

    if retrieval_results is None:
        retrieval_results = [make_retrieval_result()]

    retriever = FakeRetriever(retrieval_results)
    context_builder = FakeContextBuilder(context_package)
    prompt_builder = FakePromptBuilder()
    llm_generator = FakeLLMGenerator(answer)
    evidence_assessor = FakeEvidenceAssessor(assessment_sufficient)

    pipeline = KnowledgeAugmentedPipeline(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_generator=llm_generator,
        evidence_assessor=evidence_assessor,
    )

    return (
        pipeline,
        retriever,
        context_builder,
        prompt_builder,
        llm_generator,
        evidence_assessor,
    )


def test_sufficient_evidence_uses_document_qa():
    """Use strict document generation when evidence is sufficient."""

    pipeline, _, _, prompt_builder, generator, assessor = make_pipeline()

    response = pipeline.answer(
        "How many annual leave days are allowed?",
        ScopeType.PERSONAL,
        "user-1",
    )

    assert response.mode == RequestMode.KNOWLEDGE_AUGMENTED_QA
    assert response.insufficient_evidence is False
    assert response.general_knowledge_used is False
    assert len(response.sections) == 1
    assert response.sections[0].origin == EvidenceOrigin.DOCUMENT
    assert response.sources[0].citation_id == "S1"

    assert len(prompt_builder.document_calls) == 1
    assert len(prompt_builder.augmented_calls) == 0
    assert len(generator.calls) == 1
    assert len(assessor.calls) == 1


def test_insufficient_evidence_uses_knowledge_augmentation():
    """Use the augmented prompt when document evidence is insufficient."""

    answer = (
        "From uploaded documents\n"
        "Employees receive 20 days of annual leave. [S1]\n\n"
        "Additional general knowledge\n"
        "Annual leave policies vary by organization."
    )

    pipeline, _, _, prompt_builder, _, _ = make_pipeline(
        assessment_sufficient=False,
        answer=answer,
    )

    response = pipeline.answer(
        "Explain annual leave policies.",
        ScopeType.PERSONAL,
        "user-1",
    )

    assert response.insufficient_evidence is True
    assert response.general_knowledge_used is True
    assert len(response.sections) == 2

    assert response.sections[0].origin == EvidenceOrigin.DOCUMENT
    assert response.sections[0].citations == ["S1"]

    assert response.sections[1].origin == EvidenceOrigin.GENERAL_KNOWLEDGE
    assert response.sections[1].citations == []

    assert len(prompt_builder.document_calls) == 0
    assert len(prompt_builder.augmented_calls) == 1


def test_no_document_evidence_is_disclosed():
    """Disclose missing document evidence before presenting general knowledge."""

    context_package = make_context_package(with_source=False)

    answer = (
        "Additional general knowledge\n"
        "Annual leave is a period of paid time off granted to employees."
    )

    pipeline, _, _, _, _, _ = make_pipeline(
        assessment_sufficient=False,
        answer=answer,
        context_package=context_package,
        retrieval_results=[],
    )

    response = pipeline.answer(
        "What is annual leave?",
        ScopeType.PERSONAL,
        "user-1",
    )

    assert response.sources == []
    assert response.insufficient_evidence is True
    assert response.general_knowledge_used is True
    assert len(response.sections) == 1
    assert response.sections[0].origin == EvidenceOrigin.GENERAL_KNOWLEDGE
    assert "No relevant document evidence was found" in response.sections[0].content


def test_retrieval_uses_requested_scope():
    """Verify retrieval receives the requested knowledge scope."""

    pipeline, retriever, _, _, _, _ = make_pipeline()

    pipeline.answer(
        "What is the leave policy?",
        ScopeType.PROJECT,
        "project-123",
        top_k=7,
    )

    assert retriever.calls[0]["scope_type"] == "project"
    assert retriever.calls[0]["scope_id"] == "project-123"
    assert retriever.calls[0]["top_k"] == 7
    assert retriever.calls[0]["retrieval_top_k"] == 14


def test_context_is_built_before_evidence_assessment():
    """Verify context construction and evidence assessment occur."""

    pipeline, _, context_builder, _, _, assessor = make_pipeline()

    pipeline.answer(
        "What is the leave policy?",
        ScopeType.PERSONAL,
        "user-1",
    )

    assert len(context_builder.calls) == 1
    assert len(assessor.calls) == 1
    assert assessor.calls[0]["context_package"] is context_builder.context_package


def test_unknown_document_citation_is_rejected():
    """Reject document citations that do not exist in retrieved sources."""

    answer = (
        "From uploaded documents\n"
        "The policy allows 20 days of leave. [S9]\n\n"
        "Additional general knowledge\n"
        "Annual leave varies between organizations."
    )

    pipeline, *_ = make_pipeline(
        assessment_sufficient=False,
        answer=answer,
    )

    with pytest.raises(RuntimeError, match="unknown citations"):
        pipeline.answer(
            "What is the leave policy?",
            ScopeType.PERSONAL,
            "user-1",
        )


def test_document_section_without_citations_is_rejected():
    """Reject substantive document content without citations."""

    answer = (
        "From uploaded documents\n"
        "The policy allows 20 days of leave.\n\n"
        "Additional general knowledge\n"
        "Annual leave varies between organizations."
    )

    pipeline, *_ = make_pipeline(
        assessment_sufficient=False,
        answer=answer,
    )

    with pytest.raises(RuntimeError, match="without document citations"):
        pipeline.answer(
            "What is the leave policy?",
            ScopeType.PERSONAL,
            "user-1",
        )


def test_missing_general_knowledge_section_is_rejected():
    """Reject augmented answers without a general-knowledge section."""

    answer = (
        "From uploaded documents\n"
        "The policy allows 20 days of leave. [S1]"
    )

    pipeline, *_ = make_pipeline(
        assessment_sufficient=False,
        answer=answer,
    )

    with pytest.raises(RuntimeError, match="Additional general knowledge"):
        pipeline.answer(
            "What is the leave policy?",
            ScopeType.PERSONAL,
            "user-1",
        )


def test_general_knowledge_section_cannot_have_citations():
    """Reject citation markers inside the general-knowledge section."""

    answer = (
        "From uploaded documents\n"
        "The policy allows 20 days of leave. [S1]\n\n"
        "Additional general knowledge\n"
        "Annual leave varies between organizations. [S1]"
    )

    pipeline, *_ = make_pipeline(
        assessment_sufficient=False,
        answer=answer,
    )

    with pytest.raises(RuntimeError, match="general-knowledge section"):
        pipeline.answer(
            "What is the leave policy?",
            ScopeType.PERSONAL,
            "user-1",
        )


def test_text_outside_source_headings_is_rejected():
    """Reject generated text that is not assigned to a source section."""

    answer = (
        "Here is the answer.\n\n"
        "From uploaded documents\n"
        "The policy allows 20 days of leave. [S1]\n\n"
        "Additional general knowledge\n"
        "Annual leave varies between organizations."
    )

    pipeline, *_ = make_pipeline(
        assessment_sufficient=False,
        answer=answer,
    )

    with pytest.raises(RuntimeError, match="outside"):
        pipeline.answer(
            "What is the leave policy?",
            ScopeType.PERSONAL,
            "user-1",
        )


def test_invalid_scope_type_is_rejected():
    """Reject a scope type that is not a ScopeType enum."""

    pipeline, *_ = make_pipeline()

    with pytest.raises(TypeError, match="ScopeType"):
        pipeline.answer(
            "What is the leave policy?",
            "personal",
            "user-1",
        )


def test_invalid_top_k_is_rejected():
    """Reject retrieval limits outside the supported range."""

    pipeline, *_ = make_pipeline()

    with pytest.raises(ValueError, match="top_k"):
        pipeline.answer(
            "What is the leave policy?",
            ScopeType.PERSONAL,
            "user-1",
            top_k=0,
        )


def test_empty_question_is_rejected():
    """Reject empty questions before retrieval."""

    pipeline, retriever, *_ = make_pipeline()

    with pytest.raises(ValueError, match="question"):
        pipeline.answer(
            "   ",
            ScopeType.PERSONAL,
            "user-1",
        )

    assert retriever.calls == []


def test_missing_retriever_is_rejected():
    """Require a retriever when constructing the pipeline."""

    with pytest.raises(ValueError, match="retriever must be provided"):
        KnowledgeAugmentedPipeline(retriever=None)


def test_non_list_retrieval_results_are_rejected():
    """Reject malformed retriever output."""

    pipeline, retriever, *_ = make_pipeline()
    retriever.results = None

    with pytest.raises(TypeError, match="must return a list"):
        pipeline.answer(
            "What is the leave policy?",
            ScopeType.PERSONAL,
            "user-1",
        )
