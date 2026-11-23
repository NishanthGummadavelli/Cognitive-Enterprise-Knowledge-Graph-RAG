
"""Orchestrate strict document-grounded question answering."""

from src.rag.citation_manager import CitationManager
from src.rag.context_builder import ContextBuilder
from src.rag.llm_generator import LLMGenerator
from src.rag.models import (
    AnswerSection,
    EvidenceOrigin,
    RAGResponse,
    RequestMode,
    ScopeType,
    SourceReference,
)
from src.rag.prompt_builder import PromptBuilder


class RAGPipeline:
    """Coordinate retrieval, context building, generation, and citations."""

    INSUFFICIENT_EVIDENCE_PHRASE = (
        "the available documents do not contain sufficient information"
    )

    def __init__(
        self,
        retriever,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_generator: LLMGenerator | None = None,
        citation_manager: CitationManager | None = None,
    ) -> None:
        """Initialize the pipeline with its required components."""

        if retriever is None:
            raise ValueError("retriever must be provided.")

        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_generator = llm_generator or LLMGenerator()
        self.citation_manager = citation_manager or CitationManager()

    @staticmethod
    def _validate_inputs(
        question: str,
        scope_type: ScopeType,
        scope_id: str,
        top_k: int,
    ) -> None:
        """Validate the question, knowledge scope, and retrieval limit."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        if not isinstance(scope_type, ScopeType):
            raise TypeError("scope_type must be a ScopeType instance.")

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("scope_id must be a non-empty string.")

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 50
        ):
            raise ValueError("top_k must be an integer between 1 and 50.")

    @staticmethod
    def _convert_sources(
        context_package,
    ) -> list[SourceReference]:
        """Convert context sources into the shared response source model."""

        return [
            SourceReference(
                citation_id=source.citation_id,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                source=source.source,
                page_numbers=source.page_numbers,
                heading_path=source.heading_path,
            )
            for source in context_package.sources
        ]

    @classmethod
    def _is_insufficient_evidence(cls, answer: str) -> bool:
        """Identify the pipeline's explicit insufficient-evidence response."""

        normalized_answer = answer.casefold()

        return (
            cls.INSUFFICIENT_EVIDENCE_PHRASE in normalized_answer
            or "insufficient evidence to answer" in normalized_answer
        )

    def answer(
        self,
        question: str,
        scope_type: ScopeType,
        scope_id: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """Generate a strictly document-grounded answer with citations."""

        self._validate_inputs(
            question=question,
            scope_type=scope_type,
            scope_id=scope_id,
            top_k=top_k,
        )

        # Retrieve evidence only from the requested knowledge scope.
        retrieval_results = self.retriever.search(
            query=question.strip(),
            scope_type=scope_type.value,
            scope_id=scope_id,
            top_k=top_k,
            retrieval_top_k=max(top_k * 2, 10),
        )

        if not isinstance(retrieval_results, list):
            raise TypeError("Retriever must return a list of results.")

        # Prepare and validate the evidence before generation.
        context_package = self.context_builder.build(
            retrieval_results=retrieval_results,
            scope_type=scope_type.value,
            scope_id=scope_id,
        )

        sources = self._convert_sources(context_package)

        # Avoid generating an answer when retrieval found no evidence.
        if not context_package.sources:
            return RAGResponse(
                mode=RequestMode.DOCUMENT_QA,
                answer=(
                    "The available documents do not contain sufficient "
                    "information to answer this question."
                ),
                sources=[],
                insufficient_evidence=True,
            )

        prompt_package = self.prompt_builder.build_document_qa(
            question=question,
            context_package=context_package,
        )

        generated_answer = self.llm_generator.generate(prompt_package)

        # Validate all generated citation identifiers.
        validation = self.citation_manager.validate(
            answer=generated_answer,
            sources=sources,
        )

        if validation.unknown_citations:
            raise RuntimeError(
                "Generated answer contains unknown citations: "
                + ", ".join(validation.unknown_citations)
            )

        insufficient_evidence = self._is_insufficient_evidence(
            generated_answer
        )

        # Substantive document answers must contain citations.
        if validation.missing_citations and not insufficient_evidence:
            raise RuntimeError(
                "Generated document-grounded answer contains no citations."
            )

        cited_sources = self.citation_manager.filter_valid_sources(
            answer=generated_answer,
            sources=sources,
        )

        sections: list[AnswerSection] = []

        if not insufficient_evidence:
            sections.append(
                AnswerSection(
                    heading="Answer from uploaded documents",
                    content=generated_answer,
                    origin=EvidenceOrigin.DOCUMENT,
                    citations=validation.found_citations,
                )
            )

        return RAGResponse(
            mode=RequestMode.DOCUMENT_QA,
            answer=generated_answer,
            sections=sections,
            sources=cited_sources,
            insufficient_evidence=insufficient_evidence,
            general_knowledge_used=False,
        )
