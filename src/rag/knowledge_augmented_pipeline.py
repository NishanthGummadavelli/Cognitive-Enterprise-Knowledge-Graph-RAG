
"""Orchestrate document-first knowledge-augmented question answering."""

import re

from src.rag.citation_manager import CitationManager
from src.rag.context_builder import ContextBuilder
from src.rag.evidence_assessor import EvidenceSufficiencyAssessor
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


class KnowledgeAugmentedPipeline:
    """Coordinate retrieval, evidence assessment, generation, and citations."""

    DOCUMENT_HEADING = "From uploaded documents"
    GENERAL_HEADING = "Additional general knowledge"
    CITATION_PATTERN = re.compile(r"\[S\d+\]")

    def __init__(
        self,
        retriever,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_generator: LLMGenerator | None = None,
        evidence_assessor: EvidenceSufficiencyAssessor | None = None,
        citation_manager: CitationManager | None = None,
    ) -> None:
        """Initialize the pipeline with its required dependencies."""

        if retriever is None:
            raise ValueError("retriever must be provided.")

        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_generator = llm_generator or LLMGenerator()
        self.evidence_assessor = evidence_assessor or (
            EvidenceSufficiencyAssessor(self.llm_generator)
        )
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
    def _convert_sources(context_package) -> list[SourceReference]:
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

    @staticmethod
    def _normalize_heading(line: str) -> str:
        """Normalize Markdown heading and bold formatting for section matching."""

        normalized = line.strip()
        normalized = re.sub(r"^#{1,6}\s*", "", normalized)
        normalized = normalized.strip("*_ ").strip()
        normalized = normalized.rstrip(":").strip()

        return normalized.casefold()

    @classmethod
    def _split_answer_sections(
        cls,
        answer: str,
    ) -> tuple[str, str]:
        """Extract both source sections and reject unclassified generated text."""

        document_lines: list[str] = []
        general_lines: list[str] = []
        current_section: str | None = None

        for line in answer.splitlines():
            normalized_heading = cls._normalize_heading(line)

            if normalized_heading == cls.DOCUMENT_HEADING.casefold():
                if current_section is not None:
                    raise RuntimeError(
                        "Generated answer contains duplicate or misplaced "
                        "source section headings."
                    )

                current_section = "document"
                continue

            if normalized_heading == cls.GENERAL_HEADING.casefold():
                if current_section == "general":
                    raise RuntimeError(
                        "Generated answer contains duplicate general-knowledge "
                        "headings."
                    )

                current_section = "general"
                continue

            if current_section is None:
                if line.strip():
                    raise RuntimeError(
                        "Generated answer contains unclassified content outside "
                        "the required source headings."
                    )
                continue

            if current_section == "document":
                document_lines.append(line)
            else:
                general_lines.append(line)

        if current_section is None:
            raise RuntimeError(
                "Knowledge-augmented answer does not contain the required "
                "source section headings."
            )

        document_content = "\n".join(document_lines).strip()
        general_content = "\n".join(general_lines).strip()

        return document_content, general_content

    @classmethod
    def _validate_general_section(cls, content: str) -> None:
        """Reject document citation markers in general-knowledge content."""

        if cls.CITATION_PATTERN.search(content):
            raise RuntimeError(
                "Generated general-knowledge section contains document "
                "citation markers. General knowledge must not claim "
                "document citations."
            )

    def _validate_document_section(
        self,
        document_content: str,
        sources: list[SourceReference],
    ) -> tuple[list[str], list[SourceReference]]:
        """Validate document citations and return only cited sources."""

        if not document_content:
            return [], []

        validation = self.citation_manager.validate(
            answer=document_content,
            sources=sources,
        )

        if validation.unknown_citations:
            raise RuntimeError(
                "Generated document section contains unknown citations: "
                + ", ".join(validation.unknown_citations)
            )

        if validation.missing_citations:
            raise RuntimeError(
                "Generated document section contains factual content "
                "without document citations."
            )

        cited_sources = self.citation_manager.filter_valid_sources(
            answer=document_content,
            sources=sources,
        )

        return validation.found_citations, cited_sources

    def answer(
        self,
        question: str,
        scope_type: ScopeType,
        scope_id: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """Generate a document-first answer with clearly separated provenance."""

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

        # Build citation-ready context from retrieved evidence.
        context_package = self.context_builder.build(
            retrieval_results=retrieval_results,
            scope_type=scope_type.value,
            scope_id=scope_id,
        )

        sources = self._convert_sources(context_package)

        # Assess whether the retrieved documents adequately answer the question.
        assessment = self.evidence_assessor.assess(
            question=question.strip(),
            context_package=context_package,
        )

        # Use strict document generation when the evidence is sufficient.
        if assessment.sufficient:
            prompt_package = self.prompt_builder.build_document_qa(
                question=question.strip(),
                context_package=context_package,
            )

            generated_answer = self.llm_generator.generate(prompt_package)

            validation = self.citation_manager.validate(
                answer=generated_answer,
                sources=sources,
            )

            if validation.unknown_citations:
                raise RuntimeError(
                    "Generated answer contains unknown citations: "
                    + ", ".join(validation.unknown_citations)
                )

            if validation.missing_citations:
                raise RuntimeError(
                    "Generated document-grounded answer contains no citations."
                )

            cited_sources = self.citation_manager.filter_valid_sources(
                answer=generated_answer,
                sources=sources,
            )

            document_section = AnswerSection(
                heading=self.DOCUMENT_HEADING,
                content=generated_answer,
                origin=EvidenceOrigin.DOCUMENT,
                citations=validation.found_citations,
            )

            return RAGResponse(
                mode=RequestMode.KNOWLEDGE_AUGMENTED_QA,
                answer=generated_answer,
                sections=[document_section],
                sources=cited_sources,
                insufficient_evidence=False,
                general_knowledge_used=False,
            )

        # Supplement missing or insufficient document evidence with
        # pretrained knowledge, keeping the two sources separate.
        prompt_package = self.prompt_builder.build_knowledge_augmented_qa(
            question=question.strip(),
            context_package=context_package,
        )

        generated_answer = self.llm_generator.generate(prompt_package)

        document_content, general_content = self._split_answer_sections(
            generated_answer
        )

        # Validate both sections independently before constructing the response.
        self._validate_general_section(general_content)

        document_citations, cited_sources = self._validate_document_section(
            document_content=document_content,
            sources=sources,
        )

        sections: list[AnswerSection] = []

        if document_content:
            sections.append(
                AnswerSection(
                    heading=self.DOCUMENT_HEADING,
                    content=document_content,
                    origin=EvidenceOrigin.DOCUMENT,
                    citations=document_citations,
                )
            )

        if not general_content:
            raise RuntimeError(
                "Knowledge-augmented answer does not contain an "
                "Additional general knowledge section."
            )

        # Explicitly disclose when retrieval found no document evidence.
        if not context_package.sources:
            general_content = (
                "No relevant document evidence was found in the "
                "authorized uploaded documents.\n\n"
                + general_content
            )

        sections.append(
            AnswerSection(
                heading=self.GENERAL_HEADING,
                content=general_content,
                origin=EvidenceOrigin.GENERAL_KNOWLEDGE,
                citations=[],
            )
        )

        return RAGResponse(
            mode=RequestMode.KNOWLEDGE_AUGMENTED_QA,
            answer=generated_answer,
            sections=sections,
            sources=cited_sources,
            insufficient_evidence=True,
            general_knowledge_used=True,
        )
