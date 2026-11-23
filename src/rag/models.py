
"""Define shared request, response, and source models for the RAG system."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequestMode(str, Enum):
    """Represent the supported user-facing capabilities."""

    DOCUMENT_QA = "document_qa"
    DOCUMENT_SUMMARY = "document_summary"
    KNOWLEDGE_AUGMENTED_QA = "knowledge_augmented_qa"


class ScopeType(str, Enum):
    """Represent the supported knowledge-space types."""

    PERSONAL = "personal"
    PROJECT = "project"


class SummaryStyle(str, Enum):
    """Represent the available document summarization styles."""

    COMPREHENSIVE = "comprehensive"
    EXECUTIVE = "executive"
    SECTION_WISE = "section_wise"
    KEY_POINTS = "key_points"
    ACTION_ITEMS = "action_items"


class EvidenceOrigin(str, Enum):
    """Identify where an answer section obtains its information."""

    DOCUMENT = "document"
    GENERAL_KNOWLEDGE = "general_knowledge"


class SourceReference(BaseModel):
    """Represent a retrieved document source and its citation metadata."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page_numbers: list[int] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)


class AnswerSection(BaseModel):
    """Represent an answer section and identify its information origin."""

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1)
    content: str = Field(min_length=1)
    origin: EvidenceOrigin
    citations: list[str] = Field(default_factory=list)


class RAGRequest(BaseModel):
    """Represent a request for any supported RAG system capability."""

    model_config = ConfigDict(extra="forbid")

    mode: RequestMode

    scope_type: ScopeType
    scope_id: str = Field(min_length=1)

    question: str | None = None

    document_ids: list[str] | None = None

    summary_style: SummaryStyle | None = None

    top_k: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def validate_request_mode(self) -> "RAGRequest":
        """Validate fields according to the selected request mode."""

        if self.mode in {
            RequestMode.DOCUMENT_QA,
            RequestMode.KNOWLEDGE_AUGMENTED_QA,
        }:
            if not self.question or not self.question.strip():
                raise ValueError(
                    "A non-empty question is required for question-answering."
                )

            if self.summary_style is not None:
                raise ValueError(
                    "summary_style is only supported for document summaries."
                )

        if self.mode == RequestMode.DOCUMENT_SUMMARY:
            if not self.document_ids:
                raise ValueError(
                    "At least one document_id is required for summarization."
                )

            if any(
                not isinstance(document_id, str) or not document_id.strip()
                for document_id in self.document_ids
            ):
                raise ValueError(
                    "Every document_id must be a non-empty string."
                )

            if len(self.document_ids) != len(set(self.document_ids)):
                raise ValueError(
                    "document_ids must not contain duplicates."
                )

            if self.question and not self.question.strip():
                raise ValueError(
                    "If provided, the summary question must not be empty."
                )

            if self.summary_style is None:
                self.summary_style = SummaryStyle.COMPREHENSIVE

        return self


class RAGResponse(BaseModel):
    """Represent a generated answer with provenance and citation metadata."""

    model_config = ConfigDict(extra="forbid")

    mode: RequestMode
    answer: str = Field(min_length=1)

    sections: list[AnswerSection] = Field(default_factory=list)

    sources: list[SourceReference] = Field(default_factory=list)

    insufficient_evidence: bool = False

    general_knowledge_used: bool = False

    @model_validator(mode="after")
    def validate_response_sources(self) -> "RAGResponse":
        """Ensure every document citation references a supplied source."""

        available_citations = {
            source.citation_id for source in self.sources
        }

        for section in self.sections:
            if section.origin == EvidenceOrigin.GENERAL_KNOWLEDGE:
                if section.citations:
                    raise ValueError(
                        "General-knowledge sections must not claim "
                        "document citations."
                    )

            for citation in section.citations:
                if citation not in available_citations:
                    raise ValueError(
                        f"Unknown citation reference: {citation}"
                    )

        has_general_knowledge = any(
            section.origin == EvidenceOrigin.GENERAL_KNOWLEDGE
            for section in self.sections
        )

        if has_general_knowledge and not self.general_knowledge_used:
            raise ValueError(
                "general_knowledge_used must be True when a response "
                "contains general-knowledge sections."
            )

        if (
            self.mode == RequestMode.DOCUMENT_QA
            and self.general_knowledge_used
        ):
            raise ValueError(
                "Document-grounded Q&A cannot use general knowledge."
            )

        return self
