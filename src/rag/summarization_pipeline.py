
"""Implement document-grounded, hierarchical document summarization."""

import re
from dataclasses import replace
from typing import Any, Callable, Sequence

from src.rag.citation_manager import CitationManager
from src.rag.context_builder import (
    ContextBuilder,
    ContextPackage,
)
from src.rag.llm_generator import LLMGenerator
from src.rag.models import (
    AnswerSection,
    EvidenceOrigin,
    RAGResponse,
    RequestMode,
    ScopeType,
    SourceReference,
    SummaryStyle,
)
from src.rag.prompt_builder import PromptBuilder


DocumentChunkLoader = Callable[
    [list[str], str, str],
    list[dict[str, Any]],
]


class SummarizationPipeline:
    """Summarize selected documents using hierarchical RAG generation."""

    def __init__(
        self,
        document_chunk_loader: DocumentChunkLoader,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_generator: LLMGenerator | None = None,
        citation_manager: CitationManager | None = None,
        max_batch_chars: int = 10000,
    ) -> None:
        """Initialize the pipeline and its injectable dependencies."""

        if not callable(document_chunk_loader):
            raise TypeError("document_chunk_loader must be callable.")

        if (
            isinstance(max_batch_chars, bool)
            or not isinstance(max_batch_chars, int)
            or max_batch_chars <= 0
        ):
            raise ValueError("max_batch_chars must be a positive integer.")

        self.document_chunk_loader = document_chunk_loader
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_generator = llm_generator or LLMGenerator()
        self.citation_manager = citation_manager or CitationManager()
        self.max_batch_chars = max_batch_chars

    @staticmethod
    def _validate_inputs(
        document_ids: Sequence[str],
        scope_type: ScopeType,
        scope_id: str,
        summary_style: SummaryStyle,
    ) -> list[str]:
        """Validate selected documents, scope, and summary style."""

        if (
            isinstance(document_ids, (str, bytes))
            or not isinstance(document_ids, (list, tuple))
            or not document_ids
        ):
            raise ValueError("At least one document ID must be provided.")

        normalized_ids = []

        for document_id in document_ids:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("Document IDs must be non-empty strings.")

            normalized_ids.append(document_id.strip())

        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("Duplicate document IDs are not allowed.")

        if not isinstance(scope_type, ScopeType):
            raise TypeError("scope_type must be a ScopeType instance.")

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("scope_id must be a non-empty string.")

        if not isinstance(summary_style, SummaryStyle):
            raise TypeError("summary_style must be a SummaryStyle instance.")

        return normalized_ids

    @staticmethod
    def _get_document_id(result: dict[str, Any]) -> str | None:
        """Extract the document ID from a retrieved chunk."""

        metadata = result.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError("Chunk metadata must be a dictionary.")

        return (
            result.get("document_id")
            or metadata.get("document_id")
        )

    @classmethod
    def _validate_chunks(
        cls,
        chunks: list[dict[str, Any]],
        document_ids: list[str],
        scope_type: ScopeType,
        scope_id: str,
    ) -> list[dict[str, Any]]:
        """Reject chunks outside the requested documents or scope."""

        if not isinstance(chunks, list):
            raise TypeError("Document chunk loader must return a list.")

        selected_ids = set(document_ids)
        found_document_ids = set()
        seen_chunk_ids = set()
        validated_chunks = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise TypeError("Every document chunk must be a dictionary.")

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            metadata = chunk.get("metadata")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Every chunk must have a valid chunk_id.")

            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk ID detected: {chunk_id}")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Chunk {chunk_id} has empty text.")

            if not isinstance(metadata, dict):
                raise ValueError(f"Chunk {chunk_id} has invalid metadata.")

            document_id = cls._get_document_id(chunk)

            if document_id not in selected_ids:
                raise ValueError(
                    f"Chunk {chunk_id} does not belong to a selected document."
                )

            if chunk.get("scope_type") != scope_type.value:
                raise ValueError(
                    f"Chunk {chunk_id} has an unexpected scope type."
                )

            if chunk.get("scope_id") != scope_id:
                raise ValueError(
                    f"Chunk {chunk_id} has an unexpected scope ID."
                )

            # Ensure metadata cannot contradict the requested scope.
            if (
                metadata.get("scope_type") != scope_type.value
                or metadata.get("scope_id") != scope_id
            ):
                raise ValueError(
                    f"Chunk {chunk_id} has inconsistent scope metadata."
                )

            seen_chunk_ids.add(chunk_id)
            found_document_ids.add(document_id)
            validated_chunks.append(chunk)

        missing_documents = selected_ids - found_document_ids

        if missing_documents:
            raise ValueError(
                "No chunks were found for selected document(s): "
                + ", ".join(sorted(missing_documents))
            )

        # Preserve document order and original chunk order.
        validated_chunks.sort(
            key=lambda item: (
                cls._get_document_id(item),
                int(item["metadata"].get("chunk_index", 0)),
            )
        )

        return validated_chunks

    def _create_batches(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Split chunks into batches that fit the configured character limit."""

        batches = []
        current_batch = []
        current_chars = 0

        for chunk in chunks:
            chunk_chars = len(chunk["text"])

            if chunk_chars > self.max_batch_chars:
                raise ValueError(
                    f"Chunk {chunk['chunk_id']} exceeds the configured "
                    "batch character limit. Rechunk the document first."
                )

            if (
                current_batch
                and current_chars + chunk_chars > self.max_batch_chars
            ):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(chunk)
            current_chars += chunk_chars

        if current_batch:
            batches.append(current_batch)

        return batches

    def _build_batch_context(
        self,
        chunks: list[dict[str, Any]],
        scope_type: ScopeType,
        scope_id: str,
        citation_offset: int,
    ) -> ContextPackage:
        """Build a batch context and assign globally unique citation IDs."""

        context_package = self.context_builder.build(
            retrieval_results=chunks,
            scope_type=scope_type.value,
            scope_id=scope_id,
        )

        if context_package.omitted_chunk_ids:
            raise RuntimeError(
                "ContextBuilder omitted chunks during summarization: "
                + ", ".join(context_package.omitted_chunk_ids)
            )

        if len(context_package.sources) != len(chunks):
            raise RuntimeError(
                "ContextBuilder did not preserve every selected chunk."
            )

        citation_mapping = {
            source.citation_id: f"S{citation_offset + index + 1}"
            for index, source in enumerate(context_package.sources)
        }

        updated_context = context_package.context_text

        # Replace longer citation identifiers first.
        for old_id, new_id in citation_mapping.items():
            updated_context = re.sub(
                rf"\[{re.escape(old_id)}\]",
                f"[{new_id}]",
                updated_context,
            )

        updated_sources = [
            replace(
                source,
                citation_id=citation_mapping[source.citation_id],
            )
            for source in context_package.sources
        ]

        return replace(
            context_package,
            context_text=updated_context,
            sources=updated_sources,
        )

    @staticmethod
    def _convert_sources(
        context_sources,
    ) -> list[SourceReference]:
        """Convert context sources into final response source references."""

        return [
            SourceReference(
                citation_id=source.citation_id,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                source=source.source,
                page_numbers=source.page_numbers,
                heading_path=source.heading_path,
            )
            for source in context_sources
        ]

    def summarize(
        self,
        document_ids: list[str],
        scope_type: ScopeType,
        scope_id: str,
        summary_style: SummaryStyle = SummaryStyle.COMPREHENSIVE,
    ) -> RAGResponse:
        """Generate a cited summary of all chunks in the selected documents."""

        document_ids = self._validate_inputs(
            document_ids=document_ids,
            scope_type=scope_type,
            scope_id=scope_id,
            summary_style=summary_style,
        )

        # Load every chunk, not just the most relevant chunks.
        retrieved_chunks = self.document_chunk_loader(
            document_ids,
            scope_type.value,
            scope_id,
        )

        chunks = self._validate_chunks(
            chunks=retrieved_chunks,
            document_ids=document_ids,
            scope_type=scope_type,
            scope_id=scope_id,
        )

        batches = self._create_batches(chunks)

        batch_summaries = []
        all_context_sources = []
        citation_offset = 0

        for batch in batches:
            # Build context with globally unique source identifiers.
            context_package = self._build_batch_context(
                chunks=batch,
                scope_type=scope_type,
                scope_id=scope_id,
                citation_offset=citation_offset,
            )

            batch_sources = self._convert_sources(
                context_package.sources
            )

            # Intermediate summaries preserve detail using comprehensive style.
            prompt_package = self.prompt_builder.build_document_summary(
                context_package=context_package,
                summary_style=(
                    summary_style
                    if len(batches) == 1
                    else SummaryStyle.COMPREHENSIVE
                    ),
            )

            batch_summary = self.llm_generator.generate(prompt_package)

            validation = self.citation_manager.validate(
                answer=batch_summary,
                sources=batch_sources,
            )

            if validation.unknown_citations:
                raise RuntimeError(
                    "Intermediate summary contains unknown citations: "
                    + ", ".join(validation.unknown_citations)
                )

            if validation.missing_citations:
                raise RuntimeError(
                    "Intermediate summary contains no source citations."
                )

            batch_summaries.append(batch_summary)
            all_context_sources.extend(context_package.sources)
            citation_offset += len(context_package.sources)

        all_sources = self._convert_sources(all_context_sources)

        # A single batch already represents the final summary.
        if len(batch_summaries) == 1:
            final_summary = batch_summaries[0]

        else:
            # Combine intermediate summaries while retaining original citations.
            combined_context = "\n\n".join(
                f"Intermediate summary {index + 1}:\n{summary}"
                for index, summary in enumerate(batch_summaries)
            )

            final_context = ContextPackage(
                context_text=combined_context,
                sources=all_context_sources,
                scope_type=scope_type.value,
                scope_id=scope_id,
                included_chunks=len(chunks),
                omitted_chunk_ids=[],
            )

            final_prompt = self.prompt_builder.build_document_summary(
                context_package=final_context,
                summary_style=summary_style,
            )

            final_summary = self.llm_generator.generate(final_prompt)

        final_validation = self.citation_manager.validate(
            answer=final_summary,
            sources=all_sources,
        )

        if final_validation.unknown_citations:
            raise RuntimeError(
                "Final summary contains unknown citations: "
                + ", ".join(final_validation.unknown_citations)
            )

        if final_validation.missing_citations:
            raise RuntimeError(
                "Final summary contains no source citations."
            )

        cited_sources = self.citation_manager.filter_valid_sources(
            answer=final_summary,
            sources=all_sources,
        )

        section = AnswerSection(
            heading=f"{summary_style.value.replace('_', ' ').title()} Summary",
            content=final_summary,
            origin=EvidenceOrigin.DOCUMENT,
            citations=final_validation.found_citations,
        )

        return RAGResponse(
            mode=RequestMode.DOCUMENT_SUMMARY,
            answer=final_summary,
            sections=[section],
            sources=cited_sources,
            insufficient_evidence=False,
            general_knowledge_used=False,
        )
