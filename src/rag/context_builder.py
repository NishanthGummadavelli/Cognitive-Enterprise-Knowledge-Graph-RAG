
"""Build scope-safe, citation-aware context from hybrid retrieval results."""

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextSource:
    """Represent one retrieved evidence chunk and its citation information."""

    citation_id: str
    chunk_id: str
    document_id: str
    source: str
    page_numbers: list[int]
    heading_path: list[str]
    text: str
    retrieval_score: float | None


@dataclass(frozen=True)
class ContextPackage:
    """Contain the formatted LLM context and its associated source records."""

    context_text: str
    sources: list[ContextSource]
    scope_type: str
    scope_id: str
    included_chunks: int
    omitted_chunk_ids: list[str]


class ContextBuilder:
    """Convert hybrid retrieval results into grounded, citation-aware context."""

    VALID_SCOPE_TYPES = {"personal", "project"}

    def __init__(self, max_context_chars: int = 12000) -> None:
        """Initialize the builder with a maximum context size."""

        if isinstance(max_context_chars, bool) or not isinstance(
            max_context_chars, int
        ):
            raise TypeError("max_context_chars must be an integer.")

        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero.")

        self.max_context_chars = max_context_chars

    @staticmethod
    def _parse_list(value: Any) -> list[Any]:
        """Convert list-like metadata, including JSON strings, into a list."""

        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return []

            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [value]

            if isinstance(parsed, list):
                return parsed

            return [parsed]

        return [value]

    @staticmethod
    def _format_location(
        source: str,
        page_numbers: list[int],
        heading_path: list[str],
    ) -> str:
        """Create a readable source-location description."""

        parts: list[str] = []

        if source:
            parts.append(f"Document: {source}")

        if page_numbers:
            pages = ", ".join(str(page) for page in page_numbers)
            parts.append(f"Pages: {pages}")

        if heading_path:
            heading = " > ".join(heading_path)
            parts.append(f"Section: {heading}")

        return " | ".join(parts)

    def _build_source(
        self,
        result: dict[str, Any],
        citation_number: int,
    ) -> ContextSource:
        """Convert one validated retrieval result into a citation source."""

        metadata = result.get("metadata") or {}

        if not isinstance(metadata, dict):
            raise ValueError("Retrieval result metadata must be a dictionary.")

        chunk_id = result.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Every retrieval result must contain a chunk_id.")

        text = result.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Retrieved chunk '{chunk_id}' has no usable text."
            )

        document_id = (
            result.get("document_id")
            or metadata.get("document_id")
            or "unknown"
        )

        source = (
            result.get("source")
            or metadata.get("source")
            or metadata.get("filename")
            or "Unknown source"
        )

        page_values = self._parse_list(metadata.get("page_numbers"))
        heading_values = self._parse_list(metadata.get("heading_path"))

        page_numbers: list[int] = []

        for page in page_values:
            try:
                page_number = int(page)
            except (TypeError, ValueError):
                continue

            if page_number not in page_numbers:
                page_numbers.append(page_number)

        heading_path = [
            str(item)
            for item in heading_values
            if item is not None and str(item).strip()
        ]

        score = result.get("rrf_score")

        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None

        return ContextSource(
            citation_id=f"S{citation_number}",
            chunk_id=chunk_id,
            document_id=str(document_id),
            source=str(source),
            page_numbers=page_numbers,
            heading_path=heading_path,
            text=text.strip(),
            retrieval_score=score,
        )

    def build(
        self,
        retrieval_results: list[dict[str, Any]],
        scope_type: str,
        scope_id: str,
    ) -> ContextPackage:
        """Build context from retrieval results belonging to one scope."""

        if scope_type not in self.VALID_SCOPE_TYPES:
            raise ValueError(
                "scope_type must be either 'personal' or 'project'."
            )

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("scope_id must be a non-empty string.")

        if not isinstance(retrieval_results, list):
            raise TypeError("retrieval_results must be a list.")

        sources: list[ContextSource] = []
        omitted_chunk_ids: list[str] = []
        seen_chunk_ids: set[str] = set()
        context_blocks: list[str] = []
        current_length = 0

        for result in retrieval_results:
            if not isinstance(result, dict):
                raise TypeError("Every retrieval result must be a dictionary.")

            result_scope_type = result.get("scope_type")
            result_scope_id = result.get("scope_id")

            # Fail closed: never allow missing or mismatched scope metadata.
            if (
                result_scope_type != scope_type
                or result_scope_id != scope_id
            ):
                raise PermissionError(
                    "A retrieval result does not belong to the requested "
                    "knowledge scope."
                )

            chunk_id = result.get("chunk_id")

            if isinstance(chunk_id, str) and chunk_id in seen_chunk_ids:
                continue

            if isinstance(chunk_id, str):
                seen_chunk_ids.add(chunk_id)

            source = self._build_source(result, len(sources) + 1)

            location = self._format_location(
                source.source,
                source.page_numbers,
                source.heading_path,
            )

            block = (
                f"[{source.citation_id}]\n"
                f"Chunk ID: {source.chunk_id}\n"
                f"Document ID: {source.document_id}\n"
                f"{location}\n"
                f"Evidence:\n{source.text}"
            )

            additional_length = len(block)

            if context_blocks:
                additional_length += 2

            if current_length + additional_length > self.max_context_chars:
                omitted_chunk_ids.append(source.chunk_id)
                continue

            context_blocks.append(block)
            sources.append(source)
            current_length += additional_length

        return ContextPackage(
            context_text="\n\n".join(context_blocks),
            sources=sources,
            scope_type=scope_type,
            scope_id=scope_id,
            included_chunks=len(sources),
            omitted_chunk_ids=omitted_chunk_ids,
        )
