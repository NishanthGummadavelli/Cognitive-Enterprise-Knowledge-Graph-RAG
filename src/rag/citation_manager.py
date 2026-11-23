
"""Validate and manage citations in generated RAG responses."""

import re
from dataclasses import dataclass

from src.rag.models import SourceReference


@dataclass(frozen=True)
class CitationValidationResult:
    """Represent the outcome of citation validation."""

    valid: bool
    found_citations: list[str]
    unknown_citations: list[str]
    unused_citations: list[str]
    missing_citations: bool


class CitationManager:
    """Validate generated citation references against retrieved sources."""

    CITATION_PATTERN = re.compile(r"\[(S\d+)\]")

    def extract_citations(self, text: str) -> list[str]:
        """Extract unique citation identifiers in their order of appearance."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        return list(dict.fromkeys(self.CITATION_PATTERN.findall(text)))

    def validate(
        self,
        answer: str,
        sources: list[SourceReference],
    ) -> CitationValidationResult:
        """Check whether generated citations reference supplied sources."""

        if not isinstance(answer, str):
            raise TypeError("answer must be a string.")

        if not isinstance(sources, list):
            raise TypeError("sources must be a list.")

        if any(
            not isinstance(source, SourceReference)
            for source in sources
        ):
            raise TypeError(
                "Every source must be a SourceReference instance."
            )

        source_ids = [source.citation_id for source in sources]

        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Source citation IDs must be unique.")

        found_citations = self.extract_citations(answer)

        available_citations = set(source_ids)
        found_set = set(found_citations)

        unknown_citations = [
            citation
            for citation in found_citations
            if citation not in available_citations
        ]

        unused_citations = [
            citation
            for citation in source_ids
            if citation not in found_set
        ]

        return CitationValidationResult(
            valid=not unknown_citations,
            found_citations=found_citations,
            unknown_citations=unknown_citations,
            unused_citations=unused_citations,
            missing_citations=not found_citations,
        )

    def filter_valid_sources(
        self,
        answer: str,
        sources: list[SourceReference],
    ) -> list[SourceReference]:
        """Return only sources referenced by the answer and known to the manager."""

        validation = self.validate(answer, sources)

        if validation.unknown_citations:
            raise ValueError(
                "Cannot filter sources because the answer contains "
                "unknown citation identifiers: "
                + ", ".join(validation.unknown_citations)
            )

        referenced_ids = set(validation.found_citations)

        return [
            source
            for source in sources
            if source.citation_id in referenced_ids
        ]

    def remove_invalid_citations(
        self,
        answer: str,
        sources: list[SourceReference],
    ) -> str:
        """Remove citation markers that do not reference supplied sources."""

        validation = self.validate(answer, sources)

        invalid_ids = set(validation.unknown_citations)

        if not invalid_ids:
            return answer

        def replace_invalid(match: re.Match[str]) -> str:
            """Replace an unknown citation marker with an empty string."""

            citation = match.group(1)

            if citation in invalid_ids:
                return ""

            return match.group(0)

        return self.CITATION_PATTERN.sub(replace_invalid, answer)
