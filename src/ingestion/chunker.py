
"""Create non-overlapping, metadata-rich chunks from document elements."""

from dataclasses import dataclass
import re

from src.ingestion.loaders import DocumentElement


@dataclass
class DocumentChunk:
    """Represent a canonical text chunk and its original document context."""

    chunk_id: str
    document_id: str
    text: str
    source: str
    page_numbers: tuple[int, ...]
    heading_path: tuple[str, ...]
    chunk_index: int
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    """Split long text into non-overlapping pieces at safe text boundaries.

    Args:
        text: Paragraph text to split.
        chunk_size: Maximum target size of each piece, in characters.

    Returns:
        Non-empty pieces that preserve the original text content in order.
    """

    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Prefer a sentence boundary in the latter half of the chunk.
            sentence_matches = list(
                re.finditer(r"[.!?](?:\s+|$)", text[start:end])
            )

            if sentence_matches:
                boundary = start + sentence_matches[-1].end()

                if boundary > start + chunk_size // 2:
                    end = boundary
            else:
                # If no suitable sentence boundary exists, use a word boundary.
                boundary = text.rfind(" ", start, end)

                if boundary > start + chunk_size // 2:
                    end = boundary

        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        # Move directly to the end of this piece; no text is overlapped.
        start = end

        # Skip whitespace at the start of the next piece.
        while start < len(text) and text[start].isspace():
            start += 1

    return pieces


def chunk_document(
    elements: list[DocumentElement],
    document_id: str,
    chunk_size: int = 500,
) -> list[DocumentChunk]:
    """Convert structured document elements into canonical text chunks.

    Related paragraphs are grouped together. Headings and source-file
    boundaries separate chunks. Long paragraphs are split without overlap.

    Args:
        elements: Structured elements returned by the document loader.
        document_id: Stable ID assigned to the original document.
        chunk_size: Maximum target chunk size, measured in characters.

    Returns:
        Document chunks with stable IDs, source metadata and neighbor links.

    Raises:
        ValueError: If the document ID or chunk size is invalid.
    """

    # Validate the document identifier and chunk configuration.
    if not document_id.strip():
        raise ValueError("document_id cannot be empty")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    chunks: list[DocumentChunk] = []
    heading_path: list[str] = []

    current_text: list[str] = []
    current_pages: set[int] = set()
    current_source = ""

    def flush_chunk() -> None:
        """Save the current paragraph group as a canonical document chunk."""

        nonlocal current_text, current_pages, current_source

        combined_text = "\n\n".join(current_text).strip()

        if not combined_text:
            return

        chunk_index = len(chunks)

        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                document_id=document_id,
                text=combined_text,
                source=current_source,
                page_numbers=tuple(sorted(current_pages)),
                heading_path=tuple(heading_path),
                chunk_index=chunk_index,
            )
        )

        # Reset the buffer before collecting the next chunk.
        current_text = []
        current_pages = set()

    for element in elements:
        # A heading starts a new logical section.
        if element.element_type == "heading":
            flush_chunk()

            level = element.heading_level or 1

            # Retain parent headings and replace headings at this level or deeper.
            heading_path = heading_path[: level - 1]
            heading_path.append(element.text.strip())
            continue

        text = element.text.strip()

        if not text:
            continue

        # Preserve page information for later source citations.
        page_numbers = (
            {element.page_number}
            if element.page_number is not None
            else set()
        )

        # Split oversized paragraphs without duplicating their content.
        pieces = (
            _split_long_text(text, chunk_size)
            if len(text) > chunk_size
            else [text]
        )

        for piece in pieces:
            # Do not combine content from different source files.
            if current_text and current_source != element.source:
                flush_chunk()

            proposed_length = len("\n\n".join(current_text + [piece]))

            # Start a new chunk if the next paragraph would exceed the target.
            if current_text and proposed_length > chunk_size:
                flush_chunk()

            if not current_text:
                current_source = element.source

            current_text.append(piece)
            current_pages.update(page_numbers)

            # Keep pieces from an oversized paragraph as separate chunks.
            if len(pieces) > 1:
                flush_chunk()

    # Save any remaining content after processing all elements.
    flush_chunk()

    # Link immediate neighbors only when they share the same source and section.
    for index, chunk in enumerate(chunks):
        if index > 0:
            previous_chunk = chunks[index - 1]

            if (
                previous_chunk.source == chunk.source
                and previous_chunk.heading_path == chunk.heading_path
            ):
                chunk.previous_chunk_id = previous_chunk.chunk_id
                previous_chunk.next_chunk_id = chunk.chunk_id

    return chunks

