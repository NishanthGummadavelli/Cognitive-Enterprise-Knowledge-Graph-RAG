
"""Coordinate document loading, cleaning, classification and chunking."""

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.classifier import DocumentClassifier
from src.ingestion.cleaner import clean_text
from src.ingestion.chunker import DocumentChunk, chunk_document
from src.ingestion.loaders import load_document


@dataclass
class IngestedChunk:
    """Represent a document chunk with inherited document metadata."""

    chunk: DocumentChunk
    metadata: dict[str, object]



@dataclass
class IngestionResult:
    """Represent the complete output of processing one document."""

    document_id: str
    filename: str
    file_hash: str
    content_hash: str
    predicted_category: str
    classification_confidence: float
    total_characters: int
    total_chunks: int
    chunks: list[IngestedChunk]



class DocumentIngestionPipeline:
    """Process enterprise documents into classified, structured chunks."""

    def __init__(
        self,
        classifier: DocumentClassifier | None = None,
        chunk_size: int = 500,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            classifier: An existing classifier instance, if available.
            chunk_size: Target maximum chunk size in characters.

        Raises:
            ValueError: If the chunk size is not positive.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        # Load the trained classifier once for reuse across documents.
        self.classifier = classifier or DocumentClassifier()
        self.chunk_size = chunk_size

    def ingest(self, file_path: str | Path) -> IngestionResult:
        """Load, clean, classify and chunk one supported document.

        Args:
            file_path: Path to the PDF, DOCX or TXT document.

        Returns:
            A structured result containing document and chunk metadata.

        Raises:
            FileNotFoundError: If the input document does not exist.
            ValueError: If the document is unsupported or contains no usable text.
        """

        path = Path(file_path)

        # Validate the input before attempting document extraction.
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        # Extract structured elements using the existing document loaders.
        elements = load_document(str(path))

        if not elements:
            raise ValueError(
                f"No readable text was extracted from: {path.name}"
            )

        # Combine headings and paragraphs for document-level classification.
        full_text = clean_text(
            "\n\n".join(element.text for element in elements)
        )

        if not full_text:
            raise ValueError(
                f"The document contains no usable text: {path.name}"
            )

        
        # Calculate the hash of the original uploaded file.
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        # Calculate a separate hash of the normalized extracted content.
        content_hash = hashlib.sha256(
            full_text.encode("utf-8")
        ).hexdigest()

        # Generate an independent identifier for this document record.
        document_id = f"doc_{uuid.uuid4().hex}"


        # Predict one category for the complete document.
        classification = self.classifier.predict(full_text)

        predicted_category = str(
            classification["predicted_category"]
        )
        confidence = float(classification["confidence"])

        # Create canonical non-overlapping chunks using the existing chunker.
        document_chunks = chunk_document(
            elements=elements,
            document_id=document_id,
            chunk_size=self.chunk_size,
        )

        if not document_chunks:
            raise ValueError(
                f"No content chunks could be created for: {path.name}"
            )

        # Prepare metadata shared by the document and all its chunks.
        document_metadata = {
            "document_id": document_id,
            "filename": path.name,
            "file_hash": file_hash,
            "content_hash": content_hash,
            "predicted_category": predicted_category,
            "classification_confidence": confidence,
        }

        # Attach document-level metadata to every generated chunk.
        ingested_chunks = [
            IngestedChunk(
                chunk=chunk,
                metadata={
                    **document_metadata,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "page_numbers": list(chunk.page_numbers),
                    "heading_path": list(chunk.heading_path),
                },
            )
            for chunk in document_chunks
        ]

        # Return the complete structured ingestion result.
        return IngestionResult(
            document_id=document_id,
            filename=path.name,
            file_hash=file_hash,
            content_hash = content_hash,
            predicted_category=predicted_category,
            classification_confidence=confidence,
            total_characters=len(full_text),
            total_chunks=len(ingested_chunks),
            chunks=ingested_chunks,
        )
