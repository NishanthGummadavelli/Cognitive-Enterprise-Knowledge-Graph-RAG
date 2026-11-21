
"""Test the integrated enterprise document ingestion pipeline."""

from pathlib import Path

from src.ingestion.pipeline import DocumentIngestionPipeline


def main() -> None:
    """Run the ingestion pipeline against a sample document."""

    # Replace this with the path to an actual PDF, DOCX or TXT file.
    sample_path = Path("data/raw/sample_policy.txt")

    pipeline = DocumentIngestionPipeline(chunk_size=500)

    result = pipeline.ingest(sample_path)

    print("=" * 60)
    print("DOCUMENT INGESTION PIPELINE TEST")
    print("=" * 60)

    print(f"Filename: {result.filename}")
    print(f"Document ID: {result.document_id}")
    print(f"SHA-256: {result.file_hash}")
    print(f"File hash: {result.file_hash}")
    print(f"Content hash: {result.content_hash}")
    print(f"Predicted category: {result.predicted_category}")
    print(
        "Classification confidence: "
        f"{result.classification_confidence:.4f}"
    )
    print(f"Total characters: {result.total_characters}")
    print(f"Total chunks: {result.total_chunks}")

    print("\nFIRST CHUNK")
    print("-" * 60)

    first_chunk = result.chunks[0]

    print(f"Chunk ID: {first_chunk.chunk.chunk_id}")
    print(f"Chunk index: {first_chunk.chunk.chunk_index}")
    print(f"Heading path: {first_chunk.chunk.heading_path}")
    print(f"Metadata: {first_chunk.metadata}")
    print(f"\nText:\n{first_chunk.chunk.text}")

    print("\nINGESTION PIPELINE TEST COMPLETED")


if __name__ == "__main__":
    main()
