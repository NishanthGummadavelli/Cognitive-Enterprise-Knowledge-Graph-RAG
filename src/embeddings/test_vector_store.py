
"""Test persistent ChromaDB storage and semantic retrieval."""

from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import VectorStore
from src.ingestion.pipeline import DocumentIngestionPipeline


def main() -> None:
    """Ingest a sample document, store its vectors and retrieve a chunk."""

    # Initialize the embedding model and ingestion pipeline.
    encoder = EmbeddingEncoder()
    pipeline = DocumentIngestionPipeline(chunk_size=500)

    # Process the sample enterprise document.
    result = pipeline.ingest("data/raw/sample_policy.txt")

    # Connect to the persistent vector database.
    store = VectorStore(encoder=encoder)

    # Store the document chunks and their embeddings.
    stored_count = store.add_document(result)

    print("=" * 60)
    print("CHROMADB VECTOR STORAGE TEST")
    print("=" * 60)

    print(f"Document: {result.filename}")
    print(f"Document ID: {result.document_id}")
    print(f"Chunks submitted: {stored_count}")
    print(f"Total chunks in database: {store.count()}")

    # Search using a question with different wording.
    query = "How much annual leave can employees take?"

    search_results = store.search(query=query, top_k=3)

    print("\nSEMANTIC SEARCH RESULTS")
    print("-" * 60)

    for rank, item in enumerate(search_results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Distance: {item['distance']:.4f}")
        print(f"Metadata: {item['metadata']}")
        print(f"Text: {item['text']}")

    print("\nVECTOR STORAGE TEST COMPLETED")


if __name__ == "__main__":
    main()
