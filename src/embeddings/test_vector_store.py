
"""Test persistent ChromaDB storage and scope-aware semantic retrieval."""

from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import VectorStore
from src.ingestion.pipeline import DocumentIngestionPipeline


def main() -> None:
    """Ingest a sample document, store its vectors and retrieve a scoped chunk."""

    encoder = EmbeddingEncoder()
    pipeline = DocumentIngestionPipeline(chunk_size=500)

    result = pipeline.ingest("data/raw/sample_policy.txt")

    # This standalone test uses a dedicated personal test identity.
    # The production document service supplies real scope metadata.
    test_user_id = "vector_store_test_user"

    for item in result.chunks:
        item.metadata.update(
            {
                "scope_type": "personal",
                "scope_id": test_user_id,
                "uploaded_by": test_user_id,
            }
        )

    store = VectorStore(encoder=encoder)

    stored_count = store.add_document(result)

    print("=" * 60)
    print("CHROMADB SCOPE-AWARE VECTOR STORAGE TEST")
    print("=" * 60)

    print(f"Document: {result.filename}")
    print(f"Document ID: {result.document_id}")
    print(f"Chunks submitted: {stored_count}")
    print(f"Total chunks in database: {store.count()}")

    query = "How much annual leave can employees take?"

    search_results = store.search(
        query=query,
        top_k=3,
        scope_type="personal",
        scope_id=test_user_id,
    )

    print("\nSCOPE-AWARE SEMANTIC SEARCH RESULTS")
    print("-" * 60)

    for rank, item in enumerate(search_results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Distance: {item['distance']:.4f}")
        print(f"Scope: {item['scope_type']} / {item['scope_id']}")
        print(f"Metadata: {item['metadata']}")
        print(f"Text: {item['text']}")

        assert item["metadata"]["scope_type"] == "personal"
        assert item["metadata"]["scope_id"] == test_user_id

    print("\nVECTOR STORAGE AND SCOPED SEARCH TEST COMPLETED")


if __name__ == "__main__":
    main()
