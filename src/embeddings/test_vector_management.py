
"""Test version-aware ChromaDB upsert and deletion operations."""

from pathlib import Path

from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import VectorStore
from src.ingestion.chunker import DocumentChunk
from src.ingestion.pipeline import IngestedChunk, IngestionResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DB_PATH = PROJECT_ROOT / "data" / "chroma_test_management"

TEST_USER_ID = "management_test_user"


def create_test_document(version: int) -> IngestionResult:
    """Create a scoped document fixture for vector-store testing."""

    document_id = "doc_test_management"

    chunk = DocumentChunk(
        chunk_id=f"{document_id}_v{version}_chunk_0000",
        document_id=document_id,
        text=(
            "Employee leave policy explains annual leave "
            f"approval and HR responsibilities. Version {version}."
        ),
        source="sample_policy.txt",
        page_numbers=(),
        heading_path=("Employee Leave Policy",),
        chunk_index=0,
    )

    metadata = {
        "document_id": document_id,
        "filename": "sample_policy.txt",
        "version": version,
        "chunk_id": chunk.chunk_id,
        "chunk_index": 0,
        "predicted_category": "HR",
        "scope_type": "personal",
        "scope_id": TEST_USER_ID,
        "uploaded_by": TEST_USER_ID,
    }

    return IngestionResult(
        document_id=document_id,
        filename="sample_policy.txt",
        file_hash=f"test_file_hash_v{version}",
        content_hash=f"test_content_hash_v{version}",
        predicted_category="HR",
        classification_confidence=0.9,
        total_characters=len(chunk.text),
        total_chunks=1,
        chunks=[
            IngestedChunk(
                chunk=chunk,
                metadata=metadata,
            )
        ],
    )


def main() -> None:
    """Verify insertion, replacement and deletion in isolated storage."""

    encoder = EmbeddingEncoder()

    vector_store = VectorStore(
        encoder=encoder,
        db_path=TEST_DB_PATH,
        collection_name="management_test",
    )

    print("=" * 55)
    print("CHROMADB MANAGEMENT TEST")
    print("=" * 55)

    # Reset only the dedicated test document.
    vector_store.collection.delete(
        where={"document_id": "doc_test_management"}
    )

    # Insert version 1.
    version_one = create_test_document(version=1)
    stored = vector_store.upsert_document(version_one)

    print(f"\nVersion 1 chunks stored: {stored}")
    print(f"Database count: {vector_store.count()}")

    assert stored == 1
    assert vector_store.count() == 1

    # Verify that the inserted chunk is retrievable in its own scope.
    version_one_results = vector_store.search(
        query="employee annual leave approval",
        top_k=5,
        scope_type="personal",
        scope_id=TEST_USER_ID,
    )

    assert len(version_one_results) == 1
    assert (
        version_one_results[0]["metadata"]["scope_id"]
        == TEST_USER_ID
    )

    # Insert version 2 using a separate chunk ID.
    version_two = create_test_document(version=2)
    stored = vector_store.upsert_document(version_two)

    print(f"\nVersion 2 chunks stored: {stored}")
    print(f"Database count: {vector_store.count()}")

    assert stored == 1
    assert vector_store.count() == 2

    # Repeat version 2 to verify that upsert does not duplicate it.
    vector_store.upsert_document(version_two)

    print(f"\nCount after repeating version 2: {vector_store.count()}")
    assert vector_store.count() == 2

    # Delete only version 1.
    deleted = vector_store.delete_chunks(
        [version_one.chunks[0].chunk.chunk_id]
    )

    print(f"\nVersion 1 chunks deleted: {deleted}")
    print(f"Final database count: {vector_store.count()}")

    assert deleted == 1
    assert vector_store.count() == 1

    print("\nALL CHROMADB MANAGEMENT TESTS PASSED")


if __name__ == "__main__":
    main()
