
"""Test strict personal and project isolation in ChromaDB retrieval."""

from pathlib import Path

import numpy as np

from src.embeddings.vector_store import VectorStore
from src.ingestion.chunker import DocumentChunk
from src.ingestion.pipeline import IngestedChunk, IngestionResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DB_PATH = PROJECT_ROOT / "data" / "chroma_test_scoped_search"

TEST_COLLECTION = "scope_isolation_test"


class FakeEmbeddingEncoder:
    """Provide deterministic embeddings for fast retrieval tests."""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Return a fixed two-dimensional vector for each document chunk."""

        return np.asarray(
            [[1.0, 0.0] for _ in texts],
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Return the same fixed vector for every test query."""

        return np.asarray([1.0, 0.0], dtype=np.float32)


def create_test_document(
    document_id: str,
    scope_type: str,
    scope_id: str,
    uploaded_by: str,
    text: str,
) -> IngestionResult:
    """Create a one-chunk document fixture with explicit scope metadata."""

    chunk = DocumentChunk(
        chunk_id=f"{document_id}_chunk_0000",
        document_id=document_id,
        text=text,
        source=f"{document_id}.txt",
        page_numbers=(1,),
        heading_path=("Test Document",),
        chunk_index=0,
    )

    metadata = {
        "document_id": document_id,
        "filename": f"{document_id}.txt",
        "version": 1,
        "chunk_id": chunk.chunk_id,
        "chunk_index": 0,
        "predicted_category": "General & Administration",
        "scope_type": scope_type,
        "scope_id": scope_id,
        "uploaded_by": uploaded_by,
    }

    return IngestionResult(
        document_id=document_id,
        filename=f"{document_id}.txt",
        file_hash=f"file_hash_{document_id}",
        content_hash=f"content_hash_{document_id}",
        predicted_category="General & Administration",
        classification_confidence=0.9,
        total_characters=len(text),
        total_chunks=1,
        chunks=[
            IngestedChunk(
                chunk=chunk,
                metadata=metadata,
            )
        ],
    )


def main() -> None:
    """Verify that vector retrieval never crosses knowledge-space boundaries."""

    store = VectorStore(
        encoder=FakeEmbeddingEncoder(),
        db_path=TEST_DB_PATH,
        collection_name=TEST_COLLECTION,
    )

    # Use fixed IDs and remove only these test records before execution.
    test_chunk_ids = [
        "personal_user_a_chunk_0000",
        "personal_user_b_chunk_0000",
        "project_alpha_chunk_0000",
    ]

    store.collection.delete(ids=test_chunk_ids)

    documents = [
        create_test_document(
            document_id="personal_user_a",
            scope_type="personal",
            scope_id="user_a",
            uploaded_by="user_a",
            text="User A knows about cloud migration and AWS.",
        ),
        create_test_document(
            document_id="personal_user_b",
            scope_type="personal",
            scope_id="user_b",
            uploaded_by="user_b",
            text="User B knows about database administration.",
        ),
        create_test_document(
            document_id="project_alpha",
            scope_type="project",
            scope_id="project_alpha",
            uploaded_by="user_a",
            text="Project Alpha contains shared cloud architecture knowledge.",
        ),
    ]

    print("=" * 60)
    print("SCOPE-AWARE VECTOR RETRIEVAL TEST")
    print("=" * 60)

    for document in documents:
        stored = store.upsert_document(document)
        assert stored == 1

    print("\nThree documents stored in separate scopes.")

    # Test personal scope A.
    user_a_results = store.search(
        query="cloud architecture",
        top_k=5,
        scope_type="personal",
        scope_id="user_a",
    )

    assert len(user_a_results) == 1
    assert user_a_results[0]["metadata"]["document_id"] == "personal_user_a"
    assert user_a_results[0]["metadata"]["scope_id"] == "user_a"

    print("TEST 1 PASSED: User A retrieves only personal documents.")

    # Test personal scope B.
    user_b_results = store.search(
        query="cloud architecture",
        top_k=5,
        scope_type="personal",
        scope_id="user_b",
    )

    assert len(user_b_results) == 1
    assert user_b_results[0]["metadata"]["document_id"] == "personal_user_b"
    assert user_b_results[0]["metadata"]["scope_id"] == "user_b"

    print("TEST 2 PASSED: User B retrieves only personal documents.")

    # Test project scope.
    project_results = store.search(
        query="cloud architecture",
        top_k=5,
        scope_type="project",
        scope_id="project_alpha",
    )

    assert len(project_results) == 1
    assert project_results[0]["metadata"]["document_id"] == "project_alpha"
    assert project_results[0]["metadata"]["scope_type"] == "project"

    print("TEST 3 PASSED: Project Alpha retrieves only project documents.")

    # Test that another user's scope does not reveal User A's document.
    user_c_results = store.search(
        query="cloud architecture",
        top_k=5,
        scope_type="personal",
        scope_id="user_c",
    )

    assert user_c_results == []

    print("TEST 4 PASSED: Unknown personal scope returns no documents.")

    # Test that an unscoped retrieval request is rejected.
    try:
        store.search(
            query="cloud architecture",
            top_k=5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Unscoped retrieval should be rejected.")

    print("TEST 5 PASSED: Unscoped retrieval is rejected.")

    # Test that an invalid scope type is rejected.
    try:
        store.search(
            query="cloud architecture",
            top_k=5,
            scope_type="legacy",
            scope_id="legacy_unassigned",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid scope type should be rejected.")

    print("TEST 6 PASSED: Invalid scope type is rejected.")

    # Test that personal scope identity is enforced during storage.
    invalid_document = create_test_document(
        document_id="invalid_personal_scope",
        scope_type="personal",
        scope_id="user_a",
        uploaded_by="user_b",
        text="This document has inconsistent personal scope metadata.",
    )

    try:
        store.upsert_document(invalid_document)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Inconsistent personal scope metadata should be rejected."
        )

    print("TEST 7 PASSED: Invalid personal scope metadata is rejected.")

    # Remove only the records created by this test.
    deleted = store.delete_chunks(test_chunk_ids)
    assert deleted == 3

    print("\nTest records cleaned up.")
    print("\nALL SCOPE-AWARE VECTOR RETRIEVAL TESTS PASSED")


if __name__ == "__main__":
    main()
