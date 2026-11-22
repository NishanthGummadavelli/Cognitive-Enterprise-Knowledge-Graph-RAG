
"""Test scope-aware BM25 keyword retrieval."""

from pathlib import Path

import numpy as np

from src.embeddings.vector_store import VectorStore
from src.ingestion.chunker import DocumentChunk
from src.ingestion.pipeline import IngestedChunk, IngestionResult
from src.retrieval.keyword_retriever import KeywordRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DB_PATH = PROJECT_ROOT / "data" / "chroma_test_keyword"

TEST_COLLECTION = "keyword_scope_test"


class FakeEmbeddingEncoder:
    """Provide deterministic embeddings without loading BGE."""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Return fixed vectors for test documents."""

        return np.asarray(
            [[1.0, 0.0] for _ in texts],
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Return a fixed vector for test queries."""

        return np.asarray([1.0, 0.0], dtype=np.float32)


def create_test_document(
    document_id: str,
    scope_type: str,
    scope_id: str,
    uploaded_by: str,
    text: str,
) -> IngestionResult:
    """Create a one-chunk document with explicit scope metadata."""

    chunk = DocumentChunk(
        chunk_id=f"{document_id}_chunk_0000",
        document_id=document_id,
        text=text,
        source=f"{document_id}.txt",
        page_numbers=(1,),
        heading_path=("Keyword Retrieval Test",),
        chunk_index=0,
    )

    metadata = {
        "document_id": document_id,
        "filename": f"{document_id}.txt",
        "version": 1,
        "chunk_id": chunk.chunk_id,
        "chunk_index": 0,
        "predicted_category": "IT",
        "scope_type": scope_type,
        "scope_id": scope_id,
        "uploaded_by": uploaded_by,
    }

    return IngestionResult(
        document_id=document_id,
        filename=f"{document_id}.txt",
        file_hash=f"file_hash_{document_id}",
        content_hash=f"content_hash_{document_id}",
        predicted_category="IT",
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
    """Verify BM25 ranking and personal/project scope isolation."""

    store = VectorStore(
        encoder=FakeEmbeddingEncoder(),
        db_path=TEST_DB_PATH,
        collection_name=TEST_COLLECTION,
    )

    retriever = KeywordRetriever(store)

    test_chunk_ids = [
        "keyword_user_a_chunk_0000",
        "keyword_user_b_chunk_0000",
        "keyword_project_alpha_chunk_0000",
    ]

    store.collection.delete(ids=test_chunk_ids)

    documents = [
        create_test_document(
            document_id="keyword_user_a",
            scope_type="personal",
            scope_id="user_a",
            uploaded_by="user_a",
            text=(
                "AWS Lambda processes messages from Amazon SQS. "
                "CloudWatch monitors Lambda execution."
            ),
        ),
        create_test_document(
            document_id="keyword_user_b",
            scope_type="personal",
            scope_id="user_b",
            uploaded_by="user_b",
            text=(
                "PostgreSQL database administration and "
                "relational database management."
            ),
        ),
        create_test_document(
            document_id="keyword_project_alpha",
            scope_type="project",
            scope_id="project_alpha",
            uploaded_by="user_a",
            text=(
                "AWS Lambda and SQS are used in the "
                "Project Alpha messaging architecture."
            ),
        ),
    ]

    print("=" * 60)
    print("BM25 KEYWORD RETRIEVAL TEST")
    print("=" * 60)

    for document in documents:
        assert store.upsert_document(document) == 1

    print("\nThree scoped documents stored.")

    # Test 1: Exact keyword retrieval in personal scope.
    user_a_results = retriever.search(
        query="AWS Lambda SQS",
        scope_type="personal",
        scope_id="user_a",
        top_k=5,
    )

    assert len(user_a_results) == 1
    assert user_a_results[0]["document_id"] == "keyword_user_a"
    assert user_a_results[0]["scope_id"] == "user_a"

    print("TEST 1 PASSED: Exact keyword retrieval works.")

    # Test 2: Personal scope isolation.
    user_b_results = retriever.search(
        query="AWS Lambda SQS",
        scope_type="personal",
        scope_id="user_b",
        top_k=5,
    )

    assert user_b_results == []

    print("TEST 2 PASSED: User B cannot retrieve User A's document.")

    # Test 3: Project scope retrieval.
    project_results = retriever.search(
        query="AWS Lambda SQS",
        scope_type="project",
        scope_id="project_alpha",
        top_k=5,
    )

    assert len(project_results) == 1
    assert (
        project_results[0]["document_id"]
        == "keyword_project_alpha"
    )

    print("TEST 3 PASSED: Project retrieval is scope-aware.")

    # Test 4: Unknown scope returns no results.
    unknown_results = retriever.search(
        query="AWS Lambda",
        scope_type="personal",
        scope_id="user_c",
        top_k=5,
    )

    assert unknown_results == []

    print("TEST 4 PASSED: Unknown scope returns no results.")

    # Test 5: Empty query rejection.
    try:
        retriever.search(
            query=" ",
            scope_type="personal",
            scope_id="user_a",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Empty query should be rejected.")

    print("TEST 5 PASSED: Empty query is rejected.")

    # Test 6: Invalid scope rejection.
    try:
        retriever.search(
            query="AWS",
            scope_type="legacy",
            scope_id="legacy_unassigned",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid scope should be rejected.")

    print("TEST 6 PASSED: Invalid scope is rejected.")

    # Clean up only records created by this test.
    deleted = store.delete_chunks(test_chunk_ids)
    assert deleted == 3

    print("\nTest records cleaned up.")
    print("\nALL BM25 KEYWORD RETRIEVAL TESTS PASSED")


if __name__ == "__main__":
    main()
