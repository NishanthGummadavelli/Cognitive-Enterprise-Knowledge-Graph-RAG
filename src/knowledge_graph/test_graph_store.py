
"""Integration tests for Neo4j document and chunk storage."""

import uuid

from src.ingestion.chunker import DocumentChunk
from src.ingestion.pipeline import IngestedChunk, IngestionResult
from src.knowledge_graph.connection import Neo4jConnection
from src.knowledge_graph.graph_store import GraphStore


def create_test_document(
    document_id: str,
    version: int,
) -> IngestionResult:
    """Build a small versioned document without reading a physical file."""

    # Generate stable, version-specific chunk IDs.
    first_chunk_id = f"{document_id}_v{version}_chunk_0000"
    second_chunk_id = f"{document_id}_v{version}_chunk_0001"

    first_chunk = DocumentChunk(
        chunk_id=first_chunk_id,
        document_id=document_id,
        text=f"Test document content for version {version}.",
        source="graph_store_test.txt",
        page_numbers=[1],
        heading_path=["Test"],
        chunk_index=0,
        previous_chunk_id=None,
        next_chunk_id=second_chunk_id,
    )

    second_chunk = DocumentChunk(
        chunk_id=second_chunk_id,
        document_id=document_id,
        text=f"Additional test content for version {version}.",
        source="graph_store_test.txt",
        page_numbers=[1],
        heading_path=["Test"],
        chunk_index=1,
        previous_chunk_id=first_chunk_id,
        next_chunk_id=None,
    )

    # Attach the version metadata expected by GraphStore.
    chunks = [
        IngestedChunk(
            chunk=first_chunk,
            metadata={"version": version},
        ),
        IngestedChunk(
            chunk=second_chunk,
            metadata={"version": version},
        ),
    ]

    return IngestionResult(
        document_id=document_id,
        filename="graph_store_test.txt",
        file_hash=f"test_file_hash_v{version}",
        content_hash=f"test_content_hash_v{version}",
        predicted_category="IT",
        classification_confidence=0.95,
        total_characters=sum(len(item.chunk.text) for item in chunks),
        total_chunks=len(chunks),
        chunks=chunks,
    )



def get_document_graph_counts(
    connection: Neo4jConnection,
    document_id: str,
) -> dict[str, int]:
    """Retrieve document, chunk and relationship counts for one test document."""

    with connection.driver.session(
        database=connection.database
    ) as session:
        record = session.run(
            """
            OPTIONAL MATCH (d:Document {document_id: $document_id})
            WITH d

            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            WITH d, count(c) AS chunk_count

            OPTIONAL MATCH
                (first:Chunk {document_id: $document_id})
                -[r:NEXT_CHUNK]->(second:Chunk)

            RETURN
                count(d) AS document_count,
                chunk_count,
                count(r) AS next_chunk_count
            """,
            document_id=document_id,
        ).single()

        if record is None:
            raise RuntimeError("Could not retrieve graph test counts.")

        return {
            "documents": record["document_count"],
            "chunks": record["chunk_count"],
            "next_chunk_links": record["next_chunk_count"],
        }


def test_graph_store() -> None:
    """Test graph creation, retries, version updates and deletion."""

    connection = Neo4jConnection()
    document_id = f"kg_test_{uuid.uuid4().hex}"

    graph_store = GraphStore(connection)

    try:
        connection.verify_connection()

        # TEST 1: Store the initial document version.
        print("\nTEST 1: INITIAL DOCUMENT UPLOAD")

        version_one = create_test_document(document_id, version=1)

        result = graph_store.upsert_document(version_one)

        assert result["chunks_stored"] == 2
        assert result["chunks_deleted"] == 0

        counts = get_document_graph_counts(connection, document_id)

        assert counts == {
            "documents": 1,
            "chunks": 2,
            "next_chunk_links": 1,
        }

        print("Document node created.")
        print("Two chunk nodes and their relationships created.")

        # TEST 2: Repeat the same upload to verify idempotency.
        print("\nTEST 2: DUPLICATE UPLOAD")

        retry_result = graph_store.upsert_document(version_one)

        assert retry_result["chunks_stored"] == 2
        assert retry_result["chunks_deleted"] == 0

        counts = get_document_graph_counts(connection, document_id)

        assert counts == {
            "documents": 1,
            "chunks": 2,
            "next_chunk_links": 1,
        }

        print("Repeated upload did not create duplicate nodes.")

        # TEST 3: Update the document to version 2.
        print("\nTEST 3: DOCUMENT VERSION UPDATE")

        version_two = create_test_document(document_id, version=2)

        update_result = graph_store.upsert_document(version_two)

        assert update_result["chunks_stored"] == 2
        assert update_result["chunks_deleted"] == 2

        counts = get_document_graph_counts(connection, document_id)

        assert counts == {
            "documents": 1,
            "chunks": 2,
            "next_chunk_links": 1,
        }

        print("Version 2 stored successfully.")
        print("Old version chunks removed.")

        # TEST 4: Delete the document and its chunks.
        print("\nTEST 4: DOCUMENT DELETION")

        deleted_count = graph_store.delete_document(document_id)

        assert deleted_count == 2

        counts = get_document_graph_counts(connection, document_id)

        assert counts == {
            "documents": 0,
            "chunks": 0,
            "next_chunk_links": 0,
        }

        print("Document and chunk nodes deleted successfully.")

        # Verify that repeated deletion is safe.
        repeated_delete_count = graph_store.delete_document(document_id)

        assert repeated_delete_count == 0

        print("Repeated deletion handled safely.")
        print("\nALL GRAPH STORAGE TESTS PASSED.")

    finally:
        # Clean up the uniquely identified test document if a test fails.
        try:
            graph_store.delete_document(document_id)
        finally:
            connection.close()


if __name__ == "__main__":
    test_graph_store()
