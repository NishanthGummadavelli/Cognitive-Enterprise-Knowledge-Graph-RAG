
"""Integration tests for Neo4j document and knowledge graph storage."""

import uuid

from src.extraction.schemas import (
    Entity,
    ExtractionResult,
    Relationship,
)
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
            metadata={"version": version,
                      "scope_type": "personal",
                      "scope_id": "graph_store_test_user",
                    "uploaded_by": "graph_store_test_user",
                },
        ),
        IngestedChunk(
            chunk=second_chunk,
            metadata={"version": version,
                      "scope_type": "personal",
                      "scope_id": "graph_store_test_user",
                      "uploaded_by": "graph_store_test_user",
                },
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


def create_test_extractions(
    document: IngestionResult,
    version: int,
) -> dict[str, ExtractionResult]:
    """Build deterministic entity and relationship data for graph tests."""

    first_chunk_id = document.chunks[0].chunk.chunk_id
    second_chunk_id = document.chunks[1].chunk.chunk_id

    first_text = document.chunks[0].chunk.text

    return {
        first_chunk_id: ExtractionResult(
            entities=[
                Entity(
                    name="Test document",
                    entity_type="DOCUMENT",
                ),
                Entity(
                    name=f"version {version}",
                    entity_type="VERSION",
                ),
            ],
            relationships=[
                Relationship(
                    source="Test document",
                    relationship_type="HAS_VERSION",
                    target=f"version {version}",
                    evidence=first_text,
                )
            ],
        ),
        second_chunk_id: ExtractionResult(
            entities=[],
            relationships=[],
        ),
    }


def get_document_graph_counts(
    connection: Neo4jConnection,
    document_id: str,
) -> dict[str, int]:
    """Retrieve document, chunk and knowledge counts for one test document."""

    query = """
    OPTIONAL MATCH (d:Document {document_id: $document_id})
    WITH d

    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    WITH d, count(DISTINCT c) AS chunk_count

    OPTIONAL MATCH
        (first:Chunk {document_id: $document_id})
        -[r:NEXT_CHUNK]->(second:Chunk)
    WITH
        d,
        chunk_count,
        count(DISTINCT r) AS next_chunk_count

    OPTIONAL MATCH
        (chunk:Chunk {document_id: $document_id})
        -[:MENTIONS]->(e:Entity)
    WITH
        d,
        chunk_count,
        next_chunk_count,
        count(DISTINCT e) AS entity_count

    OPTIONAL MATCH
        (supported_chunk:Chunk {document_id: $document_id})
        -[:SUPPORTS]->(kr:KnowledgeRelationship)

    RETURN
        count(DISTINCT d) AS document_count,
        chunk_count,
        next_chunk_count,
        entity_count,
        count(DISTINCT kr) AS knowledge_relationship_count
    """

    with connection.driver.session(
        database=connection.database
    ) as session:
        record = session.run(
            query,
            document_id=document_id,
        ).single()

        if record is None:
            raise RuntimeError("Could not retrieve graph test counts.")

        return {
            "documents": record["document_count"],
            "chunks": record["chunk_count"],
            "next_chunk_links": record["next_chunk_count"],
            "entities": record["entity_count"],
            "knowledge_relationships": (
                record["knowledge_relationship_count"]
            ),
        }


def test_graph_store() -> None:
    """Test graph creation, knowledge persistence, retries and deletion."""

    connection = Neo4jConnection()
    document_id = f"kg_test_{uuid.uuid4().hex}"

    graph_store = GraphStore(connection)

    try:
        connection.verify_connection()

        # TEST 1: Store the initial document and extracted knowledge.
        print("\nTEST 1: INITIAL DOCUMENT AND KNOWLEDGE UPLOAD")

        version_one = create_test_document(document_id, version=1)
        extractions_v1 = create_test_extractions(
            version_one,
            version=1,
        )

        result = graph_store.upsert_document(
            version_one,
            extraction_results=extractions_v1,
        )

        assert result["chunks_stored"] == 2
        assert result["chunks_deleted"] == 0
        assert result["entities_stored"] == 2
        assert result["relationships_stored"] == 1

        counts = get_document_graph_counts(connection, document_id)

        assert counts == {
            "documents": 1,
            "chunks": 2,
            "next_chunk_links": 1,
            "entities": 2,
            "knowledge_relationships": 1,
        }

        print("Document and chunks created.")
        print("Two entity nodes created.")
        print("One evidence-backed relationship created.")

        # TEST 2: Repeat the same upload to verify idempotency.
        print("\nTEST 2: DUPLICATE UPLOAD")

        retry_result = graph_store.upsert_document(
            version_one,
            extraction_results=extractions_v1,
        )

        assert retry_result["chunks_stored"] == 2
        assert retry_result["chunks_deleted"] == 0
        assert retry_result["entities_stored"] == 2
        assert retry_result["relationships_stored"] == 1

        counts = get_document_graph_counts(connection, document_id)

        assert counts == {
            "documents": 1,
            "chunks": 2,
            "next_chunk_links": 1,
            "entities": 2,
            "knowledge_relationships": 1,
        }

        print("Repeated upload did not create duplicate graph records.")

        # TEST 3: Update the document and its extracted knowledge.
        print("\nTEST 3: DOCUMENT VERSION UPDATE")

        version_two = create_test_document(document_id, version=2)
        extractions_v2 = create_test_extractions(
            version_two,
            version=2,
        )

        update_result = graph_store.upsert_document(
            version_two,
            extraction_results=extractions_v2,
        )

        assert update_result["chunks_stored"] == 2
        assert update_result["chunks_deleted"] == 2
        assert update_result["entities_stored"] == 2
        assert update_result["relationships_stored"] == 1

        counts = get_document_graph_counts(connection, document_id)

        assert counts == {
            "documents": 1,
            "chunks": 2,
            "next_chunk_links": 1,
            "entities": 2,
            "knowledge_relationships": 1,
        }

        print("Version 2 stored successfully.")
        print("Old version chunks and evidence removed.")
        print("Current version knowledge retained.")

        # TEST 4: Delete the document and its extracted knowledge.
        print("\nTEST 4: DOCUMENT AND KNOWLEDGE DELETION")

        deleted_count = graph_store.delete_document(document_id)

        assert deleted_count == 2

        counts = get_document_graph_counts(connection, document_id)

        assert counts == {
            "documents": 0,
            "chunks": 0,
            "next_chunk_links": 0,
            "entities": 0,
            "knowledge_relationships": 0,
        }

        print("Document and chunk nodes deleted.")
        print("Extracted relationship evidence deleted.")
        print("Unreferenced entity nodes cleaned up.")

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

def test_shared_entity_preservation() -> None:
    """Ensure deleting one document does not delete another's shared entities."""

    connection = Neo4jConnection()

    first_document_id = f"kg_shared_a_{uuid.uuid4().hex}"
    second_document_id = f"kg_shared_b_{uuid.uuid4().hex}"

    graph_store = GraphStore(connection)

    def get_entity_usage(name: str) -> dict[str, int]:
        """Return the entity count and number of chunks mentioning it."""

        with connection.driver.session(
            database=connection.database
        ) as session:
            record = session.run(
                """
                MATCH (e:Entity {name: $name})
                OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                RETURN
                    count(DISTINCT e) AS entity_count,
                    count(DISTINCT c) AS reference_count
                """,
                name=name,
            ).single()

            return {
                "entities": record["entity_count"],
                "references": record["reference_count"],
            }

    try:
        connection.verify_connection()

        # Create two independent documents.
        first_document = create_test_document(
            first_document_id,
            version=1,
        )
        second_document = create_test_document(
            second_document_id,
            version=1,
        )

        shared_entity = Entity(
            name="ABC Technologies",
            entity_type="ORGANIZATION",
        )

        first_unique_entity = Entity(
            name="First Document Team",
            entity_type="DEPARTMENT",
        )

        second_unique_entity = Entity(
            name="Second Document Team",
            entity_type="DEPARTMENT",
        )

        # Both documents mention the same organization.
        first_extraction = ExtractionResult(
            entities=[shared_entity, first_unique_entity],
            relationships=[],
        )

        second_extraction = ExtractionResult(
            entities=[shared_entity, second_unique_entity],
            relationships=[],
        )

        graph_store.upsert_document(
            first_document,
            extraction_results={
                first_document.chunks[0].chunk.chunk_id: first_extraction
            },
        )

        graph_store.upsert_document(
            second_document,
            extraction_results={
                second_document.chunks[0].chunk.chunk_id: second_extraction
            },
        )

        # The shared entity should have two independent references.
        assert get_entity_usage("ABC Technologies") == {
            "entities": 1,
            "references": 2,
        }

        assert get_entity_usage("First Document Team") == {
            "entities": 1,
            "references": 1,
        }

        assert get_entity_usage("Second Document Team") == {
            "entities": 1,
            "references": 1,
        }

        print("\nTEST: SHARED ENTITY PRESERVATION")
        print("Both documents reference the same entity node.")

        # Delete the first document.
        graph_store.delete_document(first_document_id)

        # The shared entity must survive because document two still uses it.
        assert get_entity_usage("ABC Technologies") == {
            "entities": 1,
            "references": 1,
        }

        # The first document's unique entity must be removed.
        assert get_entity_usage("First Document Team") == {
            "entities": 0,
            "references": 0,
        }

        # The second document's unique entity must remain.
        assert get_entity_usage("Second Document Team") == {
            "entities": 1,
            "references": 1,
        }

        print("Deleting the first document preserved the shared entity.")
        print("Unreferenced entities were removed correctly.")

        # Delete the second document.
        graph_store.delete_document(second_document_id)

        # No document references the shared entity anymore.
        assert get_entity_usage("ABC Technologies") == {
            "entities": 0,
            "references": 0,
        }

        assert get_entity_usage("Second Document Team") == {
            "entities": 0,
            "references": 0,
        }

        print("Deleting the final document removed the shared entity.")
        print("SHARED ENTITY PRESERVATION TEST PASSED.")

    finally:
        # Clean up both uniquely identified test documents.
        try:
            graph_store.delete_document(first_document_id)
            graph_store.delete_document(second_document_id)
        finally:
            connection.close()

if __name__ == "__main__":
    test_graph_store()
