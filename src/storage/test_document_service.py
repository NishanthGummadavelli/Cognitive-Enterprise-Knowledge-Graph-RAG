
"""Test document service recovery across SQLite, ChromaDB and Neo4j."""

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import VectorStore
from src.extraction.schemas import ExtractionResult
from src.ingestion.pipeline import DocumentIngestionPipeline
from src.knowledge_graph.connection import Neo4jConnection
from src.knowledge_graph.graph_store import GraphStore
from src.storage.document_registry import DocumentRegistry
from src.storage.document_service import DocumentManagementService
from src.storage.test_authorization import TestAuthorizationService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FILE = PROJECT_ROOT / "data" / "raw" / "sample_policy.txt"
TEST_VECTOR_PATH = PROJECT_ROOT / "data" / "chroma_test_service"
TEST_COLLECTION = "document_service_recovery_test"


class FakeEntityExtractor:
    """Provide deterministic extraction results without calling Ollama."""

    def extract(self, text: str) -> ExtractionResult:
        """Return a valid empty extraction result for a test chunk."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Test extractor received empty chunk text.")

        return ExtractionResult(
            entities=[],
            relationships=[],
        )


def get_graph_counts(
    connection: Neo4jConnection,
    document_id: str,
) -> dict:
    """Return the number of graph nodes associated with one test document."""

    query = """
    OPTIONAL MATCH (d:Document {document_id: $document_id})
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    RETURN
        count(DISTINCT d) AS documents,
        count(DISTINCT c) AS chunks,
        max(d.version) AS version
    """

    with connection.driver.session(
        database=connection.database
    ) as session:
        record = session.run(
            query,
            document_id=document_id,
        ).single()

        if record is None:
            raise RuntimeError("Could not retrieve graph document counts.")

        return {
            "documents": record["documents"],
            "chunks": record["chunks"],
            "version": record["version"],
        }


def main() -> None:
    """Verify recovery from upload, update and deletion failures."""

    if not SAMPLE_FILE.is_file():
        raise FileNotFoundError(f"Test document not found: {SAMPLE_FILE}")

    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)

        registry = DocumentRegistry(
            db_path=temporary_path / "test_registry.db"
        )

        encoder = EmbeddingEncoder()

        vector_store = VectorStore(
            encoder=encoder,
            db_path=TEST_VECTOR_PATH,
            collection_name=TEST_COLLECTION,
        )

        # Clear only this dedicated test collection.
        existing_ids = vector_store.collection.get()["ids"]

        if existing_ids:
            vector_store.collection.delete(ids=existing_ids)

        # Keep the Neo4j connection open throughout the test.
        neo4j_connection = Neo4jConnection()
        neo4j_connection.verify_connection()

        graph_store = GraphStore(neo4j_connection)

        # Use a fake extractor to keep recovery tests fast and deterministic.
        entity_extractor = FakeEntityExtractor()

        # Simulate an authenticated personal-space owner for this test.
        authorization_service = TestAuthorizationService(
            personal_users={"test_user"},
        )

        service = DocumentManagementService(
            ingestion_pipeline=DocumentIngestionPipeline(),
            registry=registry,
            vector_store=vector_store,
            graph_store=graph_store,
            entity_extractor=entity_extractor,
            authorization_service=authorization_service,
        )

        test_file = temporary_path / "sample_policy.txt"
        test_file.write_text(
            SAMPLE_FILE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        original_upsert = vector_store.upsert_document
        original_delete = vector_store.delete_chunks
        original_graph_upsert = graph_store.upsert_document
        original_graph_delete = graph_store.delete_document

        first_document_id = None

        try:
            print("=" * 65)
            print("DOCUMENT SERVICE FAILURE RECOVERY TEST")
            print("=" * 65)

            # TEST 1: Initial upload and recovery across both storage systems.
            print("\nTEST 1: INITIAL UPLOAD FAILURE")

            upsert_failure = {"remaining": 1}

            def fail_first_upsert(result):
                """Simulate one failed ChromaDB insertion."""

                if upsert_failure["remaining"] > 0:
                    upsert_failure["remaining"] -= 1
                    raise RuntimeError("Simulated initial vector failure.")

                return original_upsert(result)

            vector_store.upsert_document = fail_first_upsert

            try:
                service.upload_document(
                    file_path=test_file,
                    user_id="test_user",
                    scope_type="personal",
                    scope_id="test_user",
                )
                raise AssertionError("Expected initial upload failure.")
            except RuntimeError as error:
                print(f"Expected ChromaDB failure: {error}")

            # Find the registered document ID in the temporary SQLite DB.
            connection = sqlite3.connect(registry.db_path)

            try:
                row = connection.execute(
                    "SELECT document_id FROM documents"
                ).fetchone()

                if row is None:
                    raise AssertionError("Document was not registered.")

                first_document_id = row[0]

            finally:
                connection.close()

            assert registry.get_document(first_document_id)["status"] == "active"
            assert vector_store.get_document_chunk_ids(first_document_id) == []

            # The next upload should repair ChromaDB, but we deliberately
            # simulate a Neo4j failure to test the second recovery stage.
            graph_upsert_failure = {"remaining": 1}

            def fail_first_graph_upsert(
                result,
                extraction_results=None,
            ):
                """Simulate one failed Neo4j document insertion."""

                if graph_upsert_failure["remaining"] > 0:
                    graph_upsert_failure["remaining"] -= 1
                    raise RuntimeError("Simulated Neo4j insertion failure.")

                return original_graph_upsert(
                    result,
                    extraction_results=extraction_results,
                )

            graph_store.upsert_document = fail_first_graph_upsert

            try:
                service.upload_document(
                    file_path=test_file,
                    user_id="test_user",
                    scope_type="personal",
                    scope_id="test_user",
                )
                raise AssertionError("Expected Neo4j upload failure.")
            except RuntimeError as error:
                print(f"Expected Neo4j failure: {error}")

            # ChromaDB should now contain the document, while Neo4j does not.
            assert vector_store.get_document_chunk_ids(first_document_id)

            graph_state = get_graph_counts(
                neo4j_connection,
                first_document_id,
            )
            assert graph_state["documents"] == 0

            # Retry should repair the graph without duplicating vectors.
            recovered_initial = service.upload_document(
                    file_path=test_file,
                    user_id="test_user",
                    scope_type="personal",
                    scope_id="test_user",
                )

            assert recovered_initial.status == "duplicate_file"
            assert recovered_initial.recovered is True
            assert recovered_initial.chunks_stored > 0
            assert recovered_initial.graph_chunks_stored > 0
            assert recovered_initial.graph_entities_stored == 0
            assert recovered_initial.graph_relationships_stored == 0

            graph_state = get_graph_counts(
                neo4j_connection,
                first_document_id,
            )
            assert graph_state["documents"] == 1
            assert graph_state["chunks"] > 0
            assert graph_state["version"] == 1

            print("Initial upload recovered in ChromaDB and Neo4j.")
            print(f"Document ID: {first_document_id}")

            # TEST 2: Fail version 2 insertion, then retry.
            print("\nTEST 2: DOCUMENT UPDATE FAILURE")

            updated_text = (
                test_file.read_text(encoding="utf-8")
                + "\n\nRECOVERY TEST UPDATE\n"
                + "HR reviews annual leave balances every quarter."
            )
            test_file.write_text(updated_text, encoding="utf-8")

            upsert_failure["remaining"] = 1

            try:
                service.upload_document(
                    file_path=test_file,
                    user_id="test_user",
                    scope_type="personal",
                    scope_id="test_user",
                )
                raise AssertionError("Expected update failure.")
            except RuntimeError as error:
                print(f"Expected update failure: {error}")

            document_state = registry.get_document(first_document_id)
            assert document_state["current_version"] == 2

            # Old version vectors should remain until replacement succeeds.
            old_ids = vector_store.get_document_chunk_ids(first_document_id)
            assert old_ids
            assert all("_v1_" in chunk_id for chunk_id in old_ids)

            recovered_update = service.upload_document(
                    file_path=test_file,
                    user_id="test_user",
                    scope_type="personal",
                    scope_id="test_user",
                )

            assert recovered_update.version == 2
            assert recovered_update.recovered is True
            assert recovered_update.graph_chunks_stored > 0
            assert recovered_update.graph_entities_stored == 0
            assert recovered_update.graph_relationships_stored == 0

            current_ids = vector_store.get_document_chunk_ids(
                first_document_id
            )

            assert current_ids
            assert all("_v2_" in chunk_id for chunk_id in current_ids)

            graph_state = get_graph_counts(
                neo4j_connection,
                first_document_id,
            )
            assert graph_state["documents"] == 1
            assert graph_state["chunks"] == len(current_ids)
            assert graph_state["version"] == 2

            print("Retry repaired version 2 in ChromaDB and Neo4j.")
            print("Old version vectors and graph chunks were replaced.")

            # TEST 3: Fail vector deletion, then graph deletion, then retry.
            print("\nTEST 3: DOCUMENT DELETION FAILURE")

            delete_failure = {"remaining": 1}

            def fail_first_delete(chunk_ids):
                """Simulate one failed ChromaDB deletion."""

                if delete_failure["remaining"] > 0:
                    delete_failure["remaining"] -= 1
                    raise RuntimeError("Simulated vector deletion failure.")

                return original_delete(chunk_ids)

            vector_store.delete_chunks = fail_first_delete

            graph_delete_failure = {"remaining": 1}

            def fail_first_graph_delete(document_id):
                """Simulate one failed Neo4j document deletion."""

                if graph_delete_failure["remaining"] > 0:
                    graph_delete_failure["remaining"] -= 1
                    raise RuntimeError("Simulated Neo4j deletion failure.")

                return original_graph_delete(document_id)

            graph_store.delete_document = fail_first_graph_delete

            try:
                service.delete_document(
                    document_id=first_document_id,
                    requested_by="test_user",
                )
                raise AssertionError("Expected vector deletion failure.")
            except RuntimeError as error:
                print(f"Expected ChromaDB failure: {error}")

            # SQLite should already show deleted status.
            assert (
                registry.get_document(first_document_id)["status"]
                == "deleted"
            )

            # ChromaDB still has records because its deletion failed.
            assert vector_store.get_document_chunk_ids(first_document_id)

            # The next retry should clean ChromaDB, but Neo4j deletion fails.
            try:
                service.delete_document(
                    document_id=first_document_id,
                    requested_by="test_user",
                )
                raise AssertionError("Expected Neo4j deletion failure.")
            except RuntimeError as error:
                print(f"Expected Neo4j failure: {error}")

            assert vector_store.get_document_chunk_ids(first_document_id) == []

            graph_state = get_graph_counts(
                neo4j_connection,
                first_document_id,
            )
            assert graph_state["documents"] == 1
            assert graph_state["chunks"] > 0

            # Final retry should finish Neo4j cleanup.
            recovered_deletion = service.delete_document(
                    document_id=first_document_id,
                    requested_by="test_user",
                )

            assert recovered_deletion.status == "deleted"
            assert recovered_deletion.recovered is True
            assert (
                vector_store.get_document_chunk_ids(first_document_id) == []
            )

            graph_state = get_graph_counts(
                neo4j_connection,
                first_document_id,
            )
            assert graph_state["documents"] == 0
            assert graph_state["chunks"] == 0

            print("Retry completed deletion in ChromaDB and Neo4j.")

            print("\n" + "=" * 65)
            print("ALL DOCUMENT SERVICE RECOVERY TESTS PASSED")
            print("=" * 65)

        finally:
            # Restore the original methods before attempting cleanup.
            vector_store.upsert_document = original_upsert
            vector_store.delete_chunks = original_delete
            graph_store.upsert_document = original_graph_upsert
            graph_store.delete_document = original_graph_delete

            # Remove any graph data left behind if a test failed midway.
            if first_document_id is not None:
                try:
                    graph_store.delete_document(first_document_id)
                except Exception as error:
                    print(f"Warning: Neo4j test cleanup failed: {error}")

            # Clean only the dedicated ChromaDB test collection.
            remaining_ids = vector_store.collection.get()["ids"]

            if remaining_ids:
                vector_store.collection.delete(ids=remaining_ids)

            neo4j_connection.close()


if __name__ == "__main__":
    main()
