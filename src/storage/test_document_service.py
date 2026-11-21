
"""Test document service recovery after simulated storage failures."""

from pathlib import Path
from tempfile import TemporaryDirectory

from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import VectorStore
from src.ingestion.pipeline import DocumentIngestionPipeline
from src.storage.document_registry import DocumentRegistry
from src.storage.document_service import DocumentManagementService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FILE = PROJECT_ROOT / "data" / "raw" / "sample_policy.txt"
TEST_VECTOR_PATH = PROJECT_ROOT / "data" / "chroma_test_service"
TEST_COLLECTION = "document_service_recovery_test"


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

        service = DocumentManagementService(
            ingestion_pipeline=DocumentIngestionPipeline(),
            registry=registry,
            vector_store=vector_store,
        )

        test_file = temporary_path / "sample_policy.txt"
        test_file.write_text(
            SAMPLE_FILE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        original_upsert = vector_store.upsert_document
        original_delete = vector_store.delete_chunks

        try:
            print("=" * 60)
            print("DOCUMENT SERVICE FAILURE RECOVERY TEST")
            print("=" * 60)

            # TEST 1: Fail initial vector insertion, then retry.
            print("\nTEST 1: INITIAL UPLOAD FAILURE")

            upsert_failure = {"remaining": 1}

            def fail_first_upsert(result):
                """Simulate one failed vector insertion."""

                if upsert_failure["remaining"] > 0:
                    upsert_failure["remaining"] -= 1
                    raise RuntimeError("Simulated initial vector failure.")

                return original_upsert(result)

            vector_store.upsert_document = fail_first_upsert

            try:
                service.upload_document(test_file)
                raise AssertionError("Expected initial upload failure.")
            except RuntimeError as error:
                print(f"Expected failure: {error}")

            # SQLite should contain the registered document even though
            # its vectors have not yet been indexed.
            
            # Find the actual registered document ID from SQLite.
            import sqlite3

            connection = sqlite3.connect(registry.db_path)
            try:
                first_document_id = connection.execute(
                    "SELECT document_id FROM documents"
                ).fetchone()[0]
            finally:
                connection.close()

            assert registry.get_document(first_document_id)["status"] == "active"
            assert vector_store.get_document_chunk_ids(first_document_id) == []

            recovered_initial = service.upload_document(test_file)

            assert recovered_initial.status == "duplicate_file"
            assert recovered_initial.recovered is True
            assert recovered_initial.chunks_stored > 0

            print("Retry repaired initial indexing.")
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
                service.upload_document(test_file)
                raise AssertionError("Expected update failure.")
            except RuntimeError as error:
                print(f"Expected failure: {error}")

            document_state = registry.get_document(first_document_id)
            assert document_state["current_version"] == 2

            # Old version should still be present until replacement succeeds.
            old_ids = vector_store.get_document_chunk_ids(first_document_id)
            assert old_ids
            assert all("_v1_" in chunk_id for chunk_id in old_ids)

            recovered_update = service.upload_document(test_file)

            assert recovered_update.version == 2
            assert recovered_update.recovered is True

            current_ids = vector_store.get_document_chunk_ids(
                first_document_id
            )

            assert current_ids
            assert all("_v2_" in chunk_id for chunk_id in current_ids)

            print("Retry repaired version 2 and removed old vectors.")

            # TEST 3: Fail deletion cleanup, then retry.
            print("\nTEST 3: DOCUMENT DELETION FAILURE")

            delete_failure = {"remaining": 1}

            def fail_first_delete(chunk_ids):
                """Simulate one failed vector deletion."""

                if delete_failure["remaining"] > 0:
                    delete_failure["remaining"] -= 1
                    raise RuntimeError("Simulated vector deletion failure.")

                return original_delete(chunk_ids)

            vector_store.delete_chunks = fail_first_delete

            try:
                service.delete_document(first_document_id)
                raise AssertionError("Expected deletion failure.")
            except RuntimeError as error:
                print(f"Expected failure: {error}")

            # SQLite should already show deleted status.
            assert (
                registry.get_document(first_document_id)["status"]
                == "deleted"
            )

            # ChromaDB should still have records after the simulated failure.
            assert vector_store.get_document_chunk_ids(first_document_id)

            recovered_deletion = service.delete_document(first_document_id)

            assert recovered_deletion.status == "deleted"
            assert recovered_deletion.recovered is True
            assert (
                vector_store.get_document_chunk_ids(first_document_id) == []
            )

            print("Retry completed deletion and removed remaining vectors.")

            print("\n" + "=" * 60)
            print("ALL FAILURE RECOVERY TESTS PASSED")
            print("=" * 60)

        finally:
            # Restore the original methods and clean up test vectors.
            vector_store.upsert_document = original_upsert
            vector_store.delete_chunks = original_delete

            remaining_ids = vector_store.collection.get()["ids"]

            if remaining_ids:
                vector_store.collection.delete(ids=remaining_ids)


if __name__ == "__main__":
    main()
