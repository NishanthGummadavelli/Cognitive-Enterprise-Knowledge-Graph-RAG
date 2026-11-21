
"""Test SQLite registration, versioning and soft deletion."""

import sqlite3
import tempfile
from pathlib import Path

from src.storage.document_registry import DocumentRegistry


def main() -> None:
    """Verify document registration, updates and deletion."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        # Use an isolated database to avoid changing project data.
        database_path = Path(temporary_directory) / "test_registry.db"
        registry = DocumentRegistry(db_path=database_path)

        print("=" * 60)
        print("DOCUMENT REGISTRY LIFECYCLE TEST")
        print("=" * 60)

        # Register the first version.
        first = registry.register_or_update_document(
            filename="employee_policy.pdf",
            file_hash="file_v1",
            content_hash="content_v1",
        )

        print("\nINITIAL REGISTRATION")
        print(f"Status: {first.status}")
        print(f"Version: {first.version}")

        # Register an updated version.
        second = registry.register_or_update_document(
            filename="employee_policy.pdf",
            file_hash="file_v2",
            content_hash="content_v2",
        )

        print("\nDOCUMENT UPDATE")
        print(f"Status: {second.status}")
        print(f"Version: {second.version}")
        print(f"Same document ID: {first.document_id == second.document_id}")

        # Soft-delete the document.
        deleted = registry.delete_document(first.document_id)

        print("\nDOCUMENT DELETION")
        print(f"Status: {deleted.status}")
        print(f"Document ID: {deleted.document_id}")

        # Inspect the document and its preserved history.
        connection = sqlite3.connect(database_path)

        try:
            document = connection.execute(
                """
                SELECT status, current_version
                FROM documents
                WHERE document_id = ?
                """,
                (first.document_id,),
            ).fetchone()

            history_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM document_versions
                WHERE document_id = ?
                """,
                (first.document_id,),
            ).fetchone()[0]

        finally:
            connection.close()

        print("\nPRESERVED RECORDS")
        print(f"Document status: {document[0]}")
        print(f"Current version: {document[1]}")
        print(f"Version history records: {history_count}")

        # Verify deletion did not erase the document history.
        assert first.status == "registered"
        assert second.status == "updated"
        assert second.version == 2
        assert deleted.status == "deleted"
        assert document[0] == "deleted"
        assert document[1] == 2
        assert history_count == 2

        # Verify that a deleted document cannot be deleted again.
        try:
            registry.delete_document(first.document_id)
        except ValueError:
            print("\nRepeated deletion correctly rejected.")
        else:
            raise AssertionError(
                "Repeated deletion should have been rejected."
            )

        # Verify the deleted document is no longer treated as an active
        # document when the same filename and content are registered again.
        replacement = registry.register_or_update_document(
            filename="employee_policy.pdf",
            file_hash="file_v1",
            content_hash="content_v1",
        )

        print("\nRE-UPLOAD AFTER DELETION")
        print(f"Status: {replacement.status}")
        print(
            "New document ID:",
            replacement.document_id != first.document_id,
        )

        assert replacement.status == "registered"
        assert replacement.document_id != first.document_id

        print("\nALL DOCUMENT LIFECYCLE TESTS PASSED")


if __name__ == "__main__":
    main()
