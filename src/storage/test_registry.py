
"""Test scope isolation, migration and lifecycle in the document registry."""

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from src.storage.document_registry import DocumentRegistry


def create_legacy_database(db_path: Path) -> None:
    """Create the previous registry schema to test safe migration."""

    connection = sqlite3.connect(db_path)

    try:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                current_version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE document_versions (
                document_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (document_id, version),
                FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE
            );

            INSERT INTO documents (
                document_id,
                filename,
                file_hash,
                content_hash,
                current_version
            )
            VALUES (
                'legacy_document_001',
                'legacy_policy.txt',
                'legacy_file_hash',
                'legacy_content_hash',
                1
            );

            INSERT INTO document_versions (
                document_id,
                version,
                filename,
                file_hash,
                content_hash
            )
            VALUES (
                'legacy_document_001',
                1,
                'legacy_policy.txt',
                'legacy_file_hash',
                'legacy_content_hash'
            );
            """
        )
        connection.commit()

    finally:
        connection.close()


def register(
    registry: DocumentRegistry,
    filename: str,
    file_hash: str,
    content_hash: str,
    scope_type: str,
    scope_id: str,
    uploaded_by: str,
):
    """Register a document using the supplied scope and uploader."""

    return registry.register_or_update_document(
        filename=filename,
        file_hash=file_hash,
        content_hash=content_hash,
        scope_type=scope_type,
        scope_id=scope_id,
        uploaded_by=uploaded_by,
    )


def main() -> None:
    """Run registry lifecycle, isolation and migration tests."""

    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)

        registry = DocumentRegistry(
            db_path=temporary_path / "test_registry.db"
        )

        print("=" * 65)
        print("SCOPE-AWARE DOCUMENT REGISTRY TESTS")
        print("=" * 65)

        # TEST 1: Register a personal document.
        print("\nTEST 1: PERSONAL DOCUMENT REGISTRATION")

        first = register(
            registry,
            "policy.txt",
            "file_hash_001",
            "content_hash_001",
            "personal",
            "user_a",
            "user_a",
        )

        assert first.status == "registered"
        assert first.version == 1

        document = registry.get_document(first.document_id)

        assert document is not None
        assert document["scope_type"] == "personal"
        assert document["scope_id"] == "user_a"
        assert document["uploaded_by"] == "user_a"

        print("Personal document registered successfully.")

        # TEST 2: Same user uploads the exact same file.
        print("\nTEST 2: DUPLICATE FILE DETECTION")

        duplicate_file = register(
            registry,
            "policy.txt",
            "file_hash_001",
            "content_hash_001",
            "personal",
            "user_a",
            "user_a",
        )

        assert duplicate_file.status == "duplicate_file"
        assert duplicate_file.document_id == first.document_id

        print("Duplicate file detected within the same scope.")

        # TEST 3: Same content under another filename, same scope.
        print("\nTEST 3: DUPLICATE CONTENT DETECTION")

        duplicate_content = register(
            registry,
            "another_policy.txt",
            "file_hash_002",
            "content_hash_001",
            "personal",
            "user_a",
            "user_a",
        )

        assert duplicate_content.status == "duplicate_content"
        assert duplicate_content.document_id == first.document_id

        print("Duplicate content detected within the same scope.")

        # TEST 4: Another user uploads the same content.
        print("\nTEST 4: PERSONAL USER ISOLATION")

        second_user = register(
            registry,
            "policy.txt",
            "file_hash_001",
            "content_hash_001",
            "personal",
            "user_b",
            "user_b",
        )

        assert second_user.status == "registered"
        assert second_user.document_id != first.document_id

        second_user_document = registry.get_document(
            second_user.document_id
        )

        assert second_user_document["scope_id"] == "user_b"

        print("User B received an independent document record.")

        # TEST 5: Same content in two different projects.
        print("\nTEST 5: PROJECT SCOPE ISOLATION")

        project_one = register(
            registry,
            "requirements.txt",
            "project_file_hash",
            "project_content_hash",
            "project",
            "project_001",
            "user_a",
        )

        project_two = register(
            registry,
            "requirements.txt",
            "project_file_hash",
            "project_content_hash",
            "project",
            "project_002",
            "user_b",
        )

        assert project_one.status == "registered"
        assert project_two.status == "registered"
        assert project_one.document_id != project_two.document_id

        print("Identical content was isolated across projects.")

        # TEST 6: Another member uploads the same content in one project.
        print("\nTEST 6: DUPLICATE DETECTION WITHIN A PROJECT")

        project_duplicate = register(
            registry,
            "copy_requirements.txt",
            "another_project_file_hash",
            "project_content_hash",
            "project",
            "project_001",
            "user_b",
        )

        assert project_duplicate.status == "duplicate_content"
        assert project_duplicate.document_id == project_one.document_id

        print("Duplicate content detected within the same project.")

        # TEST 7: Update an existing document.
        print("\nTEST 7: DOCUMENT VERSION UPDATE")

        updated = register(
            registry,
            "policy.txt",
            "file_hash_003",
            "content_hash_003",
            "personal",
            "user_a",
            "user_a",
        )

        assert updated.status == "updated"
        assert updated.document_id == first.document_id
        assert updated.version == 2

        updated_document = registry.get_document(first.document_id)

        assert updated_document["current_version"] == 2
        assert updated_document["scope_type"] == "personal"
        assert updated_document["scope_id"] == "user_a"

        print("Document updated while retaining its identity and scope.")

        # TEST 8: Reject deletion through a mismatched scope.
        print("\nTEST 8: SCOPE VALIDATION DURING DELETION")

        try:
            registry.delete_document(
                first.document_id,
                scope_type="personal",
                scope_id="user_b",
            )
            raise AssertionError("Expected scope mismatch rejection.")

        except ValueError:
            print("Deletion rejected for the wrong personal scope.")

        assert registry.get_document(first.document_id)["status"] == "active"

        # TEST 9: Soft deletion in the correct scope.
        print("\nTEST 9: SOFT DELETION")

        deleted = registry.delete_document(
            first.document_id,
            scope_type="personal",
            scope_id="user_a",
        )

        assert deleted.status == "deleted"
        assert registry.get_document(first.document_id)["status"] == "deleted"

        print("Document soft-deleted successfully.")

        # TEST 10: Re-upload after deletion creates a new identity.
        print("\nTEST 10: RE-UPLOAD AFTER DELETION")

        replacement = register(
            registry,
            "policy.txt",
            "file_hash_003",
            "content_hash_003",
            "personal",
            "user_a",
            "user_a",
        )

        assert replacement.status == "registered"
        assert replacement.document_id != first.document_id
        assert replacement.version == 1

        print("Re-upload created a new document identity.")

        # TEST 11: Verify version history remains available.
        print("\nTEST 11: VERSION HISTORY PRESERVATION")

        connection = sqlite3.connect(registry.db_path)

        try:
            versions = connection.execute(
                """
                SELECT version, file_hash, content_hash
                FROM document_versions
                WHERE document_id = ?
                ORDER BY version
                """,
                (first.document_id,),
            ).fetchall()

        finally:
            connection.close()

        assert len(versions) == 2
        assert versions[0][0] == 1
        assert versions[1][0] == 2

        print("Both document versions remain in SQLite.")

        # TEST 12: Validate legacy database migration.
        print("\nTEST 12: LEGACY DATABASE MIGRATION")

        legacy_db_path = temporary_path / "legacy_registry.db"
        create_legacy_database(legacy_db_path)

        migrated_registry = DocumentRegistry(
            db_path=legacy_db_path
        )

        legacy_document = migrated_registry.get_document(
            "legacy_document_001"
        )

        assert legacy_document is not None
        assert legacy_document["status"] == "active"
        assert legacy_document["scope_type"] == "legacy"
        assert legacy_document["scope_id"] == "legacy_unassigned"
        assert legacy_document["uploaded_by"] == "legacy_unassigned"

        connection = sqlite3.connect(legacy_db_path)

        try:
            legacy_versions = connection.execute(
                """
                SELECT COUNT(*)
                FROM document_versions
                WHERE document_id = 'legacy_document_001'
                """
            ).fetchone()[0]

        finally:
            connection.close()

        assert legacy_versions == 1

        print("Legacy document and version history preserved.")
        print("Legacy data remains isolated from personal/project scopes.")

        print("\n" + "=" * 65)
        print("ALL DOCUMENT REGISTRY TESTS PASSED")
        print("=" * 65)


if __name__ == "__main__":
    main()
