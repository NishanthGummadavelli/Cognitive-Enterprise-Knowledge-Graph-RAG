
"""Manage scoped document identity, version history and deletion."""

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "document_registry.db"

VALID_SCOPE_TYPES = {"personal", "project"}


@dataclass
class RegistryResult:
    """Describe the outcome of a document registration operation."""

    status: str
    document_id: str
    version: int
    duplicate_reason: str | None = None


class DocumentRegistry:
    """Manage personal and collaborative project document records."""

    def __init__(
        self,
        db_path: Path = DEFAULT_REGISTRY_PATH,
    ) -> None:
        """Initialize the registry database.

        Args:
            db_path: Location of the SQLite database file.

        Raises:
            RuntimeError: If the database cannot be initialized.
        """

        self.db_path = Path(db_path)

        try:
            self.db_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._initialize_database()

        except sqlite3.Error as error:
            raise RuntimeError(
                f"Could not initialize document registry: {error}"
            ) from error

    @staticmethod
    def _validate_scope(
        scope_type: str,
        scope_id: str,
        uploaded_by: str,
    ) -> None:
        """Validate the ownership and scope information.

        Args:
            scope_type: Either personal or project.
            scope_id: User ID or project ID.
            uploaded_by: ID of the user uploading the document.

        Raises:
            ValueError: If any scope field is invalid.
        """

        if scope_type not in VALID_SCOPE_TYPES:
            raise ValueError(
                "Scope type must be either 'personal' or 'project'."
            )

        for field_name, value in (
            ("scope_id", scope_id),
            ("uploaded_by", uploaded_by),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty.")

        if scope_type == "personal" and scope_id != uploaded_by:
            raise ValueError(
                "A personal document must belong to its uploading user."
            )

    def _initialize_database(self) -> None:
        """Create or safely migrate the registry database."""

        connection = sqlite3.connect(self.db_path)

        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    current_version INTEGER NOT NULL
                        CHECK (current_version >= 1),
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'deleted')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Add scope columns to databases created by the earlier version.
            existing_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(documents)"
                ).fetchall()
            }

            migrations = {
                "scope_type": (
                    "TEXT NOT NULL DEFAULT 'legacy'"
                ),
                "scope_id": (
                    "TEXT NOT NULL DEFAULT 'legacy_unassigned'"
                ),
                "uploaded_by": (
                    "TEXT NOT NULL DEFAULT 'legacy_unassigned'"
                ),
            }

            for column_name, column_definition in migrations.items():
                if column_name not in existing_columns:
                    connection.execute(
                        f"""
                        ALTER TABLE documents
                        ADD COLUMN {column_name} {column_definition}
                        """
                    )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL
                        CHECK (version >= 1),
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (document_id, version),
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE
                )
                """
            )

            # Scope-aware indexes support isolated duplicate detection.
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_scope_file_hash
                ON documents(scope_type, scope_id, file_hash)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_scope_content_hash
                ON documents(scope_type, scope_id, content_hash)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_scope_filename
                ON documents(scope_type, scope_id, filename)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_versions_file_hash
                ON document_versions(file_hash)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_versions_content_hash
                ON document_versions(content_hash)
                """
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def register_or_update_document(
        self,
        filename: str,
        file_hash: str,
        content_hash: str,
        scope_type: str,
        scope_id: str,
        uploaded_by: str,
    ) -> RegistryResult:
        """Register a document or update its existing version within a scope.

        Duplicate detection is limited to the supplied scope. The same file
        can therefore exist independently in different personal spaces or
        collaborative projects.

        Args:
            filename: Name of the uploaded document.
            file_hash: SHA-256 hash of the original file.
            content_hash: SHA-256 hash of normalized extracted content.
            scope_type: Personal or project.
            scope_id: User ID for personal scope or project ID for project scope.
            uploaded_by: User ID of the uploader.

        Returns:
            Registration result containing document identity and version.

        Raises:
            ValueError: If required inputs or scope information are invalid.
            RuntimeError: If the database operation fails.
        """

        if not all(
            isinstance(value, str) and value.strip()
            for value in (filename, file_hash, content_hash)
        ):
            raise ValueError(
                "Filename and both document hashes are required."
            )

        self._validate_scope(scope_type, scope_id, uploaded_by)

        filename = filename.strip()
        scope_id = scope_id.strip()
        uploaded_by = uploaded_by.strip()

        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )

        try:
            connection.execute("BEGIN IMMEDIATE")

            scope_parameters = (scope_type, scope_id)

            # Check for an identical original file in this scope.
            existing_file = connection.execute(
                """
                SELECT document_id, current_version
                FROM documents
                WHERE scope_type = ?
                  AND scope_id = ?
                  AND file_hash = ?
                  AND status = 'active'
                LIMIT 1
                """,
                (*scope_parameters, file_hash),
            ).fetchone()

            if existing_file:
                connection.commit()

                return RegistryResult(
                    status="duplicate_file",
                    document_id=existing_file[0],
                    version=existing_file[1],
                    duplicate_reason="identical_file",
                )

            # Find an existing document with the same filename in this scope.
            matching_documents = connection.execute(
                """
                SELECT document_id, current_version, content_hash
                FROM documents
                WHERE scope_type = ?
                  AND scope_id = ?
                  AND filename = ?
                  AND status = 'active'
                """,
                (*scope_parameters, filename),
            ).fetchall()

            if len(matching_documents) > 1:
                raise ValueError(
                    f"Multiple active documents use filename '{filename}' "
                    "within this scope."
                )

            if matching_documents:
                document_id, current_version, previous_content_hash = (
                    matching_documents[0]
                )

                # Check for identical content in another document in the
                # same scope.
                existing_content = connection.execute(
                    """
                    SELECT document_id, current_version
                    FROM documents
                    WHERE scope_type = ?
                      AND scope_id = ?
                      AND content_hash = ?
                      AND status = 'active'
                      AND document_id != ?
                    LIMIT 1
                    """,
                    (
                        scope_type,
                        scope_id,
                        content_hash,
                        document_id,
                    ),
                ).fetchone()

                if existing_content:
                    connection.commit()

                    return RegistryResult(
                        status="duplicate_content",
                        document_id=existing_content[0],
                        version=existing_content[1],
                        duplicate_reason="identical_content",
                    )

                if content_hash == previous_content_hash:
                    connection.commit()

                    return RegistryResult(
                        status="duplicate_content",
                        document_id=document_id,
                        version=current_version,
                        duplicate_reason="unchanged_content",
                    )

                new_version = current_version + 1

                connection.execute(
                    """
                    INSERT INTO document_versions (
                        document_id,
                        version,
                        filename,
                        file_hash,
                        content_hash
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        new_version,
                        filename,
                        file_hash,
                        content_hash,
                    ),
                )

                connection.execute(
                    """
                    UPDATE documents
                    SET file_hash = ?,
                        content_hash = ?,
                        current_version = ?,
                        uploaded_by = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = ?
                    """,
                    (
                        file_hash,
                        content_hash,
                        new_version,
                        uploaded_by,
                        document_id,
                    ),
                )

                connection.commit()

                return RegistryResult(
                    status="updated",
                    document_id=document_id,
                    version=new_version,
                )

            # Check identical content elsewhere in this scope.
            existing_content = connection.execute(
                """
                SELECT document_id, current_version
                FROM documents
                WHERE scope_type = ?
                  AND scope_id = ?
                  AND content_hash = ?
                  AND status = 'active'
                LIMIT 1
                """,
                (
                    scope_type,
                    scope_id,
                    content_hash,
                ),
            ).fetchone()

            if existing_content:
                connection.commit()

                return RegistryResult(
                    status="duplicate_content",
                    document_id=existing_content[0],
                    version=existing_content[1],
                    duplicate_reason="identical_content",
                )

            document_id = f"doc_{uuid.uuid4().hex}"

            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename,
                    file_hash,
                    content_hash,
                    current_version,
                    scope_type,
                    scope_id,
                    uploaded_by
                )
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    file_hash,
                    content_hash,
                    scope_type,
                    scope_id,
                    uploaded_by,
                ),
            )

            connection.execute(
                """
                INSERT INTO document_versions (
                    document_id,
                    version,
                    filename,
                    file_hash,
                    content_hash
                )
                VALUES (?, 1, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    file_hash,
                    content_hash,
                ),
            )

            connection.commit()

            return RegistryResult(
                status="registered",
                document_id=document_id,
                version=1,
            )

        except ValueError:
            connection.rollback()
            raise

        except Exception as error:
            connection.rollback()

            raise RuntimeError(
                f"Document registration or update failed: {error}"
            ) from error

        finally:
            connection.close()

    def delete_document(
        self,
        document_id: str,
        scope_type: str,
        scope_id: str,
    ) -> RegistryResult:
        """Soft-delete a document after verifying its scope.

        This method verifies scope ownership but does not determine project
        membership or project-owner permissions. Those checks belong in the
        document management service.

        Args:
            document_id: Stable identifier of the document.
            scope_type: Expected document scope type.
            scope_id: Expected user or project scope ID.

        Returns:
            Result describing the deletion operation.

        Raises:
            ValueError: If the document is missing, deleted or outside scope.
            RuntimeError: If the database operation fails.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("Scope ID cannot be empty.")

        if scope_type not in VALID_SCOPE_TYPES:
            raise ValueError("Invalid document scope type.")

        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )

        try:
            connection.execute("BEGIN IMMEDIATE")

            existing = connection.execute(
                """
                SELECT current_version, status, scope_type, scope_id
                FROM documents
                WHERE document_id = ?
                """,
                (document_id.strip(),),
            ).fetchone()

            if existing is None:
                raise ValueError(
                    f"Document not found: {document_id}"
                )

            current_version, status, stored_scope_type, stored_scope_id = (
                existing
            )

            if (
                stored_scope_type != scope_type
                or stored_scope_id != scope_id.strip()
            ):
                raise ValueError(
                    "Document does not belong to the supplied scope."
                )

            if status == "deleted":
                raise ValueError(
                    f"Document is already deleted: {document_id}"
                )

            connection.execute(
                """
                UPDATE documents
                SET status = 'deleted',
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (document_id.strip(),),
            )

            connection.commit()

            return RegistryResult(
                status="deleted",
                document_id=document_id,
                version=current_version,
            )

        except ValueError:
            connection.rollback()
            raise

        except Exception as error:
            connection.rollback()

            raise RuntimeError(
                f"Document deletion failed: {error}"
            ) from error

        finally:
            connection.close()

    def get_document(
        self,
        document_id: str,
    ) -> dict[str, object] | None:
        """Retrieve the current registry state of a document.

        Args:
            document_id: Stable document identifier.

        Returns:
            Document details, including ownership and scope, or None.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row

        try:
            record = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    file_hash,
                    content_hash,
                    current_version,
                    status,
                    scope_type,
                    scope_id,
                    uploaded_by,
                    created_at,
                    updated_at
                FROM documents
                WHERE document_id = ?
                """,
                (document_id.strip(),),
            ).fetchone()

            return dict(record) if record is not None else None

        except sqlite3.Error as error:
            raise RuntimeError(
                f"Could not retrieve document state: {error}"
            ) from error

        finally:
            connection.close()
