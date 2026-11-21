
"""Manage persistent enterprise document identity and version history."""

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path


# Resolve the project root independently of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Store the registry separately from ChromaDB.
DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT / "data" / "document_registry.db"
)


@dataclass
class RegistryResult:
    """Describe the outcome of a document registration operation."""

    status: str
    document_id: str
    version: int
    duplicate_reason: str | None = None


class DocumentRegistry:
    """Initialize and manage the SQLite enterprise document registry."""

    def __init__(
        self,
        db_path: Path = DEFAULT_REGISTRY_PATH,
    ) -> None:
        """Initialize the registry database and create its tables.

        Args:
            db_path: Location of the SQLite database file.

        Raises:
            RuntimeError: If the database cannot be initialized.
        """

        self.db_path = Path(db_path)

        try:
            # Ensure the database directory exists.
            self.db_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._initialize_database()

        except sqlite3.Error as error:
            raise RuntimeError(
                f"Could not initialize document registry: {error}"
            ) from error

    def _initialize_database(self) -> None:
        """Create document tables and indexes if they do not exist."""

        connection = sqlite3.connect(self.db_path)

        try:
            # Enable foreign-key enforcement for this connection.
            connection.execute("PRAGMA foreign_keys = ON")

            # Improve reliability of persistent database operations.
            connection.execute("PRAGMA journal_mode = WAL")

            # Store the current identity and state of each document.
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

            # Preserve the information associated with every document version.
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

            # Indexes improve duplicate lookup performance.
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_file_hash
                ON documents(file_hash)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_content_hash
                ON documents(content_hash)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_documents_filename
                ON documents(filename)
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
    ) -> RegistryResult:
        """Register a new document, skip duplicates or create a new version.

        Args:
            filename: Name of the uploaded document.
            file_hash: SHA-256 hash of the original file.
            content_hash: SHA-256 hash of normalized extracted content.

        Returns:
            Registration result containing document identity and version.

        Raises:
            ValueError: If required inputs are empty or the filename is
                ambiguous.
            RuntimeError: If the database operation fails.
        """

        # Validate all inputs before accessing the database.
        if not all(
            isinstance(value, str) and value.strip()
            for value in (filename, file_hash, content_hash)
        ):
            raise ValueError(
                "Filename and both document hashes are required."
            )

        filename = filename.strip()

        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )

        try:
            # Acquire a write lock to avoid conflicting updates.
            connection.execute("BEGIN IMMEDIATE")

            # Check whether this exact file is already registered.
            existing_file = connection.execute(
                """
                SELECT document_id, current_version
                FROM documents
                WHERE file_hash = ?
                  AND status = 'active'
                LIMIT 1
                """,
                (file_hash,),
            ).fetchone()

            if existing_file:
                connection.commit()

                return RegistryResult(
                    status="duplicate_file",
                    document_id=existing_file[0],
                    version=existing_file[1],
                    duplicate_reason="identical_file",
                )

            # Find an existing active document with this filename.
            matching_documents = connection.execute(
                """
                SELECT document_id, current_version, content_hash
                FROM documents
                WHERE filename = ?
                  AND status = 'active'
                """,
                (filename,),
            ).fetchall()

            # Avoid updating an ambiguous document identity.
            if len(matching_documents) > 1:
                raise ValueError(
                    f"Multiple active documents use the filename "
                    f"'{filename}'. A unique document identifier is required."
                )

            if matching_documents:
                document_id, current_version, previous_content_hash = (
                    matching_documents[0]
                )

                # Check whether the new content belongs to another document.
                existing_content = connection.execute(
                    """
                    SELECT document_id, current_version
                    FROM documents
                    WHERE content_hash = ?
                      AND status = 'active'
                      AND document_id != ?
                    LIMIT 1
                    """,
                    (content_hash, document_id),
                ).fetchone()

                if existing_content:
                    connection.commit()

                    return RegistryResult(
                        status="duplicate_content",
                        document_id=existing_content[0],
                        version=existing_content[1],
                        duplicate_reason="identical_content",
                    )

                # Avoid creating a version when extracted content is unchanged.
                if content_hash == previous_content_hash:
                    connection.commit()

                    return RegistryResult(
                        status="duplicate_content",
                        document_id=document_id,
                        version=current_version,
                        duplicate_reason="unchanged_content",
                    )

                # Create the next version while retaining document identity.
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

                # Update the current document record.
                connection.execute(
                    """
                    UPDATE documents
                    SET file_hash = ?,
                        content_hash = ?,
                        current_version = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = ?
                    """,
                    (
                        file_hash,
                        content_hash,
                        new_version,
                        document_id,
                    ),
                )

                connection.commit()

                return RegistryResult(
                    status="updated",
                    document_id=document_id,
                    version=new_version,
                )

            # No matching filename: check for identical content elsewhere.
            existing_content = connection.execute(
                """
                SELECT document_id, current_version
                FROM documents
                WHERE content_hash = ?
                  AND status = 'active'
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()

            if existing_content:
                connection.commit()

                return RegistryResult(
                    status="duplicate_content",
                    document_id=existing_content[0],
                    version=existing_content[1],
                    duplicate_reason="identical_content",
                )

            # Generate a new identity for a completely new document.
            document_id = f"doc_{uuid.uuid4().hex}"

            # Insert the current document record.
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename,
                    file_hash,
                    content_hash,
                    current_version
                )
                VALUES (?, ?, ?, ?, 1)
                """,
                (
                    document_id,
                    filename,
                    file_hash,
                    content_hash,
                ),
            )

            # Record the initial version.
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

            # Commit both records as one transaction.
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
    ) -> RegistryResult:
        """Soft-delete a document while preserving its version history.

        Args:
            document_id: Stable identifier of the document to delete.

        Returns:
            Result describing the deletion operation.

        Raises:
            ValueError: If the document ID is empty or not active.
            RuntimeError: If the database operation fails.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )

        try:
            # Acquire a write lock before changing the document status.
            connection.execute("BEGIN IMMEDIATE")

            existing = connection.execute(
                """
                SELECT current_version, status
                FROM documents
                WHERE document_id = ?
                """,
                (document_id.strip(),),
            ).fetchone()

            if existing is None:
                raise ValueError(
                    f"Document not found: {document_id}"
                )

            current_version, status = existing

            if status == "deleted":
                raise ValueError(
                    f"Document is already deleted: {document_id}"
                )

            # Mark the document as deleted without removing its history.
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
            Current document details, or None if it does not exist.
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
                    status
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
