
"""Coordinate document lifecycle and recover vector-store inconsistencies."""

from dataclasses import dataclass, replace
from pathlib import Path

from src.embeddings.vector_store import VectorStore
from src.ingestion.pipeline import (
    DocumentIngestionPipeline,
    IngestionResult,
)
from src.storage.document_registry import DocumentRegistry


@dataclass
class DocumentServiceResult:
    """Describe the outcome of a document management operation."""

    status: str
    document_id: str
    version: int
    chunks_stored: int = 0
    chunks_deleted: int = 0
    duplicate_reason: str | None = None
    recovered: bool = False


class DocumentManagementService:
    """Coordinate ingestion, SQLite registration and ChromaDB storage."""

    def __init__(
        self,
        ingestion_pipeline: DocumentIngestionPipeline,
        registry: DocumentRegistry,
        vector_store: VectorStore,
    ) -> None:
        """Initialize the service with reusable project components."""

        self.ingestion_pipeline = ingestion_pipeline
        self.registry = registry
        self.vector_store = vector_store

    @staticmethod
    def _apply_registry_identity(
        result: IngestionResult,
        document_id: str,
        version: int,
    ) -> IngestionResult:
        """Replace temporary IDs with stable version-aware chunk IDs."""

        # Build a mapping so neighbor references can also be updated.
        chunk_id_mapping = {
            item.chunk.chunk_id: (
                f"{document_id}_v{version}_chunk_"
                f"{item.chunk.chunk_index:04d}"
            )
            for item in result.chunks
        }

        updated_chunks = []

        for item in result.chunks:
            original_chunk = item.chunk
            new_chunk_id = chunk_id_mapping[original_chunk.chunk_id]

            updated_chunk = replace(
                original_chunk,
                chunk_id=new_chunk_id,
                document_id=document_id,
                previous_chunk_id=chunk_id_mapping.get(
                    original_chunk.previous_chunk_id
                ),
                next_chunk_id=chunk_id_mapping.get(
                    original_chunk.next_chunk_id
                ),
            )

            updated_metadata = {
                **item.metadata,
                "document_id": document_id,
                "version": version,
                "chunk_id": new_chunk_id,
            }

            updated_chunks.append(
                replace(
                    item,
                    chunk=updated_chunk,
                    metadata=updated_metadata,
                )
            )

        return replace(
            result,
            document_id=document_id,
            chunks=updated_chunks,
            total_chunks=len(updated_chunks),
        )

    def _store_and_reconcile(
        self,
        result: IngestionResult,
    ) -> tuple[int, int]:
        """Store the current version and remove stale vectors.

        The operation is retryable because version-specific chunk IDs
        are deterministic.

        Returns:
            Number of current chunks stored and stale chunks deleted.
        """

        expected_ids = {
            item.chunk.chunk_id for item in result.chunks
        }

        if not expected_ids:
            raise ValueError("Cannot index a document without chunks.")

        # Store the complete current version before removing old vectors.
        try:
            stored_count = self.vector_store.upsert_document(result)

            if stored_count != len(expected_ids):
                raise RuntimeError(
                    "Stored chunk count does not match expected chunk count."
                )

            # Verify that all expected IDs really exist in ChromaDB.
            existing_ids = set(
                self.vector_store.get_document_chunk_ids(
                    result.document_id
                )
            )

            missing_ids = expected_ids - existing_ids

            if missing_ids:
                raise RuntimeError(
                    f"Vector storage is incomplete: {len(missing_ids)} "
                    "expected chunks are missing."
                )

        except Exception as error:
            # Remove partially stored current-version chunks where possible.
            try:
                self.vector_store.delete_chunks(list(expected_ids))
            except Exception:
                # A later retry will reconcile any remaining partial writes.
                pass

            raise RuntimeError(
                "The SQLite registry was updated, but vector indexing "
                "failed. The document can be repaired by retrying the "
                "same upload."
            ) from error

        # Keep only the current version after successful verification.
        stale_ids = list(existing_ids - expected_ids)
        deleted_count = 0

        if stale_ids:
            try:
                deleted_count = self.vector_store.delete_chunks(stale_ids)
            except Exception as error:
                raise RuntimeError(
                    "The current version was indexed, but stale vectors "
                    "could not be removed. Retry the same upload to "
                    "complete reconciliation."
                ) from error

        return stored_count, deleted_count

    def upload_document(
        self,
        file_path: str | Path,
    ) -> DocumentServiceResult:
        """Register, index or recover a document upload."""

        # Process the source file using the existing ingestion pipeline.
        ingestion_result = self.ingestion_pipeline.ingest(file_path)

        # SQLite remains the authority for document identity and version.
        registry_result = self.registry.register_or_update_document(
            filename=ingestion_result.filename,
            file_hash=ingestion_result.file_hash,
            content_hash=ingestion_result.content_hash,
        )

        # Identical content belonging to a different active document
        # should not create another document or another set of vectors.
        if (
            registry_result.status == "duplicate_content"
            and registry_result.duplicate_reason == "identical_content"
        ):
            return DocumentServiceResult(
                status=registry_result.status,
                document_id=registry_result.document_id,
                version=registry_result.version,
                duplicate_reason=registry_result.duplicate_reason,
            )

        # These statuses mean this document may already have been
        # registered during an earlier attempt.
        recovering = registry_result.status in {
            "duplicate_file",
            "duplicate_content",
        }

        # Rebuild the expected chunk IDs using the authoritative version.
        finalized_result = self._apply_registry_identity(
            result=ingestion_result,
            document_id=registry_result.document_id,
            version=registry_result.version,
        )

        # This also handles retries after a previous indexing failure.
        stored_count, deleted_count = self._store_and_reconcile(
            finalized_result
        )

        return DocumentServiceResult(
            status=registry_result.status,
            document_id=registry_result.document_id,
            version=registry_result.version,
            chunks_stored=stored_count,
            chunks_deleted=deleted_count,
            duplicate_reason=registry_result.duplicate_reason,
            recovered=recovering,
        )

    def delete_document(
        self,
        document_id: str,
    ) -> DocumentServiceResult:
        """Soft-delete a document and retry any outstanding vector cleanup."""

        # Inspect the registry before deciding whether deletion is needed.
        document = self.registry.get_document(document_id)

        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        if document["status"] == "active":
            # Soft-delete first; the registry preserves version history.
            registry_result = self.registry.delete_document(document_id)
            version = registry_result.version

        elif document["status"] == "deleted":
            # A previous attempt may have deleted SQLite state but failed
            # before removing the ChromaDB records.
            version = int(document["current_version"])

        else:
            raise RuntimeError(
                f"Unsupported document status: {document['status']}"
            )

        # Retrieve any remaining vectors, including stale version vectors.
        chunk_ids = self.vector_store.get_document_chunk_ids(document_id)

        try:
            deleted_count = self.vector_store.delete_chunks(chunk_ids)

            # Verify cleanup instead of assuming deletion succeeded.
            remaining_ids = self.vector_store.get_document_chunk_ids(
                document_id
            )

            if remaining_ids:
                raise RuntimeError(
                    f"{len(remaining_ids)} vector chunks remain after deletion."
                )

        except Exception as error:
            raise RuntimeError(
                "The document is marked deleted in SQLite, but vector "
                "cleanup is incomplete. Retry delete_document() to "
                "finish the cleanup."
            ) from error

        return DocumentServiceResult(
            status="deleted",
            document_id=document_id,
            version=version,
            chunks_deleted=deleted_count,
            recovered=document["status"] == "deleted",
        )
