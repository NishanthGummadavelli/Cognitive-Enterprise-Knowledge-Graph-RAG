
"""Coordinate scope-aware document lifecycle across all storage systems."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Protocol

from src.embeddings.vector_store import VectorStore
from src.extraction.entity_extractor import EntityExtractor
from src.extraction.schemas import ExtractionResult
from src.ingestion.pipeline import (
    DocumentIngestionPipeline,
    IngestionResult,
)
from src.knowledge_graph.graph_store import GraphStore
from src.storage.document_registry import DocumentRegistry


class AuthorizationService(Protocol):
    """Define the authorization operations required by document management."""

    def authorize_scope_access(
        self,
        user_id: str,
        scope_type: str,
        scope_id: str,
    ) -> None:
        """Verify that a user can upload documents into the requested scope."""

    def authorize_document_deletion(
        self,
        user_id: str,
        document: Mapping[str, object],
    ) -> None:
        """Verify that a user is allowed to delete the specified document."""


@dataclass
class DocumentServiceResult:
    """Describe the outcome of a document management operation."""

    status: str
    document_id: str
    version: int
    chunks_stored: int = 0
    chunks_deleted: int = 0
    graph_chunks_stored: int = 0
    graph_chunks_deleted: int = 0
    graph_entities_stored: int = 0
    graph_relationships_stored: int = 0
    duplicate_reason: str | None = None
    recovered: bool = False


class DocumentManagementService:
    """Coordinate ingestion, authorization and storage across all systems."""

    def __init__(
        self,
        ingestion_pipeline: DocumentIngestionPipeline,
        registry: DocumentRegistry,
        vector_store: VectorStore,
        graph_store: GraphStore,
        entity_extractor: EntityExtractor,
        authorization_service: AuthorizationService,
    ) -> None:
        """Initialize the service with all required components."""

        self.ingestion_pipeline = ingestion_pipeline
        self.registry = registry
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor
        self.authorization_service = authorization_service

    @staticmethod
    def _validate_scope_identity(
        user_id: str,
        scope_type: str,
        scope_id: str,
    ) -> None:
        """Validate the user and scope identity before processing a document."""

        if not all(
            isinstance(value, str) and value.strip()
            for value in (user_id, scope_type, scope_id)
        ):
            raise ValueError(
                "User ID, scope type and scope ID are required."
            )

        if scope_type not in {"personal", "project"}:
            raise ValueError(
                "Scope type must be either 'personal' or 'project'."
            )

        if scope_type == "personal" and scope_id != user_id:
            raise PermissionError(
                "A user can upload documents only into their own "
                "personal knowledge space."
            )

    @staticmethod
    def _apply_registry_identity(
        result: IngestionResult,
        document_id: str,
        version: int,
        scope_type: str,
        scope_id: str,
        uploaded_by: str,
    ) -> IngestionResult:
        """Apply stable IDs and scope metadata to every document chunk."""

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
                "scope_type": scope_type,
                "scope_id": scope_id,
                "uploaded_by": uploaded_by,
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

    def _extract_chunk_knowledge(
        self,
        result: IngestionResult,
    ) -> dict[str, ExtractionResult]:
        """Extract entities and relationships from every finalized chunk.

        Returns:
            Extraction results keyed by stable chunk ID.
        """

        if not result.chunks:
            raise ValueError(
                "Cannot extract knowledge without document chunks."
            )

        extraction_results = {}

        for item in result.chunks:
            chunk = item.chunk

            try:
                extraction = self.entity_extractor.extract(chunk.text)

                if not isinstance(extraction, ExtractionResult):
                    raise TypeError(
                        "Entity extractor must return an ExtractionResult."
                    )

                extraction_results[chunk.chunk_id] = extraction

            except Exception as error:
                raise RuntimeError(
                    "Entity extraction failed for chunk "
                    f"'{chunk.chunk_id}'. The document was registered in "
                    "SQLite but has not been indexed in ChromaDB or Neo4j. "
                    "Retry the same upload after resolving the extraction "
                    "problem."
                ) from error

        return extraction_results

    def _store_and_reconcile(
        self,
        result: IngestionResult,
    ) -> tuple[int, int]:
        """Store the current version in ChromaDB and remove stale vectors.

        Returns:
            Number of current chunks stored and stale chunks deleted.
        """

        expected_ids = {
            item.chunk.chunk_id for item in result.chunks
        }

        if not expected_ids:
            raise ValueError("Cannot index a document without chunks.")

        try:
            stored_count = self.vector_store.upsert_document(result)

            if stored_count != len(expected_ids):
                raise RuntimeError(
                    "Stored chunk count does not match expected chunk count."
                )

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
            try:
                self.vector_store.delete_chunks(list(expected_ids))
            except Exception:
                pass

            raise RuntimeError(
                "The SQLite registry was updated, but vector indexing "
                "failed. Retry the same upload to repair the document."
            ) from error

        stale_ids = list(existing_ids - expected_ids)
        deleted_count = 0

        if stale_ids:
            try:
                deleted_count = self.vector_store.delete_chunks(stale_ids)
            except Exception as error:
                raise RuntimeError(
                    "The current version was indexed, but stale vectors "
                    "could not be removed. Retry the same upload."
                ) from error

        return stored_count, deleted_count

    def _store_graph_and_reconcile(
        self,
        result: IngestionResult,
        extraction_results: Mapping[str, ExtractionResult],
    ) -> tuple[int, int, int, int]:
        """Store the document and its extracted knowledge in Neo4j.

        Returns:
            Stored chunks, deleted chunks, stored entities and
            stored knowledge relationships.
        """

        try:
            graph_result = self.graph_store.upsert_document(
                result,
                extraction_results=extraction_results,
            )

            expected_count = len(result.chunks)

            if graph_result["chunks_stored"] != expected_count:
                raise RuntimeError(
                    "Graph chunk count does not match expected chunk count."
                )

            return (
                graph_result["chunks_stored"],
                graph_result["chunks_deleted"],
                graph_result["entities_stored"],
                graph_result["relationships_stored"],
            )

        except Exception as error:
            raise RuntimeError(
                "SQLite and ChromaDB indexing succeeded, but Neo4j graph "
                "indexing failed. Retry the same upload to repair the graph."
            ) from error

    def upload_document(
        self,
        file_path: str | Path,
        user_id: str,
        scope_type: str,
        scope_id: str,
    ) -> DocumentServiceResult:
        """Authorize, register, extract and index a document in its scope."""

        self._validate_scope_identity(
            user_id=user_id,
            scope_type=scope_type,
            scope_id=scope_id,
        )

        # Verify membership or ownership before processing the source file.
        self.authorization_service.authorize_scope_access(
            user_id,
            scope_type,
            scope_id,
        )

        ingestion_result = self.ingestion_pipeline.ingest(file_path)

        # SQLite remains the authority for document identity and version.
        registry_result = self.registry.register_or_update_document(
            filename=ingestion_result.filename,
            file_hash=ingestion_result.file_hash,
            content_hash=ingestion_result.content_hash,
            scope_type=scope_type,
            scope_id=scope_id,
            uploaded_by=user_id,
        )

        # Identical content in this scope already has a document identity.
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

        recovering = registry_result.status in {
            "duplicate_file",
            "duplicate_content",
        }

        finalized_result = self._apply_registry_identity(
            result=ingestion_result,
            document_id=registry_result.document_id,
            version=registry_result.version,
            scope_type=scope_type,
            scope_id=scope_id,
            uploaded_by=user_id,
        )

        extraction_results = self._extract_chunk_knowledge(
            finalized_result
        )

        stored_count, deleted_count = self._store_and_reconcile(
            finalized_result
        )

        (
            graph_stored,
            graph_deleted,
            graph_entities,
            graph_relationships,
        ) = self._store_graph_and_reconcile(
            finalized_result,
            extraction_results,
        )

        return DocumentServiceResult(
            status=registry_result.status,
            document_id=registry_result.document_id,
            version=registry_result.version,
            chunks_stored=stored_count,
            chunks_deleted=deleted_count,
            graph_chunks_stored=graph_stored,
            graph_chunks_deleted=graph_deleted,
            graph_entities_stored=graph_entities,
            graph_relationships_stored=graph_relationships,
            duplicate_reason=registry_result.duplicate_reason,
            recovered=recovering,
        )

    def delete_document(
        self,
        document_id: str,
        requested_by: str,
    ) -> DocumentServiceResult:
        """Authorize and delete a document from its registered scope."""

        if not isinstance(requested_by, str) or not requested_by.strip():
            raise ValueError("The requesting user ID is required.")

        document = self.registry.get_document(document_id)

        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        scope_type = document["scope_type"]
        scope_id = document["scope_id"]

        # Legacy records cannot be deleted through normal scoped access.
        if scope_type not in {"personal", "project"}:
            raise PermissionError(
                "This document has no assigned personal or project scope."
            )

        # Authorization must happen before changing registry or storage data.
        self.authorization_service.authorize_document_deletion(
            requested_by,
            document,
        )

        if document["status"] == "active":
            registry_result = self.registry.delete_document(
                document_id,
                scope_type=scope_type,
                scope_id=scope_id,
            )
            version = registry_result.version

        elif document["status"] == "deleted":
            # A previous attempt may have stopped during storage cleanup.
            version = int(document["current_version"])

        else:
            raise RuntimeError(
                f"Unsupported document status: {document['status']}"
            )

        try:
            chunk_ids = self.vector_store.get_document_chunk_ids(document_id)

            deleted_count = self.vector_store.delete_chunks(chunk_ids)

            remaining_ids = self.vector_store.get_document_chunk_ids(
                document_id
            )

            if remaining_ids:
                raise RuntimeError(
                    f"{len(remaining_ids)} vector chunks remain after deletion."
                )

        except Exception as error:
            raise RuntimeError(
                "The document is marked deleted in SQLite, but ChromaDB "
                "cleanup is incomplete. Retry delete_document()."
            ) from error

        try:
            graph_deleted_count = self.graph_store.delete_document(
                document_id
            )

        except Exception as error:
            raise RuntimeError(
                "SQLite deletion and ChromaDB cleanup succeeded, but "
                "Neo4j cleanup failed. Retry delete_document() to "
                "finish graph cleanup."
            ) from error

        return DocumentServiceResult(
            status="deleted",
            document_id=document_id,
            version=version,
            chunks_deleted=deleted_count,
            graph_chunks_deleted=graph_deleted_count,
            recovered=document["status"] == "deleted",
        )
