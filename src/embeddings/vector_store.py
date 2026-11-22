
"""Store and retrieve BGE embeddings using scope-aware ChromaDB search."""

import json
from pathlib import Path
from typing import Any, Mapping

import chromadb

from src.embeddings.encoder import EmbeddingEncoder
from src.ingestion.pipeline import IngestionResult


# Resolve the project directory independently of the terminal location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Store the vector database locally instead of keeping it in memory.
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

# Use a consistent collection name throughout the project.
DEFAULT_COLLECTION_NAME = "enterprise_documents"

VALID_SCOPE_TYPES = {"personal", "project"}


class VectorStore:
    """Manage persistent document embeddings and scope-aware retrieval."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        db_path: Path = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize the persistent ChromaDB collection.

        Args:
            encoder: The BGE encoder used to generate embeddings.
            db_path: Directory where ChromaDB stores its data.
            collection_name: Name of the document collection.

        Raises:
            ValueError: If the collection name is empty.
            RuntimeError: If ChromaDB initialization fails.
        """

        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        self.encoder = encoder

        try:
            db_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(path=str(db_path))

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )

        except Exception as error:
            raise RuntimeError(
                f"Could not initialize ChromaDB: {error}"
            ) from error

    @staticmethod
    def _validate_scope(
        scope_type: str,
        scope_id: str,
    ) -> None:
        """Validate the scope required for a retrieval operation.

        Args:
            scope_type: Either personal or project.
            scope_id: User ID for personal scope or project ID for project scope.

        Raises:
            ValueError: If either scope value is invalid.
        """

        if not isinstance(scope_type, str) or scope_type not in VALID_SCOPE_TYPES:
            raise ValueError(
                "scope_type must be either 'personal' or 'project'."
            )

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("scope_id must be a non-empty string.")

    @classmethod
    def _validate_chunk_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> None:
        """Validate the authorization metadata attached to a stored chunk.

        Args:
            metadata: Metadata belonging to one ingested chunk.

        Raises:
            ValueError: If scope or uploader information is missing or invalid.
        """

        scope_type = metadata.get("scope_type")
        scope_id = metadata.get("scope_id")
        uploaded_by = metadata.get("uploaded_by")

        if not isinstance(scope_type, str):
            raise ValueError("Chunk metadata must contain scope_type.")

        if not isinstance(scope_id, str):
            raise ValueError("Chunk metadata must contain scope_id.")

        if not isinstance(uploaded_by, str) or not uploaded_by.strip():
            raise ValueError("Chunk metadata must contain uploaded_by.")

        cls._validate_scope(scope_type, scope_id)

        if scope_type == "personal" and scope_id != uploaded_by:
            raise ValueError(
                "For personal scope, scope_id must match uploaded_by."
            )

    @classmethod
    def _validate_ingestion_scope(
        cls,
        result: IngestionResult,
    ) -> None:
        """Ensure all chunks in a document belong to one valid scope.

        Args:
            result: Ingestion output whose chunks will be stored.

        Raises:
            ValueError: If chunks are missing or have inconsistent scopes.
        """

        if not result.chunks:
            raise ValueError("Cannot store a document without chunks.")

        first_metadata = result.chunks[0].metadata
        cls._validate_chunk_metadata(first_metadata)

        expected_scope = (
            first_metadata["scope_type"],
            first_metadata["scope_id"],
            first_metadata["uploaded_by"],
        )

        for item in result.chunks:
            cls._validate_chunk_metadata(item.metadata)

            actual_scope = (
                item.metadata["scope_type"],
                item.metadata["scope_id"],
                item.metadata["uploaded_by"],
            )

            if actual_scope != expected_scope:
                raise ValueError(
                    "All chunks in a document must have identical scope metadata."
                )

            if item.metadata.get("document_id") != result.document_id:
                raise ValueError(
                    "Chunk metadata document_id does not match the ingestion result."
                )

    @staticmethod
    def _prepare_metadata(
        metadata: dict[str, object],
    ) -> dict[str, str | int | float | bool]:
        """Convert chunk metadata into ChromaDB-compatible values.

        Args:
            metadata: Original document and chunk metadata.

        Returns:
            Metadata containing supported primitive values.
        """

        prepared: dict[str, str | int | float | bool] = {}

        for key, value in metadata.items():
            if isinstance(value, (list, tuple, dict)):
                prepared[key] = json.dumps(value)

            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = value

            elif value is not None:
                prepared[key] = str(value)

        return prepared

    def add_document(
        self,
        result: IngestionResult,
    ) -> int:
        """Generate and store embeddings for an ingested document.

        Args:
            result: Output produced by the ingestion pipeline.

        Returns:
            Number of chunks successfully submitted to ChromaDB.

        Raises:
            ValueError: If chunks or scope metadata are invalid.
            RuntimeError: If embedding or storage fails.
        """

        self._validate_ingestion_scope(result)

        chunk_ids = [
            item.chunk.chunk_id for item in result.chunks
        ]

        documents = [
            item.chunk.text for item in result.chunks
        ]

        metadatas = [
            self._prepare_metadata(item.metadata)
            for item in result.chunks
        ]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunk IDs must be unique within a document.")

        try:
            embeddings = self.encoder.encode_documents(documents)

            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )

        except Exception as error:
            raise RuntimeError(
                f"Could not store document embeddings: {error}"
            ) from error

        return len(chunk_ids)

    def upsert_document(
        self,
        result: IngestionResult,
    ) -> int:
        """Insert or replace all chunks belonging to an ingested document.

        Args:
            result: Document processing output with finalized chunk IDs.

        Returns:
            Number of chunks stored.

        Raises:
            ValueError: If chunks or scope metadata are invalid.
            RuntimeError: If embedding or storage fails.
        """

        self._validate_ingestion_scope(result)

        chunk_ids = [
            item.chunk.chunk_id for item in result.chunks
        ]

        documents = [
            item.chunk.text for item in result.chunks
        ]

        metadatas = [
            self._prepare_metadata(item.metadata)
            for item in result.chunks
        ]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunk IDs must be unique within a document.")

        try:
            embeddings = self.encoder.encode_documents(documents)

            self.collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )

        except Exception as error:
            raise RuntimeError(
                f"Could not upsert document embeddings: {error}"
            ) from error

        return len(chunk_ids)

    def delete_chunks(
        self,
        chunk_ids: list[str],
    ) -> int:
        """Delete a selected set of chunk IDs from ChromaDB.

        Args:
            chunk_ids: IDs of the chunks to remove.

        Returns:
            Number of IDs submitted for deletion.
        """

        if not chunk_ids:
            return 0

        if any(
            not isinstance(item, str) or not item.strip()
            for item in chunk_ids
        ):
            raise ValueError("All chunk IDs must be non-empty strings.")

        try:
            self.collection.delete(ids=chunk_ids)

        except Exception as error:
            raise RuntimeError(
                f"Could not delete document chunks: {error}"
            ) from error

        return len(chunk_ids)

    def get_document_chunk_ids(
        self,
        document_id: str,
    ) -> list[str]:
        """Return all stored chunk IDs belonging to one document.

        Args:
            document_id: Stable registry document identifier.

        Returns:
            IDs of all matching document chunks.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        try:
            records = self.collection.get(
                where={"document_id": document_id.strip()},
                include=["metadatas"],
            )

            return records["ids"]

        except Exception as error:
            raise RuntimeError(
                f"Could not retrieve document chunk IDs: {error}"
            ) from error

    def search(
        self,
        query: str,
        top_k: int = 5,
        scope_type: str | None = None,
        scope_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve semantically relevant chunks from one authorized scope.

        Both scope arguments are mandatory. This method applies the scope
        filter inside ChromaDB instead of retrieving globally and filtering
        the results afterward.

        Args:
            query: Natural-language search question.
            top_k: Maximum number of results to retrieve.
            scope_type: Personal or project.
            scope_id: User ID for personal scope or project ID for project scope.

        Returns:
            Ranked chunks containing text, metadata and cosine distance.

        Raises:
            ValueError: If the query, result count or scope is invalid.
            RuntimeError: If semantic retrieval fails.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if scope_type is None or scope_id is None:
            raise ValueError(
                "Both scope_type and scope_id are required for retrieval."
            )

        self._validate_scope(scope_type, scope_id)

        if self.collection.count() == 0:
            return []

        # Restrict ChromaDB's search to the requested knowledge space.
        scope_filter = {
            "$and": [
                {"scope_type": scope_type},
                {"scope_id": scope_id},
            ]
        }

        try:
            query_vector = self.encoder.encode_query(query.strip())

            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=int(min(top_k, self.collection.count())),
                where=scope_filter,
                include=["documents", "metadatas", "distances"],
            )

        except Exception as error:
            raise RuntimeError(
                f"Semantic search failed: {error}"
            ) from error

        retrieved_chunks: list[dict[str, Any]] = []

        result_ids = results.get("ids", [[]])
        if not result_ids:
            return []

        for index, chunk_id in enumerate(result_ids[0]):
            metadata = results["metadatas"][0][index]

            # Defensive validation: do not return a result outside the
            # requested scope, even if unexpected data is encountered.
            if (
                metadata.get("scope_type") != scope_type
                or metadata.get("scope_id") != scope_id
            ):
                continue

            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][index],
                    "metadata": metadata,
                    "distance": results["distances"][0][index],
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                }
            )

        return retrieved_chunks

    def count(self) -> int:
        """Return the total number of stored document chunks."""

        return self.collection.count()
