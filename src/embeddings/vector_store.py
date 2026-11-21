
"""Store and retrieve BGE embeddings using a persistent ChromaDB collection."""

import json
from pathlib import Path
from typing import Any

import chromadb

from src.embeddings.encoder import EmbeddingEncoder
from src.ingestion.pipeline import IngestionResult


# Resolve the project directory independently of the terminal location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Store the vector database locally instead of keeping it in memory.
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

# Use a consistent collection name throughout the project.
DEFAULT_COLLECTION_NAME = "enterprise_documents"


class VectorStore:
    """Manage persistent document embeddings and semantic retrieval."""

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

        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        self.encoder = encoder

        try:
            # Create the database directory if it does not exist.
            db_path.mkdir(parents=True, exist_ok=True)

            # Connect to persistent local vector storage.
            self.client = chromadb.PersistentClient(
                path=str(db_path)
            )

            # Use cosine distance for semantic similarity search.
            # Supply our own BGE vectors instead of Chroma's default model.
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
            # Serialize structured lists as JSON strings.
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
            ValueError: If the ingestion result has no chunks.
            RuntimeError: If embedding or storage fails.
        """

        if not result.chunks:
            raise ValueError("Cannot store a document without chunks.")

        # Extract canonical chunk IDs and their original text.
        chunk_ids = [
            item.chunk.chunk_id for item in result.chunks
        ]

        documents = [
            item.chunk.text for item in result.chunks
        ]

        # Prepare metadata for each corresponding chunk.
        metadatas = [
            self._prepare_metadata(item.metadata)
            for item in result.chunks
        ]

        try:
            # Generate BGE vectors for all chunks in one operation.
            embeddings = self.encoder.encode_documents(documents)

            # Store vectors, text, metadata and stable chunk IDs together.
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

    def upsert_document(self, result: IngestionResult) -> int:
        """Insert or replace all chunks belonging to an ingested document.

        Args:
            result: Document processing output with finalized chunk IDs.

        Returns:
            Number of chunks stored.

        Raises:
            ValueError: If the document has no chunks.
            RuntimeError: If embedding or storage fails.
        """

        if not result.chunks:
            raise ValueError("Cannot store a document without chunks.")

        # Extract chunk IDs, text and metadata in matching order.
        chunk_ids = [item.chunk.chunk_id for item in result.chunks]
        documents = [item.chunk.text for item in result.chunks]
        metadatas = [
            self._prepare_metadata(item.metadata)
            for item in result.chunks
        ]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunk IDs must be unique within a document.")

        try:
            # Generate embeddings before changing the stored records.
            embeddings = self.encoder.encode_documents(documents)

            # Upsert makes retries safe for the same version-specific IDs.
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

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Delete a selected set of chunk IDs from ChromaDB.

        Args:
            chunk_ids: IDs of the chunks to remove.

        Returns:
            Number of IDs submitted for deletion.
        """

        if not chunk_ids:
            return 0

        if any(not isinstance(item, str) or not item.strip()
               for item in chunk_ids):
            raise ValueError("All chunk IDs must be non-empty strings.")

        try:
            # Remove only the explicitly supplied chunks.
            self.collection.delete(ids=chunk_ids)

        except Exception as error:
            raise RuntimeError(
                f"Could not delete document chunks: {error}"
            ) from error

        return len(chunk_ids)

    def get_document_chunk_ids(self, document_id: str) -> list[str]:
        """Return all stored chunk IDs belonging to one document.

        Args:
            document_id: Stable registry document identifier.

        Returns:
            IDs of all matching document chunks.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        try:
            # Retrieve IDs using the shared document metadata.
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
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant chunks for a user query.

        Args:
            query: Natural-language search question.
            top_k: Maximum number of results to retrieve.

        Returns:
            Ranked chunks with text, metadata and distance.

        Raises:
            ValueError: If the query or result count is invalid.
            RuntimeError: If semantic retrieval fails.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.collection.count() == 0:
            return []

        try:
            # Convert the question into a BGE query vector.
            query_vector = self.encoder.encode_query(query)

            # Ask ChromaDB for the nearest stored vectors.
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        except Exception as error:
            raise RuntimeError(
                f"Semantic search failed: {error}"
            ) from error

        # Convert ChromaDB's grouped result format into simple records.
        retrieved_chunks = []

        for index, chunk_id in enumerate(results["ids"][0]):
            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index],
                }
            )

        return retrieved_chunks

    def count(self) -> int:
        """Return the total number of stored document chunks."""

        return self.collection.count()
