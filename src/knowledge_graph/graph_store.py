
"""Manage enterprise documents and chunks in the Neo4j knowledge graph."""

from typing import Any

from neo4j import ManagedTransaction

from src.ingestion.pipeline import IngestionResult
from src.knowledge_graph.connection import Neo4jConnection


class GraphStore:
    """Store and manage document and chunk nodes in Neo4j."""

    def __init__(self, connection: Neo4jConnection) -> None:
        """Initialize the graph store with an existing Neo4j connection."""

        self.connection = connection

    @staticmethod
    def _build_document_payload(
        result: IngestionResult,
    ) -> dict[str, Any]:
        """Convert an ingestion result into Neo4j-compatible properties."""

        if not result.chunks:
            raise ValueError("Cannot store a document without chunks.")

        # The service assigns the authoritative version to chunk metadata.
        version = int(result.chunks[0].metadata.get("version", 1))

        chunks = []

        for item in result.chunks:
            chunk = item.chunk

            chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": result.document_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "page_numbers": list(chunk.page_numbers),
                    "heading_path": list(chunk.heading_path),
                    "previous_chunk_id": chunk.previous_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                }
            )

        return {
            "document": {
                "document_id": result.document_id,
                "filename": result.filename,
                "category": result.predicted_category,
                "classification_confidence": (
                    result.classification_confidence
                ),
                "file_hash": result.file_hash,
                "content_hash": result.content_hash,
                "version": version,
                "total_characters": result.total_characters,
                "total_chunks": result.total_chunks,
                "status": "active",
            },
            "chunks": chunks,
        }

    @staticmethod
    def _write_document(
        transaction: ManagedTransaction,
        payload: dict[str, Any],
    ) -> dict[str, int]:
        """Create or update a document and reconcile its chunk nodes."""

        document = payload["document"]
        chunks = payload["chunks"]
        document_id = document["document_id"]

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        # Create or update the document without duplicating its identity.
        transaction.run(
            """
            MERGE (d:Document {document_id: $document_id})
            SET d.filename = $filename,
                d.category = $category,
                d.classification_confidence = $confidence,
                d.file_hash = $file_hash,
                d.content_hash = $content_hash,
                d.version = $version,
                d.total_characters = $total_characters,
                d.total_chunks = $total_chunks,
                d.status = $status
            """,
            document_id=document_id,
            filename=document["filename"],
            category=document["category"],
            confidence=document["classification_confidence"],
            file_hash=document["file_hash"],
            content_hash=document["content_hash"],
            version=document["version"],
            total_characters=document["total_characters"],
            total_chunks=document["total_chunks"],
            status=document["status"],
        ).consume()

        # Create or update every canonical chunk and its document link.
        transaction.run(
            """
            MATCH (d:Document {document_id: $document_id})
            UNWIND $chunks AS item

            MERGE (c:Chunk {chunk_id: item.chunk_id})
            SET c.document_id = item.document_id,
                c.text = item.text,
                c.source = item.source,
                c.chunk_index = item.chunk_index,
                c.page_numbers = item.page_numbers,
                c.heading_path = item.heading_path

            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            document_id=document_id,
            chunks=chunks,
        ).consume()

        # Remove old sequential links before rebuilding them.
        transaction.run(
            """
            MATCH (c:Chunk {document_id: $document_id})
            OPTIONAL MATCH (c)-[r:NEXT_CHUNK]->()
            DELETE r
            """,
            document_id=document_id,
        ).consume()

        # Recreate links between consecutive chunks in the same document.
        transaction.run(
            """
            UNWIND $chunks AS item
            WITH item
            WHERE item.next_chunk_id IS NOT NULL

            MATCH (current:Chunk {chunk_id: item.chunk_id})
            MATCH (next:Chunk {chunk_id: item.next_chunk_id})

            MERGE (current)-[:NEXT_CHUNK]->(next)
            """,
            chunks=chunks,
        ).consume()

        # Remove chunks belonging to older versions after the new version
        # has been written successfully within this transaction.
        stale_result = transaction.run(
            """
            MATCH (d:Document {document_id: $document_id})
                  -[:HAS_CHUNK]->(old:Chunk)
            WHERE NOT old.chunk_id IN $chunk_ids

            WITH collect(old) AS stale_chunks
            FOREACH (chunk IN stale_chunks | DETACH DELETE chunk)

            RETURN size(stale_chunks) AS deleted_count
            """,
            document_id=document_id,
            chunk_ids=chunk_ids,
        ).single()

        deleted_count = (
            stale_result["deleted_count"] if stale_result else 0
        )

        return {
            "chunks_stored": len(chunks),
            "chunks_deleted": deleted_count,
        }

    def upsert_document(
        self,
        result: IngestionResult,
    ) -> dict[str, int]:
        """Create or update a document and its chunks atomically in Neo4j.

        Repeated calls with the same stable IDs are safe because nodes and
        relationships are created using MERGE.

        Returns:
            Counts of stored and removed chunks.
        """

        payload = self._build_document_payload(result)

        with self.connection.driver.session(
            database=self.connection.database
        ) as session:
            return session.execute_write(
                self._write_document,
                payload,
            )

    @staticmethod
    def _delete_document(
        transaction: ManagedTransaction,
        document_id: str,
    ) -> int:
        """Delete a document and all chunks directly belonging to it."""

        result = transaction.run(
            """
            MATCH (d:Document {document_id: $document_id})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)

            WITH d, collect(c) AS chunks
            FOREACH (chunk IN chunks | DETACH DELETE chunk)

            WITH d, size(chunks) AS deleted_count
            DETACH DELETE d

            RETURN deleted_count
            """,
            document_id=document_id,
        ).single()

        if result is None:
            return 0

        return result["deleted_count"]

    def delete_document(self, document_id: str) -> int:
        """Delete a document and its associated chunks from Neo4j.

        Returns:
            Number of chunk nodes deleted.
        """

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        with self.connection.driver.session(
            database=self.connection.database
        ) as session:
            return session.execute_write(
                self._delete_document,
                document_id,
            )

    def count_documents(self) -> int:
        """Return the number of Document nodes in the graph."""

        with self.connection.driver.session(
            database=self.connection.database
        ) as session:
            result = session.run(
                "MATCH (d:Document) RETURN count(d) AS total"
            ).single()

            return result["total"] if result else 0

    def count_chunks(self) -> int:
        """Return the number of Chunk nodes in the graph."""

        with self.connection.driver.session(
            database=self.connection.database
        ) as session:
            result = session.run(
                "MATCH (c:Chunk) RETURN count(c) AS total"
            ).single()

            return result["total"] if result else 0
