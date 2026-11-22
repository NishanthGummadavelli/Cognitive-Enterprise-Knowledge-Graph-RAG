
"""Manage enterprise documents, chunks and extracted knowledge in Neo4j."""

import hashlib
from typing import Any, Mapping

from neo4j import ManagedTransaction

from src.extraction.schemas import ExtractionResult
from src.ingestion.pipeline import IngestionResult
from src.knowledge_graph.connection import Neo4jConnection


class GraphStore:
    """Store and manage scope-isolated documents, chunks and knowledge in Neo4j."""

    def __init__(self, connection: Neo4jConnection) -> None:
        """Initialize the graph store with an existing Neo4j connection."""

        self.connection = connection

    @staticmethod
    def _generate_entity_id(
        name: str,
        entity_type: str,
        scope_type: str = "legacy",
        scope_id: str = "legacy_unassigned",
    ) -> str:
        """Generate a deterministic entity ID within one knowledge scope."""

        normalized_name = " ".join(name.casefold().split())
        identity = (
            f"{scope_type}::{scope_id}::"
            f"{entity_type.upper()}::{normalized_name}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_relationship_id(
        chunk_id: str,
        source_entity_id: str,
        relationship_type: str,
        target_entity_id: str,
        evidence: str,
    ) -> str:
        """Generate a deterministic ID for a relationship evidence record."""

        identity = "|".join(
            [
                chunk_id,
                source_entity_id,
                relationship_type.upper(),
                target_entity_id,
                " ".join(evidence.split()),
            ]
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _get_scope_metadata(result: IngestionResult) -> dict[str, str]:
        """Read and validate authoritative scope metadata from ingested chunks."""

        if not result.chunks:
            raise ValueError("Cannot determine scope without document chunks.")

        metadata = result.chunks[0].metadata
        scope_type = metadata.get("scope_type")
        scope_id = metadata.get("scope_id")
        uploaded_by = metadata.get("uploaded_by")

        if scope_type not in {"personal", "project"}:
            raise ValueError("Document chunks must have a valid scope_type.")

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("Document chunks must have a non-empty scope_id.")

        if not isinstance(uploaded_by, str) or not uploaded_by.strip():
            raise ValueError("Document chunks must have a non-empty uploaded_by.")

        if scope_type == "personal" and scope_id != uploaded_by:
            raise ValueError(
                "Personal document scope_id must match uploaded_by."
            )

        expected = {
            "scope_type": scope_type,
            "scope_id": scope_id,
            "uploaded_by": uploaded_by,
        }

        for item in result.chunks:
            chunk_metadata = item.metadata

            actual = {
                "scope_type": chunk_metadata.get("scope_type"),
                "scope_id": chunk_metadata.get("scope_id"),
                "uploaded_by": chunk_metadata.get("uploaded_by"),
            }

            if actual != expected:
                raise ValueError(
                    "All chunks in a document must have identical scope metadata."
                )

        return expected

    @classmethod
    def _build_knowledge_payload(
        cls,
        result: IngestionResult,
        extraction_results: Mapping[str, ExtractionResult] | None,
        scope: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        """Convert extraction results into scope-specific Neo4j records."""

        if extraction_results is None:
            return []

        chunk_ids = {item.chunk.chunk_id for item in result.chunks}
        unknown_chunk_ids = set(extraction_results) - chunk_ids

        if unknown_chunk_ids:
            raise ValueError(
                "Extraction results contain unknown chunk IDs: "
                f"{sorted(unknown_chunk_ids)}"
            )

        knowledge = []

        for chunk_id, extraction in extraction_results.items():
            entities = []
            entity_ids = {}

            for entity in extraction.entities:
                entity_id = cls._generate_entity_id(
                    entity.name,
                    entity.entity_type,
                    scope["scope_type"],
                    scope["scope_id"],
                )

                entity_ids[entity.name] = entity_id

                entities.append(
                    {
                        "entity_id": entity_id,
                        "name": entity.name,
                        "entity_type": entity.entity_type.upper(),
                        "normalized_name": " ".join(
                            entity.name.casefold().split()
                        ),
                        "scope_type": scope["scope_type"],
                        "scope_id": scope["scope_id"],
                        "uploaded_by": scope["uploaded_by"],
                    }
                )

            relationships = []

            for relationship in extraction.relationships:
                source_entity_id = entity_ids[relationship.source]
                target_entity_id = entity_ids[relationship.target]
                relationship_type = relationship.relationship_type.upper()

                relationship_id = cls._generate_relationship_id(
                    chunk_id=chunk_id,
                    source_entity_id=source_entity_id,
                    relationship_type=relationship_type,
                    target_entity_id=target_entity_id,
                    evidence=relationship.evidence,
                )

                relationships.append(
                    {
                        "relationship_id": relationship_id,
                        "relationship_type": relationship_type,
                        "evidence": relationship.evidence,
                        "source_entity_id": source_entity_id,
                        "target_entity_id": target_entity_id,
                        "scope_type": scope["scope_type"],
                        "scope_id": scope["scope_id"],
                        "uploaded_by": scope["uploaded_by"],
                    }
                )

            knowledge.append(
                {
                    "chunk_id": chunk_id,
                    "entities": entities,
                    "relationships": relationships,
                }
            )

        return knowledge

    @classmethod
    def _build_document_payload(
        cls,
        result: IngestionResult,
        extraction_results: Mapping[str, ExtractionResult] | None = None,
    ) -> dict[str, Any]:
        """Convert ingestion and extraction results into a scope-aware payload."""

        if not result.chunks:
            raise ValueError("Cannot store a document without chunks.")

        scope = cls._get_scope_metadata(result)

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
                    **scope,
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
                **scope,
            },
            "chunks": chunks,
            "knowledge": cls._build_knowledge_payload(
                result,
                extraction_results,
                scope,
            ),
        }

    @staticmethod
    def _write_document(
        transaction: ManagedTransaction,
        payload: dict[str, Any],
    ) -> dict[str, int]:
        """Create or update one scoped document and reconcile its graph data."""

        document = payload["document"]
        chunks = payload["chunks"]
        knowledge = payload["knowledge"]
        document_id = document["document_id"]
        scope_type = document["scope_type"]
        scope_id = document["scope_id"]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]

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
                d.status = $status,
                d.scope_type = $scope_type,
                d.scope_id = $scope_id,
                d.uploaded_by = $uploaded_by
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
            scope_type=scope_type,
            scope_id=scope_id,
            uploaded_by=document["uploaded_by"],
        ).consume()

        # Remove the previous extracted evidence attached to this document.
        transaction.run(
            """
            MATCH (c:Chunk {document_id: $document_id})
            OPTIONAL MATCH (c)-[:SUPPORTS]->(old:KnowledgeRelationship)
            WITH collect(DISTINCT old) AS old_relationships
            FOREACH (
                relationship IN old_relationships |
                DETACH DELETE relationship
            )
            """,
            document_id=document_id,
        ).consume()

        transaction.run(
            """
            MATCH (c:Chunk {document_id: $document_id})
            OPTIONAL MATCH (c)-[mention:MENTIONS]->()
            DELETE mention
            """,
            document_id=document_id,
        ).consume()

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
                c.heading_path = item.heading_path,
                c.scope_type = item.scope_type,
                c.scope_id = item.scope_id,
                c.uploaded_by = item.uploaded_by
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            document_id=document_id,
            chunks=chunks,
        ).consume()

        transaction.run(
            """
            MATCH (c:Chunk {document_id: $document_id})
            OPTIONAL MATCH (c)-[r:NEXT_CHUNK]->()
            DELETE r
            """,
            document_id=document_id,
        ).consume()

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

        deleted_count = stale_result["deleted_count"] if stale_result else 0

        for item in knowledge:
            transaction.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                UNWIND $entities AS entity
                MERGE (e:Entity {entity_id: entity.entity_id})
                SET e.name = entity.name,
                    e.entity_type = entity.entity_type,
                    e.normalized_name = entity.normalized_name,
                    e.scope_type = entity.scope_type,
                    e.scope_id = entity.scope_id,
                    e.uploaded_by = entity.uploaded_by
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=item["chunk_id"],
                entities=item["entities"],
            ).consume()

            transaction.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                UNWIND $relationships AS item
                MATCH (source:Entity {entity_id: item.source_entity_id})
                MATCH (target:Entity {entity_id: item.target_entity_id})
                MERGE (
                    evidence:KnowledgeRelationship {
                        relationship_id: item.relationship_id
                    }
                )
                SET evidence.relationship_type = item.relationship_type,
                    evidence.evidence = item.evidence,
                    evidence.document_id = c.document_id,
                    evidence.chunk_id = c.chunk_id,
                    evidence.scope_type = item.scope_type,
                    evidence.scope_id = item.scope_id,
                    evidence.uploaded_by = item.uploaded_by
                MERGE (c)-[:SUPPORTS]->(evidence)
                MERGE (evidence)-[:SOURCE]->(source)
                MERGE (evidence)-[:TARGET]->(target)
                """,
                chunk_id=item["chunk_id"],
                relationships=item["relationships"],
            ).consume()

        # Only prune orphan entities belonging to this scope.
        transaction.run(
            """
            MATCH (e:Entity {
                scope_type: $scope_type,
                scope_id: $scope_id
            })
            WHERE NOT EXISTS {
                MATCH (:Chunk)-[:MENTIONS]->(e)
            }
            DETACH DELETE e
            """,
            scope_type=scope_type,
            scope_id=scope_id,
        ).consume()

        entity_result = transaction.run(
            """
            MATCH (c:Chunk {
                document_id: $document_id,
                scope_type: $scope_type,
                scope_id: $scope_id
            })-[:MENTIONS]->(e:Entity)
            RETURN count(DISTINCT e) AS total
            """,
            document_id=document_id,
            scope_type=scope_type,
            scope_id=scope_id,
        ).single()

        relationship_result = transaction.run(
            """
            MATCH (c:Chunk {
                document_id: $document_id,
                scope_type: $scope_type,
                scope_id: $scope_id
            })-[:SUPPORTS]->(r:KnowledgeRelationship)
            RETURN count(DISTINCT r) AS total
            """,
            document_id=document_id,
            scope_type=scope_type,
            scope_id=scope_id,
        ).single()

        return {
            "chunks_stored": len(chunks),
            "chunks_deleted": deleted_count,
            "entities_stored": entity_result["total"] if entity_result else 0,
            "relationships_stored": (
                relationship_result["total"] if relationship_result else 0
            ),
        }

    def upsert_document(
        self,
        result: IngestionResult,
        extraction_results: Mapping[str, ExtractionResult] | None = None,
    ) -> dict[str, int]:
        """Create or update a document and its extracted knowledge atomically.

        Stable document, chunk, entity and evidence IDs make repeated calls
        idempotent while scope-specific entity IDs prevent cross-scope merging.
        """

        payload = self._build_document_payload(result, extraction_results)

        with self.connection.driver.session(
            database=self.connection.database
        ) as session:
            return session.execute_write(self._write_document, payload)

    @staticmethod
    def _delete_document(
        transaction: ManagedTransaction,
        document_id: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
    ) -> int:
        """Delete one document and its graph evidence, optionally checking scope."""

        result = transaction.run(
            """
            MATCH (d:Document {document_id: $document_id})
            WHERE ($scope_type IS NULL OR d.scope_type = $scope_type)
              AND ($scope_id IS NULL OR d.scope_id = $scope_id)
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            WITH d, collect(DISTINCT c) AS chunks
            UNWIND CASE
                WHEN size(chunks) = 0 THEN [null]
                ELSE chunks
            END AS chunk
            OPTIONAL MATCH (chunk)-[:SUPPORTS]->(relationship)
            WITH d, chunks, collect(DISTINCT relationship) AS relationships
            FOREACH (
                item IN relationships |
                DETACH DELETE item
            )
            FOREACH (
                item IN chunks |
                DETACH DELETE item
            )
            WITH d, size(chunks) AS deleted_count
            DETACH DELETE d
            RETURN deleted_count
            """,
            document_id=document_id,
            scope_type=scope_type,
            scope_id=scope_id,
        ).single()

        if result is None:
            return 0

        if scope_type is not None and scope_id is not None:
            transaction.run(
                """
                MATCH (e:Entity {
                    scope_type: $scope_type,
                    scope_id: $scope_id
                })
                WHERE NOT EXISTS {
                    MATCH (:Chunk)-[:MENTIONS]->(e)
                }
                DETACH DELETE e
                """,
                scope_type=scope_type,
                scope_id=scope_id,
            ).consume()
        else:
            transaction.run(
                """
                MATCH (e:Entity)
                WHERE NOT EXISTS {
                    MATCH (:Chunk)-[:MENTIONS]->(e)
                }
                DETACH DELETE e
                """
            ).consume()

        return result["deleted_count"]

    def delete_document(
        self,
        document_id: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
    ) -> int:
        """Delete a document, optionally requiring its expected scope.

        Supplying both scope arguments prevents a caller from deleting a
        document outside the intended personal or project knowledge space.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if (scope_type is None) != (scope_id is None):
            raise ValueError("scope_type and scope_id must be supplied together.")

        if scope_type is not None and scope_type not in {"personal", "project"}:
            raise ValueError("scope_type must be 'personal' or 'project'.")

        if scope_id is not None and not scope_id.strip():
            raise ValueError("scope_id cannot be empty.")

        with self.connection.driver.session(
            database=self.connection.database
        ) as session:
            return session.execute_write(
                self._delete_document,
                document_id,
                scope_type,
                scope_id,
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

    def count_entities(self) -> int:
        """Return the number of Entity nodes in the graph."""

        with self.connection.driver.session(
            database=self.connection.database
        ) as session:
            result = session.run(
                "MATCH (e:Entity) RETURN count(e) AS total"
            ).single()
            return result["total"] if result else 0

    def count_knowledge_relationships(self) -> int:
        """Return the number of evidence-backed relationship nodes."""

        with self.connection.driver.session(
            database=self.connection.database
        ) as session:
            result = session.run(
                """
                MATCH (r:KnowledgeRelationship)
                RETURN count(r) AS total
                """
            ).single()
            return result["total"] if result else 0
