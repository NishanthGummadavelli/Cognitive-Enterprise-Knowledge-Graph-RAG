
"""Retrieve scope-specific document chunks through Neo4j knowledge graphs."""

import re
from typing import Any

from src.knowledge_graph.connection import Neo4jConnection


class GraphRetriever:
    """Retrieve document evidence by matching entities in the knowledge graph."""

    VALID_SCOPE_TYPES = {"personal", "project"}

    def __init__(self, connection: Neo4jConnection) -> None:
        """Initialize graph retrieval using the existing Neo4j connection."""
        self.connection = connection

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Convert text into normalized alphanumeric search terms."""
        return re.findall(r"\b[a-z0-9_]+\b", text.casefold())

    @classmethod
    def _validate_scope(cls, scope_type: str, scope_id: str) -> None:
        """Validate the requested personal or project knowledge scope."""
        if not isinstance(scope_type, str) or scope_type not in cls.VALID_SCOPE_TYPES:
            raise ValueError("scope_type must be 'personal' or 'project'.")

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("scope_id must be a non-empty string.")

    @staticmethod
    def _build_query() -> str:
        """Return a parameterized Cypher query for scoped graph retrieval."""
        return """
            MATCH (e:Entity {
                scope_type: $scope_type,
                scope_id: $scope_id
            })
            WHERE any(
                term IN $terms
                WHERE e.normalized_name CONTAINS term
                   OR toLower(e.name) CONTAINS term
            )

            MATCH (c:Chunk {
                scope_type: $scope_type,
                scope_id: $scope_id
            })-[:MENTIONS]->(e)

            OPTIONAL MATCH
                (c)-[:SUPPORTS]->(
                    r:KnowledgeRelationship {
                        scope_type: $scope_type,
                        scope_id: $scope_id
                    }
                )
                -[:SOURCE|TARGET]->(e)

            OPTIONAL MATCH (r)-[:SOURCE]->(source:Entity)
            OPTIONAL MATCH (r)-[:TARGET]->(target:Entity)

            RETURN
                c.chunk_id AS chunk_id,
                c.document_id AS document_id,
                c.text AS text,
                c.source AS source,
                c.page_numbers AS page_numbers,
                c.heading_path AS heading_path,
                c.chunk_index AS chunk_index,
                c.scope_type AS scope_type,
                c.scope_id AS scope_id,
                c.uploaded_by AS uploaded_by,

                collect(DISTINCT {
                    entity_id: e.entity_id,
                    name: e.name,
                    entity_type: e.entity_type
                }) AS matched_entities,

                collect(DISTINCT CASE
                    WHEN r IS NULL THEN NULL
                    ELSE {
                        relationship_id: r.relationship_id,
                        relationship_type: r.relationship_type,
                        evidence: r.evidence,
                        source_entity: source.name,
                        target_entity: target.name
                    }
                END) AS relationships
        """

    def search(
        self,
        query: str,
        scope_type: str,
        scope_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return graph-supported document chunks from the requested scope."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        self._validate_scope(scope_type, scope_id)

        terms = list(dict.fromkeys(self._tokenize(query)))

        # Avoid overly broad matches from very short query terms.
        terms = [term for term in terms if len(term) >= 2]

        if not terms:
            return []

        cypher = self._build_query()

        try:
            with self.connection.driver.session(
                database=self.connection.database
            ) as session:
                records = session.run(
                    cypher,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    terms=terms,
                )

                results = [record.data() for record in records]

        except Exception as exc:
            raise RuntimeError(
                "Neo4j graph retrieval failed for the requested scope."
            ) from exc

        ranked_results: list[dict[str, Any]] = []

        for item in results:
            # Defensively verify scope returned by Neo4j.
            if (
                item.get("scope_type") != scope_type
                or item.get("scope_id") != scope_id
            ):
                continue

            text = item.get("text") or ""
            matched_entities = item.get("matched_entities") or []

            if not item.get("chunk_id") or not text:
                continue

            matched_terms = {
                term
                for term in terms
                if any(
                    term in (entity.get("name") or "").casefold()
                    for entity in matched_entities
                )
            }

            # More matching query terms indicate stronger entity coverage.
            graph_score = len(matched_terms)

            if graph_score == 0:
                continue

            ranked_results.append(
                {
                    "chunk_id": item["chunk_id"],
                    "document_id": item.get("document_id"),
                    "text": text,
                    "source": item.get("source"),
                    "metadata": {
                        "page_numbers": item.get("page_numbers") or [],
                        "heading_path": item.get("heading_path") or [],
                        "chunk_index": item.get("chunk_index"),
                        "scope_type": item["scope_type"],
                        "scope_id": item["scope_id"],
                        "uploaded_by": item.get("uploaded_by"),
                    },
                    "matched_entities": matched_entities,
                    "relationships": item.get("relationships") or [],
                    "matched_terms": sorted(matched_terms),
                    "graph_score": graph_score,
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                }
            )

        ranked_results.sort(
            key=lambda item: item["graph_score"],
            reverse=True,
        )

        return ranked_results[:top_k]
