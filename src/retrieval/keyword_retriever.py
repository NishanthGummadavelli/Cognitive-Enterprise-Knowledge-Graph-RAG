
"""Scope-aware BM25 keyword retrieval over stored document chunks."""

import re
from typing import Any

from rank_bm25 import BM25Plus

from src.embeddings.vector_store import VectorStore


class KeywordRetriever:
    """Retrieve relevant document chunks using BM25Plus within an authorized scope."""

    VALID_SCOPE_TYPES = {"personal", "project"}

    def __init__(self, vector_store: VectorStore) -> None:
        """Initialize the retriever with the existing ChromaDB vector store."""
        self.vector_store = vector_store

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into lowercase alphanumeric keyword tokens."""
        return re.findall(r"\b[a-z0-9_]+\b", text.casefold())

    @classmethod
    def _validate_scope(cls, scope_type: str, scope_id: str) -> None:
        """Reject invalid or incomplete retrieval scope information."""
        if not isinstance(scope_type, str) or scope_type not in cls.VALID_SCOPE_TYPES:
            raise ValueError("scope_type must be 'personal' or 'project'.")

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("scope_id must be a non-empty string.")

    def search(
        self,
        query: str,
        scope_type: str,
        scope_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring keyword matches from the requested scope."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        self._validate_scope(scope_type, scope_id)

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Retrieve only chunks belonging to the requested scope.
        try:
            stored_data = self.vector_store.collection.get(
                where={
                    "$and": [
                        {"scope_type": {"$eq": scope_type}},
                        {"scope_id": {"$eq": scope_id}},
                    ]
                },
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to retrieve chunks for scope {scope_type}:{scope_id}."
            ) from exc

        chunk_ids = stored_data.get("ids", [])
        documents = stored_data.get("documents", [])
        metadatas = stored_data.get("metadatas", [])

        if not (len(chunk_ids) == len(documents) == len(metadatas)):
            raise RuntimeError("ChromaDB returned inconsistent chunk data.")

        corpus: list[list[str]] = []
        candidates: list[dict[str, Any]] = []

        for chunk_id, document_text, metadata in zip(
            chunk_ids, documents, metadatas
        ):
            if not document_text or not metadata:
                continue

            # Defensively verify scope even after the database filter.
            if (
                metadata.get("scope_type") != scope_type
                or metadata.get("scope_id") != scope_id
            ):
                continue

            tokens = self._tokenize(document_text)
            if not tokens:
                continue

            corpus.append(tokens)
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id"),
                    "text": document_text,
                    "metadata": metadata,
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                }
            )

        if not corpus:
            return []

        # BM25Plus handles small corpora more reliably than BM25Okapi.
        bm25 = BM25Plus(corpus)
        scores = bm25.get_scores(query_tokens)

        results: list[dict[str, Any]] = []

        for candidate, score in zip(candidates, scores):
            # A positive score alone is not sufficient: require actual
            # keyword overlap so unrelated chunks are not returned.
            if not any(token in candidate["text"].casefold() for token in query_tokens):
                continue

            candidate["bm25_score"] = float(score)
            results.append(candidate)

        results.sort(key=lambda item: item["bm25_score"], reverse=True)

        return results[:top_k]
