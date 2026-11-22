
"""Combine vector, keyword and graph retrieval into one hybrid pipeline."""

from typing import Any

from src.embeddings.vector_store import VectorStore
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.rank_fusion import RankFusion


class HybridRetriever:
    """Retrieve and fuse evidence from vector, keyword and graph search."""

    VALID_SCOPE_TYPES = {"personal", "project"}

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_retriever: KeywordRetriever,
        graph_retriever: GraphRetriever,
        rank_fusion: RankFusion,
    ) -> None:
        """Initialize the hybrid retriever with existing retrieval components."""
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever
        self.graph_retriever = graph_retriever
        self.rank_fusion = rank_fusion

    @classmethod
    def _validate_scope(
        cls,
        scope_type: str,
        scope_id: str,
    ) -> None:
        """Validate the personal or project knowledge-space identifier."""
        if (
            not isinstance(scope_type, str)
            or scope_type not in cls.VALID_SCOPE_TYPES
        ):
            raise ValueError(
                "scope_type must be 'personal' or 'project'."
            )

        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("scope_id must be a non-empty string.")

    @staticmethod
    def _validate_results(
        results: list[dict[str, Any]],
        retriever_name: str,
        scope_type: str,
        scope_id: str,
    ) -> list[dict[str, Any]]:
        """Reject malformed or out-of-scope results from a retriever."""
        validated = []

        for item in results:
            if not isinstance(item, dict):
                raise TypeError(
                    f"{retriever_name} returned an invalid result."
                )

            if item.get("scope_type") != scope_type:
                raise RuntimeError(
                    f"{retriever_name} returned a result from another scope."
                )

            if item.get("scope_id") != scope_id:
                raise RuntimeError(
                    f"{retriever_name} returned a result from another scope."
                )

            chunk_id = item.get("chunk_id")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise RuntimeError(
                    f"{retriever_name} returned a result without a valid chunk_id."
                )

            validated.append(item)

        return validated

    def search(
        self,
        query: str,
        scope_type: str,
        scope_id: str,
        top_k: int = 5,
        retrieval_top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Run all retrieval methods and return fused, scope-safe evidence."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer.")

        if (
            isinstance(retrieval_top_k, bool)
            or not isinstance(retrieval_top_k, int)
            or retrieval_top_k <= 0
        ):
            raise ValueError(
                "retrieval_top_k must be a positive integer."
            )

        self._validate_scope(scope_type, scope_id)

        # 1. Semantic vector retrieval.
        vector_results = self.vector_store.search(
            query=query,
            top_k=retrieval_top_k,
            scope_type=scope_type,
            scope_id=scope_id,
        )

        vector_results = self._validate_results(
            vector_results,
            "Vector retrieval",
            scope_type,
            scope_id,
        )

        # 2. Exact keyword retrieval.
        keyword_results = self.keyword_retriever.search(
            query=query,
            scope_type=scope_type,
            scope_id=scope_id,
            top_k=retrieval_top_k,
        )

        keyword_results = self._validate_results(
            keyword_results,
            "Keyword retrieval",
            scope_type,
            scope_id,
        )

        # 3. Knowledge graph retrieval.
        graph_results = self.graph_retriever.search(
            query=query,
            scope_type=scope_type,
            scope_id=scope_id,
            top_k=retrieval_top_k,
        )

        graph_results = self._validate_results(
            graph_results,
            "Graph retrieval",
            scope_type,
            scope_id,
        )

        # 4. Combine all three ranked lists using RRF.
        fused_results = self.rank_fusion.fuse(
            {
                "vector": vector_results,
                "keyword": keyword_results,
                "graph": graph_results,
            },
            top_k=top_k,
        )

        # 5. Preserve useful information from every retrieval method.
        source_results = {
            "vector": vector_results,
            "keyword": keyword_results,
            "graph": graph_results,
        }

        results_by_chunk: dict[str, dict[str, Any]] = {}

        for retriever_name, results in source_results.items():
            for item in results:
                chunk_id = item["chunk_id"]

                if chunk_id not in results_by_chunk:
                    results_by_chunk[chunk_id] = {
                        "retrieval_sources": [],
                        "metadata": {},
                    }

                merged = results_by_chunk[chunk_id]

                if retriever_name not in merged["retrieval_sources"]:
                    merged["retrieval_sources"].append(retriever_name)

                # Preserve the most useful source-specific information.
                if item.get("metadata"):
                    merged["metadata"].update(item["metadata"])

                if retriever_name == "vector":
                    merged["vector_distance"] = item.get("distance")

                elif retriever_name == "keyword":
                    merged["bm25_score"] = item.get("bm25_score")

                elif retriever_name == "graph":
                    merged["graph_score"] = item.get("graph_score")
                    merged["matched_entities"] = item.get(
                        "matched_entities", []
                    )
                    merged["relationships"] = item.get(
                        "relationships", []
                    )
                    merged["matched_terms"] = item.get(
                        "matched_terms", []
                    )

                if item.get("source"):
                    merged["source"] = item["source"]

        final_results = []

        for fused in fused_results:
            chunk_id = fused["chunk_id"]
            enrichment = results_by_chunk.get(chunk_id, {})

            final_item = {
                **fused,
                "retrieval_sources": enrichment.get(
                    "retrieval_sources", []
                ),
                "metadata": enrichment.get("metadata", {}),
                "scope_type": scope_type,
                "scope_id": scope_id,
            }

            for field in (
                "vector_distance",
                "bm25_score",
                "graph_score",
                "matched_entities",
                "relationships",
                "matched_terms",
                "source",
            ):
                if field in enrichment:
                    final_item[field] = enrichment[field]

            final_results.append(final_item)

        return final_results
