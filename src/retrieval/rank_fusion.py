
"""Combine ranked retrieval results using Reciprocal Rank Fusion (RRF)."""

from typing import Any


class RankFusion:
    """Fuse ranked results from multiple retrieval methods using RRF."""

    def __init__(self, k: int = 60) -> None:
        """Initialize the fusion component with the RRF ranking constant."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        self.k = k

    def fuse(
        self,
        ranked_results: dict[str, list[dict[str, Any]]],
        top_k: int | None = 10,
    ) -> list[dict[str, Any]]:
        """Combine ranked lists and return chunks ordered by their RRF scores."""
        if not isinstance(ranked_results, dict):
            raise TypeError("ranked_results must be a dictionary.")

        if not ranked_results:
            return []

        if top_k is not None and (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None.")

        fused_chunks: dict[str, dict[str, Any]] = {}

        for retriever_name, results in ranked_results.items():
            if not isinstance(retriever_name, str) or not retriever_name.strip():
                raise ValueError("Retriever names must be non-empty strings.")

            if not isinstance(results, list):
                raise TypeError(
                    f"Results from '{retriever_name}' must be a list."
                )

            seen_in_retriever: set[str] = set()

            for rank, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    raise TypeError("Each retrieval result must be a dictionary.")

                chunk_id = result.get("chunk_id")

                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError(
                        f"Every result from '{retriever_name}' "
                        "must contain a valid chunk_id."
                    )

                # A retriever should contribute only one ranking per chunk.
                if chunk_id in seen_in_retriever:
                    continue

                seen_in_retriever.add(chunk_id)

                if chunk_id not in fused_chunks:
                    fused_chunks[chunk_id] = {
                        **result,
                        "chunk_id": chunk_id,
                        "rrf_score": 0.0,
                        "retrieval_ranks": {},
                    }

                fused_chunks[chunk_id]["rrf_score"] += (
                    1.0 / (self.k + rank)
                )

                fused_chunks[chunk_id]["retrieval_ranks"][
                    retriever_name
                ] = rank

        # Highest combined RRF score comes first.
        ranked_chunks = sorted(
            fused_chunks.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        if top_k is not None:
            ranked_chunks = ranked_chunks[:top_k]

        return ranked_chunks
