
"""Test hybrid retrieval integration and knowledge-space isolation."""

from typing import Any

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rank_fusion import RankFusion


class FakeVectorStore:
    """Provide predictable, scope-aware vector search results."""

    def __init__(self) -> None:
        """Initialize sample vector results for isolated scopes."""
        self.data = {
            ("personal", "user_a"): [
                self._result("chunk_1", "user_a", 0.15),
                self._result("chunk_2", "user_a", 0.25),
            ],
            ("personal", "user_b"): [
                self._result("chunk_3", "user_b", 0.18),
            ],
            ("project", "project_alpha"): [
                self._result("chunk_4", "project_alpha", 0.12),
            ],
        }

    @staticmethod
    def _result(
        chunk_id: str,
        scope_id: str,
        distance: float,
    ) -> dict[str, Any]:
        """Build a sample vector retrieval result."""
        return {
            "chunk_id": chunk_id,
            "text": f"Content for {chunk_id}",
            "metadata": {
                "document_id": f"doc_{chunk_id}",
                "scope_type": (
                    "project" if scope_id == "project_alpha" else "personal"
                ),
                "scope_id": scope_id,
                "uploaded_by": scope_id,
            },
            "distance": distance,
            "scope_type": (
                "project" if scope_id == "project_alpha" else "personal"
            ),
            "scope_id": scope_id,
        }

    def search(
        self,
        query: str,
        top_k: int,
        scope_type: str,
        scope_id: str,
    ) -> list[dict[str, Any]]:
        """Return only vector results belonging to the requested scope."""
        return self.data.get((scope_type, scope_id), [])[:top_k]


class FakeKeywordRetriever:
    """Provide predictable, scope-aware keyword search results."""

    def search(
        self,
        query: str,
        scope_type: str,
        scope_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return sample BM25 results for the requested scope."""
        data = {
            ("personal", "user_a"): [
                self._result("chunk_2", scope_type, scope_id, 8.5),
                self._result("chunk_1", scope_type, scope_id, 6.2),
            ],
            ("personal", "user_b"): [
                self._result("chunk_3", scope_type, scope_id, 7.0),
            ],
            ("project", "project_alpha"): [
                self._result("chunk_4", scope_type, scope_id, 9.0),
            ],
        }

        return data.get((scope_type, scope_id), [])[:top_k]

    @staticmethod
    def _result(
        chunk_id: str,
        scope_type: str,
        scope_id: str,
        score: float,
    ) -> dict[str, Any]:
        """Build a sample keyword retrieval result."""
        return {
            "chunk_id": chunk_id,
            "document_id": f"doc_{chunk_id}",
            "text": f"Keyword content for {chunk_id}",
            "metadata": {
                "document_id": f"doc_{chunk_id}",
                "scope_type": scope_type,
                "scope_id": scope_id,
            },
            "bm25_score": score,
            "scope_type": scope_type,
            "scope_id": scope_id,
        }


class FakeGraphRetriever:
    """Provide predictable, scope-aware graph search results."""

    def search(
        self,
        query: str,
        scope_type: str,
        scope_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return sample graph evidence for the requested scope."""
        data = {
            ("personal", "user_a"): [
                self._result("chunk_1", scope_type, scope_id, 2),
                self._result("chunk_2", scope_type, scope_id, 1),
            ],
            ("personal", "user_b"): [
                self._result("chunk_3", scope_type, scope_id, 1),
            ],
            ("project", "project_alpha"): [
                self._result("chunk_4", scope_type, scope_id, 2),
            ],
        }

        return data.get((scope_type, scope_id), [])[:top_k]

    @staticmethod
    def _result(
        chunk_id: str,
        scope_type: str,
        scope_id: str,
        score: int,
    ) -> dict[str, Any]:
        """Build a sample graph retrieval result."""
        return {
            "chunk_id": chunk_id,
            "document_id": f"doc_{chunk_id}",
            "text": f"Graph content for {chunk_id}",
            "source": f"source_{chunk_id}.pdf",
            "metadata": {
                "document_id": f"doc_{chunk_id}",
                "scope_type": scope_type,
                "scope_id": scope_id,
                "page_numbers": [1],
            },
            "matched_entities": ["AWS Lambda"],
            "relationships": [
                {
                    "relationship_type": "USES",
                    "evidence": "Example relationship evidence",
                }
            ],
            "matched_terms": ["aws", "lambda"],
            "graph_score": score,
            "scope_type": scope_type,
            "scope_id": scope_id,
        }


def build_retriever(
    vector_store: Any | None = None,
    keyword_retriever: Any | None = None,
    graph_retriever: Any | None = None,
) -> HybridRetriever:
    """Construct a hybrid retriever using test dependencies."""
    return HybridRetriever(
        vector_store=vector_store or FakeVectorStore(),
        keyword_retriever=keyword_retriever or FakeKeywordRetriever(),
        graph_retriever=graph_retriever or FakeGraphRetriever(),
        rank_fusion=RankFusion(k=60),
    )


def test_all_three_retrievers_are_combined() -> None:
    """Verify vector, keyword and graph results are fused."""
    retriever = build_retriever()

    results = retriever.search(
        query="AWS Lambda",
        scope_type="personal",
        scope_id="user_a",
        top_k=5,
    )

    assert len(results) == 2

    for result in results:
        assert "rrf_score" in result
        assert "retrieval_ranks" in result
        assert "retrieval_sources" in result

    print("TEST 1 PASSED: All three retrieval methods are combined.")


def test_retrieval_details_are_preserved() -> None:
    """Verify source-specific scores and graph evidence survive fusion."""
    retriever = build_retriever()

    results = retriever.search(
        query="AWS Lambda",
        scope_type="personal",
        scope_id="user_a",
    )

    chunk_1 = next(
        result for result in results
        if result["chunk_id"] == "chunk_1"
    )

    assert set(chunk_1["retrieval_sources"]) == {
        "vector",
        "keyword",
        "graph",
    }
    assert chunk_1["vector_distance"] == 0.15
    assert chunk_1["bm25_score"] == 6.2
    assert chunk_1["graph_score"] == 2
    assert chunk_1["matched_entities"] == ["AWS Lambda"]
    assert chunk_1["relationships"]
    assert chunk_1["metadata"]["page_numbers"] == [1]

    print("TEST 2 PASSED: Retrieval scores and graph evidence are preserved.")


def test_rrf_ranking_is_used() -> None:
    """Verify results are ordered by the fused RRF score."""
    retriever = build_retriever()

    results = retriever.search(
        query="AWS Lambda",
        scope_type="personal",
        scope_id="user_a",
    )

    assert results[0]["chunk_id"] == "chunk_1"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]

    print("TEST 3 PASSED: Reciprocal Rank Fusion ranking is applied.")


def test_personal_scope_isolation() -> None:
    """Verify users retrieve only their own personal knowledge."""
    retriever = build_retriever()

    user_a_results = retriever.search(
        query="AWS Lambda",
        scope_type="personal",
        scope_id="user_a",
    )

    user_b_results = retriever.search(
        query="AWS Lambda",
        scope_type="personal",
        scope_id="user_b",
    )

    assert user_a_results
    assert user_b_results

    assert all(
        result["scope_id"] == "user_a"
        for result in user_a_results
    )

    assert all(
        result["scope_id"] == "user_b"
        for result in user_b_results
    )

    assert not (
        {item["chunk_id"] for item in user_a_results}
        & {item["chunk_id"] for item in user_b_results}
    )

    print("TEST 4 PASSED: Personal knowledge spaces are isolated.")


def test_project_scope_isolation() -> None:
    """Verify project retrieval returns only the requested project's data."""
    retriever = build_retriever()

    results = retriever.search(
        query="AWS Lambda",
        scope_type="project",
        scope_id="project_alpha",
    )

    assert results
    assert all(
        result["scope_type"] == "project"
        and result["scope_id"] == "project_alpha"
        for result in results
    )

    print("TEST 5 PASSED: Project knowledge space is isolated.")


def test_unknown_scope_returns_no_results() -> None:
    """Verify an unknown scope cannot retrieve another scope's data."""
    retriever = build_retriever()

    results = retriever.search(
        query="AWS Lambda",
        scope_type="personal",
        scope_id="unknown_user",
    )

    assert results == []

    print("TEST 6 PASSED: Unknown scope returns no results.")


def test_top_k_is_respected() -> None:
    """Verify the hybrid retriever respects the final result limit."""
    retriever = build_retriever()

    results = retriever.search(
        query="AWS Lambda",
        scope_type="personal",
        scope_id="user_a",
        top_k=1,
        retrieval_top_k=10,
    )

    assert len(results) == 1

    print("TEST 7 PASSED: Final top_k limit is respected.")


def test_invalid_inputs_are_rejected() -> None:
    """Verify invalid queries, scopes and result limits are rejected."""
    retriever = build_retriever()

    invalid_calls = [
        {"query": "", "scope_type": "personal", "scope_id": "user_a"},
        {"query": "AWS", "scope_type": "invalid", "scope_id": "user_a"},
        {"query": "AWS", "scope_type": "personal", "scope_id": ""},
        {
            "query": "AWS",
            "scope_type": "personal",
            "scope_id": "user_a",
            "top_k": 0,
        },
        {
            "query": "AWS",
            "scope_type": "personal",
            "scope_id": "user_a",
            "retrieval_top_k": 0,
        },
    ]

    for arguments in invalid_calls:
        try:
            retriever.search(**arguments)
        except ValueError:
            continue

        raise AssertionError(
            f"Invalid arguments were accepted: {arguments}"
        )

    print("TEST 8 PASSED: Invalid inputs are rejected.")


def test_out_of_scope_result_is_rejected() -> None:
    """Verify the hybrid layer detects a retrieval scope violation."""

    class UnsafeVectorStore:
        """Simulate a vector retriever returning another user's result."""

        def search(
            self,
            query: str,
            top_k: int,
            scope_type: str,
            scope_id: str,
        ) -> list[dict[str, Any]]:
            """Return a deliberately unauthorized result."""
            return [
                {
                    "chunk_id": "private_chunk",
                    "text": "Private user data",
                    "metadata": {},
                    "distance": 0.1,
                    "scope_type": "personal",
                    "scope_id": "user_b",
                }
            ]

    retriever = build_retriever(vector_store=UnsafeVectorStore())

    try:
        retriever.search(
            query="AWS",
            scope_type="personal",
            scope_id="user_a",
        )
    except RuntimeError:
        print("TEST 9 PASSED: Out-of-scope retrieval is rejected.")
        return

    raise AssertionError("Out-of-scope result was not rejected.")


def run_tests() -> None:
    """Run all hybrid retrieval integration tests."""
    tests = [
        test_all_three_retrievers_are_combined,
        test_retrieval_details_are_preserved,
        test_rrf_ranking_is_used,
        test_personal_scope_isolation,
        test_project_scope_isolation,
        test_unknown_scope_returns_no_results,
        test_top_k_is_respected,
        test_invalid_inputs_are_rejected,
        test_out_of_scope_result_is_rejected,
    ]

    for test in tests:
        test()

    print("\nALL HYBRID RETRIEVAL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
