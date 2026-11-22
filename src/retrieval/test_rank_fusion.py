
"""Unit tests for Reciprocal Rank Fusion."""

from src.retrieval.rank_fusion import RankFusion


def create_result(chunk_id: str, text: str) -> dict:
    """Create a sample retrieval result for testing."""
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{chunk_id}",
        "text": text,
    }


def main() -> None:
    """Run all RRF unit tests."""
    print("=" * 60)
    print("RECIPROCAL RANK FUSION TEST")
    print("=" * 60)

    fusion = RankFusion(k=60)

    # Test 1: Combine results from two retrieval methods.
    vector_results = [
        create_result("chunk_A", "AWS Lambda processes events."),
        create_result("chunk_B", "SQS stores messages."),
        create_result("chunk_C", "CloudWatch monitors services."),
    ]

    keyword_results = [
        create_result("chunk_B", "SQS stores messages."),
        create_result("chunk_A", "AWS Lambda processes events."),
        create_result("chunk_D", "Amazon EC2 runs virtual servers."),
    ]

    results = fusion.fuse(
        {
            "vector": vector_results,
            "keyword": keyword_results,
        }
    )

    assert len(results) == 4
    assert results[0]["chunk_id"] in {"chunk_A", "chunk_B"}
    assert results[1]["chunk_id"] in {"chunk_A", "chunk_B"}
    assert results[0]["rrf_score"] == results[1]["rrf_score"]

    print("TEST 1 PASSED: Results from multiple retrievers are fused.")

    # Test 2: Verify the RRF score calculation.
    chunk_a = next(
        item for item in results if item["chunk_id"] == "chunk_A"
    )

    expected_score = (1 / 61) + (1 / 62)

    assert abs(chunk_a["rrf_score"] - expected_score) < 1e-10

    print("TEST 2 PASSED: RRF scores are calculated correctly.")

    # Test 3: Verify retrieval rank tracking.
    assert chunk_a["retrieval_ranks"] == {
        "vector": 1,
        "keyword": 2,
    }

    print("TEST 3 PASSED: Original retrieval ranks are preserved.")

    # Test 4: Duplicate chunks within one retriever count only once.
    duplicate_results = fusion.fuse(
        {
            "vector": [
                create_result("chunk_X", "First occurrence."),
                create_result("chunk_X", "Duplicate occurrence."),
            ]
        }
    )

    assert len(duplicate_results) == 1
    assert abs(duplicate_results[0]["rrf_score"] - (1 / 61)) < 1e-10

    print("TEST 4 PASSED: Duplicate chunks are handled correctly.")

    # Test 5: Empty retrieval results.
    empty_results = fusion.fuse(
        {
            "vector": [],
            "keyword": [],
        }
    )

    assert empty_results == []

    print("TEST 5 PASSED: Empty retrieval results are handled.")

    # Test 6: Top-k filtering.
    limited_results = fusion.fuse(
        {
            "vector": vector_results,
            "keyword": keyword_results,
        },
        top_k=2,
    )

    assert len(limited_results) == 2

    print("TEST 6 PASSED: Top-k filtering works.")

    # Test 7: Invalid k value.
    try:
        RankFusion(k=0)
        raise AssertionError("Invalid k should have been rejected.")
    except ValueError:
        pass

    print("TEST 7 PASSED: Invalid RRF constant is rejected.")

    # Test 8: Invalid top_k value.
    try:
        fusion.fuse({"vector": vector_results}, top_k=0)
        raise AssertionError("Invalid top_k should have been rejected.")
    except ValueError:
        pass

    print("TEST 8 PASSED: Invalid top_k is rejected.")

    # Test 9: Invalid chunk ID.
    try:
        fusion.fuse(
            {
                "vector": [
                    {"text": "Missing chunk ID"}
                ]
            }
        )
        raise AssertionError("Missing chunk_id should have been rejected.")
    except ValueError:
        pass

    print("TEST 9 PASSED: Missing chunk IDs are rejected.")

    print("\nALL RRF TESTS PASSED")


if __name__ == "__main__":
    main()
