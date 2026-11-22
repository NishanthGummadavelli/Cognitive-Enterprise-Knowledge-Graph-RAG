
"""Test entity-type refinement schemas."""

from src.extraction.schemas import (
    EntityTypeClassification,
    EntityTypeRefinementResult,
)


def test_valid_entity_type_classification():
    """Verify that valid entity classifications are accepted."""

    result = EntityTypeRefinementResult(
        classifications=[
            EntityTypeClassification(
                name="AWS",
                entity_type="technology",
                reason="The document describes AWS as project infrastructure.",
            ),
            EntityTypeClassification(
                name="Human Resources",
                entity_type="department",
                reason="The document identifies it as a department.",
            ),
        ]
    )

    assert len(result.classifications) == 2
    assert result.classifications[0].entity_type == "TECHNOLOGY"
    assert result.classifications[1].entity_type == "DEPARTMENT"

    print("Entity-type classification schema test passed.")


if __name__ == "__main__":
    test_valid_entity_type_classification()
    print("\nALL ENTITY-TYPE SCHEMA TESTS PASSED.")
