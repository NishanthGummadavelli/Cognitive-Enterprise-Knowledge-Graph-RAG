
"""Test entity-type refinement and validation."""

import pytest

from src.extraction.entity_type_refiner import EntityTypeRefiner
from src.extraction.schemas import (
    Entity,
    EntityTypeClassification,
    EntityTypeRefinementResult,
    ExtractionResult,
)


def test_generic_entity_type_is_refined():
    """Verify that an unresolved entity receives a more specific type."""

    refiner = EntityTypeRefiner()

    entities = [
        Entity(name="AWS", entity_type="OTHER"),
        Entity(name="John Smith", entity_type="PERSON"),
    ]

    refinements = EntityTypeRefinementResult(
        classifications=[
            EntityTypeClassification(
                name="AWS",
                entity_type="TECHNOLOGY",
                reason="The document describes AWS as infrastructure.",
            )
        ]
    )

    updated_entities = refiner._apply_refinements(
        entities,
        refinements,
        "The project uses AWS infrastructure. John Smith manages it.",
    )

    entity_types = {
        entity.name: entity.entity_type
        for entity in updated_entities
    }

    assert entity_types["AWS"] == "TECHNOLOGY"
    assert entity_types["John Smith"] == "PERSON"

    print("Generic entity refinement test passed.")


def test_existing_entity_type_is_not_overwritten():
    """Verify that meaningful entity types remain unchanged."""

    refiner = EntityTypeRefiner()

    entities = [
        Entity(name="Cloud Migration Project", entity_type="PROJECT"),
    ]

    refinements = EntityTypeRefinementResult()

    updated_entities = refiner._apply_refinements(
        entities,
        refinements,
        "Cloud Migration Project is an enterprise initiative.",
    )

    assert updated_entities[0].entity_type == "PROJECT"

    print("Existing entity type preservation test passed.")


def test_unknown_entity_classification_is_rejected():
    """Reject classifications for entities that were not requested."""

    refiner = EntityTypeRefiner()

    entities = [
        Entity(name="AWS", entity_type="OTHER"),
    ]

    refinements = EntityTypeRefinementResult(
        classifications=[
            EntityTypeClassification(
                name="Google Cloud",
                entity_type="TECHNOLOGY",
                reason="Not part of the supplied entity list.",
            )
        ]
    )

    with pytest.raises(ValueError, match="Unexpected entity classification"):
        refiner._apply_refinements(
            entities,
            refinements,
            "The project uses AWS infrastructure.",
        )

    print("Unexpected entity rejection test passed.")


def test_relationships_are_preserved():
    """Verify that refinement does not remove extracted relationships."""

    refiner = EntityTypeRefiner()

    original_result = ExtractionResult(
        entities=[
            Entity(name="AWS", entity_type="OTHER"),
            Entity(name="Cloud Migration Project", entity_type="PROJECT"),
        ],
        relationships=[
            {
                "source": "Cloud Migration Project",
                "relationship_type": "USES",
                "target": "AWS",
                "evidence": "The Cloud Migration Project uses AWS.",
            }
        ],
    )

    updated_entities = refiner._apply_refinements(
        original_result.entities,
        EntityTypeRefinementResult(
            classifications=[
                EntityTypeClassification(
                    name="AWS",
                    entity_type="TECHNOLOGY",
                    reason="AWS is described as the infrastructure.",
                )
            ]
        ),
        "The Cloud Migration Project uses AWS.",
    )

    updated_result = ExtractionResult(
        entities=updated_entities,
        relationships=original_result.relationships,
    )

    assert len(updated_result.relationships) == 1
    assert updated_result.relationships[0].target == "AWS"

    print("Relationship preservation test passed.")


if __name__ == "__main__":
    test_generic_entity_type_is_refined()
    test_existing_entity_type_is_not_overwritten()
    test_unknown_entity_classification_is_rejected()
    test_relationships_are_preserved()

    print("\nALL ENTITY-TYPE REFINEMENT TESTS PASSED.")
