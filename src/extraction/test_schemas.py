
"""Tests for entity and relationship extraction schemas."""

import pytest
from pydantic import ValidationError

from src.extraction.schemas import (
    Entity,
    ExtractionResult,
    Relationship,
)


def test_valid_extraction():
    """Verify that valid entities and relationships are accepted."""

    result = ExtractionResult(
        entities=[
            Entity(name="John Smith", entity_type="person"),
            Entity(name="Cloud Migration Project", entity_type="project"),
            Entity(name="AWS", entity_type="technology"),
        ],
        relationships=[
            Relationship(
                source="John Smith",
                relationship_type="manages",
                target="Cloud Migration Project",
                evidence="John Smith manages the Cloud Migration Project.",
            ),
            Relationship(
                source="Cloud Migration Project",
                relationship_type="uses",
                target="aws",
                evidence="The project uses AWS infrastructure.",
            ),
        ],
    )

    assert len(result.entities) == 3
    assert len(result.relationships) == 2
    assert result.entities[0].entity_type == "PERSON"
    assert result.relationships[0].relationship_type == "MANAGES"

    print("Valid extraction test passed.")


def test_invalid_relationship():
    """Verify that relationships referencing unknown entities are rejected."""

    with pytest.raises(
        ValidationError,
        match="Target entity 'Unknown Project' was not extracted",
    ):
        ExtractionResult(
            entities=[
                Entity(name="John Smith", entity_type="PERSON"),
            ],
            relationships=[
                Relationship(
                    source="John Smith",
                    relationship_type="MANAGES",
                    target="Unknown Project",
                    evidence="John Smith manages the project.",
                ),
            ],
        )


def test_duplicate_entity_names_are_rejected_case_insensitively():
    """Prevent ambiguous entities that differ only by letter casing."""

    with pytest.raises(ValidationError, match="Duplicate entity name 'aws'"):
        ExtractionResult(
            entities=[
                Entity(name="AWS", entity_type="TECHNOLOGY"),
                Entity(name="aws", entity_type="SYSTEM"),
            ],
        )


if __name__ == "__main__":
    test_valid_extraction()
    test_invalid_relationship()

    print("\nALL SCHEMA TESTS PASSED.")
