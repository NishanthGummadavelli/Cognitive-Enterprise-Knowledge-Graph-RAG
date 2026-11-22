
"""Test contextual entity and relationship extraction using Qwen3."""

import pytest
from pydantic import ValidationError

from src.extraction.entity_extractor import (
    EntityExtractor,
    _add_missing_relationship_entities,
)
from src.extraction.schemas import ExtractionResult


def test_missing_relationship_endpoint_named_in_text_is_added():
    """Keep an explicitly mentioned endpoint when Qwen omits its entity."""

    response_data = {
        "entities": [
            {"name": "Employee Onboarding", "entity_type": "PROCESS"},
        ],
        "relationships": [
            {
                "source": "Employee Onboarding",
                "relationship_type": "USES",
                "target": "Workday",
                "evidence": "Employee onboarding uses Workday.",
            },
        ],
    }

    completed_data = _add_missing_relationship_entities(
        response_data,
        "Employee onboarding uses Workday.",
        [],
    )

    result = ExtractionResult.model_validate(completed_data)

    workday = next(
        entity for entity in result.entities
        if entity.name == "Workday"
    )

    assert workday.entity_type == "OTHER"

    print("Missing relationship endpoint recovery test passed.")


def test_missing_relationship_endpoint_absent_from_text_is_not_added():
    """Reject relationship endpoints that are absent from the document."""

    response_data = {
        "entities": [
            {"name": "Employee Onboarding", "entity_type": "PROCESS"},
        ],
        "relationships": [
            {
                "source": "Employee Onboarding",
                "relationship_type": "USES",
                "target": "Workday",
                "evidence": "Employee onboarding uses a management system.",
            },
        ],
    }

    completed_data = _add_missing_relationship_entities(
        response_data,
        "Employee onboarding uses a management system.",
        [],
    )

    with pytest.raises(
        ValidationError,
        match="Target entity 'Workday' was not extracted",
    ):
        ExtractionResult.model_validate(completed_data)

    print("Unsupported relationship endpoint rejection test passed.")


def test_entity_and_relationship_extraction():
    """Test integrated extraction and refinement using an HR document."""

    sample_text = (
        "The Human Resources department at ABC Technologies "
        "uses Workday to manage employee onboarding. "
        "Priya Sharma leads the onboarding process and "
        "coordinates with the Finance department to complete "
        "employee salary registration."
    )

    extractor = EntityExtractor()

    print("\nSending HR sample document to Qwen3...")

    # This now runs both extraction and entity-type refinement.
    result = extractor.extract(sample_text)

    print("\nFINAL EXTRACTED ENTITIES")

    for entity in result.entities:
        print(f"Name: {entity.name:<30} Type: {entity.entity_type}")

    print("\nFINAL EXTRACTED RELATIONSHIPS")

    for relationship in result.relationships:
        print(
            f"{relationship.source} "
            f"--[{relationship.relationship_type}]--> "
            f"{relationship.target}"
        )
        print(f"Evidence: {relationship.evidence}\n")

    entity_names = {
        entity.name.casefold()
        for entity in result.entities
    }

    # Verify that important entities remain present after refinement.
    assert "priya sharma" in entity_names
    assert "abc technologies" in entity_names

    print("Integrated HR extraction and refinement test passed.")


if __name__ == "__main__":
    test_entity_and_relationship_extraction()
