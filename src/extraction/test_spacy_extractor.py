
"""Test candidate entity extraction using spaCy."""

from src.extraction.spacy_extractor import SpacyEntityExtractor


def test_spacy_entity_extraction():
    """Verify that spaCy identifies candidate entities in sample text."""

    extractor = SpacyEntityExtractor()

    sample_text = (
        "John Smith manages the Cloud Migration Project. "
        "The project uses AWS infrastructure and is sponsored "
        "by ABC Technologies. Priya Sharma works with John Smith."
    )

    entities = extractor.extract_entities(sample_text)

    print("\nEntities detected by spaCy:")

    for entity in entities:
        print(
            f"Name: {entity['name']:<25} "
            f"Type: {entity['entity_type']}"
        )

    assert len(entities) > 0

    print("\nspaCy extraction test passed.")


if __name__ == "__main__":
    test_spacy_entity_extraction()
