
"""
Extract candidate named entities from enterprise document text using spaCy.
"""

import spacy
from spacy.language import Language


class SpacyEntityExtractor:
    """Identify candidate entities using a pre-trained spaCy NER model."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Load the configured spaCy language model."""

        try:
            self.nlp: Language = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load spaCy model '{model_name}'. "
                "Install it using: python -m spacy download en_core_web_sm"
            ) from exc

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Extract unique candidate entities from the supplied text."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # Run the text through spaCy's NLP pipeline.
        doc = self.nlp(text)

        entities: list[dict[str, str]] = []
        seen_entities: set[tuple[str, str]] = set()

        # Collect named entities recognized by the model.
        for entity in doc.ents:
            name = entity.text.strip()
            entity_type = entity.label_

            # Avoid duplicate entity mentions with the same type.
            key = (name.casefold(), entity_type)

            if key in seen_entities:
                continue

            seen_entities.add(key)

            entities.append(
                {
                    "name": name,
                    "entity_type": entity_type,
                }
            )

        return entities
