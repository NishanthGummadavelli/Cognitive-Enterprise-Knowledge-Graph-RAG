
"""
Refine generic entity types using document context and Qwen3.
"""

import ollama

from src.extraction.schemas import (
    Entity,
    EntityTypeRefinementResult,
    ExtractionResult,
)


class EntityTypeRefiner:
    """Refine unresolved entity categories using an LLM."""

    def __init__(
        self,
        model_name: str = "qwen3:4b",
        client: ollama.Client | None = None,
    ) -> None:
        """Initialize the Ollama client and refinement model."""

        self.model_name = model_name
        self.client = client or ollama.Client()

    def _apply_refinements(
        self,
        entities: list[Entity],
        refinements: EntityTypeRefinementResult,
        text: str,
    ) -> list[Entity]:
        """Apply validated classifications to unresolved entities only."""

        unresolved_types = {"OTHER", "UNCLASSIFIED"}

        # Identify entities that are eligible for refinement.
        unresolved_entities = {
            entity.name.casefold(): entity
            for entity in entities
            if entity.entity_type in unresolved_types
        }

        updated_entities = {
            entity.name.casefold(): entity
            for entity in entities
        }

        seen_names: set[str] = set()

        for classification in refinements.classifications:
            normalized_name = classification.name.casefold()

            # Reject classifications for entities that were not requested.
            if normalized_name not in unresolved_entities:
                raise ValueError(
                    f"Unexpected entity classification: "
                    f"'{classification.name}'."
                )

            if normalized_name in seen_names:
                raise ValueError(
                    f"Duplicate classification for '{classification.name}'."
                )

            seen_names.add(normalized_name)

            # Ensure the entity is actually mentioned in the source text.
            if classification.name.casefold() not in text.casefold():
                raise ValueError(
                    f"Entity '{classification.name}' is not present "
                    "in the original document."
                )

            # Do not accept another generic classification as a refinement.
            refined_type = classification.entity_type

            if refined_type in unresolved_types:
                continue

            original_entity = unresolved_entities[normalized_name]

            updated_entities[normalized_name] = Entity(
                name=original_entity.name,
                entity_type=refined_type,
            )

        return list(updated_entities.values())

    def refine(
        self,
        text: str,
        extraction_result: ExtractionResult,
    ) -> ExtractionResult:
        """Refine generic entity types while preserving existing relationships."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # Only send unresolved entities to the language model.
        unresolved_entities = [
            entity
            for entity in extraction_result.entities
            if entity.entity_type in {"OTHER", "UNCLASSIFIED"}
        ]

        # Avoid an unnecessary LLM call when all types are already meaningful.
        if not unresolved_entities:
            return extraction_result

        entity_names = [entity.name for entity in unresolved_entities]

        system_prompt = """
You are an enterprise knowledge graph entity classification assistant.

Your task is to classify the supplied unresolved entities using the
original document as context.

Choose the most appropriate category from:

PERSON
ORGANIZATION
DEPARTMENT
TECHNOLOGY
PROJECT
LOCATION
DOCUMENT
PROCESS
ROLE
PRODUCT
SERVICE
OTHER

Rules:
1. Classify only the entities supplied in the input.
2. Use the original document context to determine each entity's meaning.
3. Do not invent entities or classifications.
4. Do not change the spelling of entity names.
5. Return the exact entity name supplied in the input.
6. Do not use OTHER or UNCLASSIFIED when a more specific category
   is supported by the document.
7. If the context is insufficient, use OTHER.
8. Give a short reason based on the document.
"""

        user_prompt = f"""
Original document:
{text}

Entities requiring classification:
{entity_names}

Classify these entities using the original document context.
"""

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=EntityTypeRefinementResult.model_json_schema(),
                options={
                    "temperature": 0,
                    "num_predict": 500,
                },
                think=False,
            )

            response_text = response.message.content

            if not response_text:
                raise ValueError("The model returned an empty response.")

            # Validate the model's structured JSON response.
            refinements = EntityTypeRefinementResult.model_validate_json(
                response_text
            )

            updated_entities = self._apply_refinements(
                extraction_result.entities,
                refinements,
                text,
            )

            # Revalidate the complete result, including relationship endpoints.
            return ExtractionResult(
                entities=updated_entities,
                relationships=extraction_result.relationships,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Entity-type refinement failed: {exc}"
            ) from exc
