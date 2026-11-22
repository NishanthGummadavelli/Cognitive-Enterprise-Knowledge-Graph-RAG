
"""
Extract enterprise entities and relationships using Qwen3 through Ollama.
"""

import json
import re

import ollama

from src.extraction.entity_type_refiner import EntityTypeRefiner
from src.extraction.schemas import ExtractionResult
from src.extraction.spacy_extractor import SpacyEntityExtractor


def _normalize_evidence_text(text: str) -> str:
    """Normalize whitespace and case for evidence comparison."""

    return " ".join(text.casefold().split())


def _add_missing_relationship_entities(
    response_data: object,
    text: str,
    candidates: list[dict[str, str]],
) -> object:
    """Add omitted relationship endpoints only when the text names them."""

    if not isinstance(response_data, dict):
        return response_data

    entities = response_data.get("entities")
    relationships = response_data.get("relationships")

    if not isinstance(entities, list) or not isinstance(relationships, list):
        return response_data

    # Track entities already extracted by Qwen3.
    entity_names = {
        entity["name"].casefold()
        for entity in entities
        if isinstance(entity, dict)
        and isinstance(entity.get("name"), str)
    }

    # Reuse spaCy classifications when available.
    candidate_types = {
        candidate["name"].casefold(): candidate["entity_type"]
        for candidate in candidates
        if isinstance(candidate.get("name"), str)
        and isinstance(candidate.get("entity_type"), str)
    }

    # Check every relationship endpoint.
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue

        for endpoint in ("source", "target"):
            name = relationship.get(endpoint)

            if (
                not isinstance(name, str)
                or not name.strip()
                or name.casefold() in entity_names
            ):
                continue

            normalized_name = name.strip()

            # Only recover entities explicitly mentioned in the document.
            if not re.search(
                rf"(?<!\w){re.escape(normalized_name)}(?!\w)",
                text,
                flags=re.IGNORECASE,
            ):
                continue

            entities.append(
                {
                    "name": normalized_name,
                    "entity_type": candidate_types.get(
                        normalized_name.casefold(),
                        "OTHER",
                    ),
                }
            )

            entity_names.add(normalized_name.casefold())

    return response_data


class EntityExtractor:
    """Combine spaCy, Qwen3 extraction, and entity-type refinement."""

    def __init__(
        self,
        model_name: str = "qwen3:4b",
        spacy_extractor: SpacyEntityExtractor | None = None,
        entity_type_refiner: EntityTypeRefiner | None = None,
    ) -> None:
        """Initialize the extraction and refinement components."""

        self.model_name = model_name
        self.client = ollama.Client()

        self.spacy_extractor = (
            spacy_extractor or SpacyEntityExtractor()
        )

        # Use the same configured model for extraction and refinement.
        self.entity_type_refiner = (
            entity_type_refiner
            or EntityTypeRefiner(model_name=model_name)
        )

    def extract(self, text: str) -> ExtractionResult:
        """Extract, validate, and refine entities and relationships."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # Step 1: Identify candidate entities using spaCy.
        candidates = self.spacy_extractor.extract_entities(text)

        system_prompt = """
You are an enterprise knowledge graph information extraction system.

Extract entities and relationships explicitly supported by the
provided document text.

ENTITY EXTRACTION RULES:
1. Discover people, organizations, projects, technologies,
   departments, systems, policies, processes, and other meaningful entities.
2. Do not restrict extraction to entities detected by spaCy.
3. Correct spaCy entity classifications when context supports it.
4. Use concise entity names. Remove unnecessary articles such as "the".
5. Include every entity mentioned as a source or target in a relationship.
6. Do not omit technologies, even when spaCy does not detect them.
7. Treat the candidate list as suggestions, not a complete entity list.

RELATIONSHIP EXTRACTION RULES:
8. Extract only relationships supported by the original text.
9. Every relationship source and target must exist in the entities list.
10. The evidence field MUST contain an exact, continuous quotation
    copied directly from the original document text.
11. NEVER paraphrase, summarize, rewrite, shorten, or grammatically
    correct the evidence.
12. Preserve the original words and their order exactly.
13. Do not replace words with synonyms or add words that are not present.
14. You may select a complete sentence or a shorter continuous phrase
    from the original text as evidence.
15. If the original sentence spans multiple lines, reproduce its words
    in their original order. Whitespace differences are acceptable.
16. If you cannot find an exact supporting quotation, DO NOT return
    that relationship.
17. Do not invent relationships or unsupported information.

FINAL VALIDATION:
18. Before returning the result, check every relationship.
19. Verify that both its source and target appear in the entities list.
20. Verify that every evidence quotation is copied from the document.
21. If a relationship references a missing entity, add that entity
    with the most appropriate category.
22. Return empty lists when no entities or relationships are present.
23. Return only JSON matching the supplied schema.
"""

        user_prompt = f"""
DOCUMENT TEXT:
{text}

CANDIDATE ENTITIES DETECTED BY SPACY:
{candidates}

IMPORTANT:
The document text is the only source of truth.

For every relationship, copy its evidence directly from the document.
Do not paraphrase or rewrite the evidence.

Return the extracted entities and relationships.
"""

        try:
            # Step 2: Extract structured information using Qwen3.
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=ExtractionResult.model_json_schema(),
                options={
                    "temperature": 0,
                    "num_predict": 700,
                },
                think=False,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Qwen3 extraction failed using model "
                f"'{self.model_name}': {exc}"
            ) from exc

        response_text = response.message.content

        if not response_text or not response_text.strip():
            raise ValueError("Qwen3 returned an empty response.")

        # Step 3: Parse the structured JSON response.
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Qwen3 returned invalid JSON: {exc}"
            ) from exc

        # Step 4: Recover explicitly mentioned missing relationship endpoints.
        response_data = _add_missing_relationship_entities(
            response_data,
            text,
            candidates,
        )

        # Step 5: Validate entities and relationships using Pydantic.
        result = ExtractionResult.model_validate(response_data)

        # Step 6: Verify evidence using whitespace-normalized exact matching.
        normalized_document = _normalize_evidence_text(text)

        for relationship in result.relationships:
            normalized_evidence = _normalize_evidence_text(
                relationship.evidence
            )

            if (
                not normalized_evidence
                or normalized_evidence not in normalized_document
            ):
                raise ValueError(
                    "Relationship evidence was not found in the original "
                    "text, even after whitespace normalization: "
                    f"{relationship.evidence}"
                )

        # Step 7: Refine generic entity types using document context.
        refined_result = self.entity_type_refiner.refine(
            text=text,
            extraction_result=result,
        )

        return refined_result
