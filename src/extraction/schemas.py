
"""
Pydantic schemas for validating extracted entities and relationships.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Entity(BaseModel):
    """Represents an entity discovered in an enterprise document."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="The name of the discovered entity."
    )
    entity_type: str = Field(
        ...,
        min_length=1,
        description="A category such as PERSON, PROJECT, or TECHNOLOGY."
    )

    @field_validator("name", "entity_type")
    @classmethod
    def clean_required_fields(cls, value: str) -> str:
        """Remove surrounding whitespace and reject empty values."""

        value = value.strip()

        if not value:
            raise ValueError("Entity fields cannot be empty.")

        return value

    @field_validator("entity_type")
    @classmethod
    def normalize_entity_type(cls, value: str) -> str:
        """Normalize entity categories to uppercase."""

        return value.upper()


class Relationship(BaseModel):
    """Represents a connection between two extracted entities."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        ...,
        min_length=1,
        description="Name of the source entity."
    )
    relationship_type: str = Field(
        ...,
        min_length=1,
        description="The relationship connecting the two entities."
    )
    target: str = Field(
        ...,
        min_length=1,
        description="Name of the target entity."
    )
    evidence: str = Field(
        ...,
        min_length=1,
        description="Exact supporting text from the original document."
    )

    @field_validator("source", "target", "relationship_type", "evidence")
    @classmethod
    def clean_required_fields(cls, value: str) -> str:
        """Remove surrounding whitespace and reject empty values."""

        value = value.strip()

        if not value:
            raise ValueError("Relationship fields cannot be empty.")

        return value

    @field_validator("relationship_type")
    @classmethod
    def normalize_relationship_type(cls, value: str) -> str:
        """Normalize relationship labels to uppercase."""

        return value.upper()


class ExtractionResult(BaseModel):
    """Contains all entities and relationships extracted from one text chunk."""

    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_relationship_entities(self):
        """Ensure entity names are unique and relationships resolve to them."""

        entity_names: set[str] = set()

        for entity in self.entities:
            normalized_name = entity.name.casefold()

            if normalized_name in entity_names:
                raise ValueError(f"Duplicate entity name '{entity.name}'.")

            entity_names.add(normalized_name)

        for relationship in self.relationships:
            if relationship.source.casefold() not in entity_names:
                raise ValueError(
                    f"Source entity '{relationship.source}' was not extracted."
                )

            if relationship.target.casefold() not in entity_names:
                raise ValueError(
                    f"Target entity '{relationship.target}' was not extracted."
                )

        return self


class EntityTypeClassification(BaseModel):
    """Represents the refined category of an extracted entity."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="The exact name of the entity being classified."
    )
    entity_type: str = Field(
        ...,
        min_length=1,
        description="The most appropriate enterprise entity category."
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Brief explanation supported by document context."
    )

    @field_validator("name", "entity_type", "reason")
    @classmethod
    def clean_required_fields(cls, value: str) -> str:
        """Remove surrounding whitespace and reject empty values."""

        value = value.strip()

        if not value:
            raise ValueError("Classification fields cannot be empty.")

        return value

    @field_validator("entity_type")
    @classmethod
    def normalize_entity_type(cls, value: str) -> str:
        """Normalize entity categories to uppercase."""

        return value.upper()


class EntityTypeRefinementResult(BaseModel):
    """Contains refined classifications for unresolved entities."""

    model_config = ConfigDict(extra="forbid")

    classifications: list[EntityTypeClassification] = Field(
        default_factory=list
    )
