
"""Define and initialize the Neo4j enterprise knowledge graph schema."""

from src.knowledge_graph.connection import Neo4jConnection


# Uniqueness constraints protect stable identities and prevent duplicates.
SCHEMA_CONSTRAINTS = [
    """
    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
    FOR (d:Document)
    REQUIRE d.document_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
    FOR (c:Chunk)
    REQUIRE c.chunk_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE e.entity_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS
    FOR (r:KnowledgeRelationship)
    REQUIRE r.relationship_id IS UNIQUE
    """,
]

# Indexes support efficient entity searches and filtering.
SCHEMA_INDEXES = [
    """
    CREATE INDEX entity_name_index IF NOT EXISTS
    FOR (e:Entity)
    ON (e.name)
    """,
    """
    CREATE INDEX entity_type_index IF NOT EXISTS
    FOR (e:Entity)
    ON (e.entity_type)
    """,
]


def create_graph_schema(connection: Neo4jConnection) -> None:
    """Create graph uniqueness constraints and indexes if they do not exist."""

    with connection.driver.session(database=connection.database) as session:

        # Create uniqueness constraints first.
        for statement in SCHEMA_CONSTRAINTS:
            session.run(statement).consume()

        # Create indexes for frequently searched entity properties.
        for statement in SCHEMA_INDEXES:
            session.run(statement).consume()

    print("Knowledge Graph schema created successfully.")


if __name__ == "__main__":
    neo4j_connection = Neo4jConnection()

    try:
        neo4j_connection.verify_connection()
        create_graph_schema(neo4j_connection)
    finally:
        neo4j_connection.close()
