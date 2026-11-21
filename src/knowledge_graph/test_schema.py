
"""Verify that the Neo4j knowledge graph schema was created correctly."""

from src.knowledge_graph.connection import Neo4jConnection
from src.knowledge_graph.schema import (
    SCHEMA_CONSTRAINTS,
    SCHEMA_INDEXES,
    create_graph_schema,
)


def test_graph_schema() -> None:
    """Create the schema and verify its constraints and indexes."""

    connection = Neo4jConnection()

    try:
        connection.verify_connection()

        # Create the schema.
        create_graph_schema(connection)

        with connection.driver.session(
            database=connection.database
        ) as session:

            # Retrieve existing database constraints.
            constraints = list(
                session.run("SHOW CONSTRAINTS YIELD name RETURN name")
            )

            # Retrieve existing database indexes.
            indexes = list(
                session.run("SHOW INDEXES YIELD name RETURN name")
            )

            constraint_names = {
                record["name"] for record in constraints
            }
            index_names = {
                record["name"] for record in indexes
            }

            expected_constraints = {
                "document_id_unique",
                "chunk_id_unique",
                "entity_id_unique",
                "relationship_id_unique",
            }

            expected_indexes = {
                "entity_name_index",
                "entity_type_index",
            }

            # Verify that every expected schema object exists.
            assert expected_constraints.issubset(constraint_names), (
                "One or more required constraints are missing."
            )

            assert expected_indexes.issubset(index_names), (
                "One or more required indexes are missing."
            )

            print("All 4 uniqueness constraints verified.")
            print("All 2 indexes verified.")
            print("Knowledge Graph schema test passed.")

    finally:
        connection.close()


if __name__ == "__main__":
    test_graph_schema()

