
"""Test the connection between Python and Neo4j AuraDB."""

from src.knowledge_graph.connection import Neo4jConnection


def test_neo4j_connection() -> None:
    """Verify Neo4j connectivity and execute a basic database query."""

    connection = Neo4jConnection()

    try:
        # Verify credentials and database connectivity.
        connection.verify_connection()

        # Execute a simple query to confirm database access.
        with connection.driver.session(
            database=connection.database
        ) as session:
            result = session.run("RETURN 'Neo4j Connected Successfully!' AS message")
            record = result.single()

            if record is None:
                raise RuntimeError("Neo4j returned an empty query result.")

            print(record["message"])

    finally:
        # Always release the database connection.
        connection.close()


if __name__ == "__main__":
    test_neo4j_connection()
