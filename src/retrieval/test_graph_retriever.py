
"""Integration tests for scope-aware Neo4j graph retrieval."""

import uuid

from src.knowledge_graph.connection import Neo4jConnection
from src.retrieval.graph_retriever import GraphRetriever


def create_test_graph(
    connection: Neo4jConnection,
    test_run_id: str,
    scope_type: str,
    scope_id: str,
    uploaded_by: str,
    document_name: str,
    entity_names: list[str],
    relationship: bool = False,
) -> str:
    """Create temporary scoped chunks, entities and optional relationship evidence."""

    chunk_id = f"{test_run_id}_{scope_type}_{scope_id}_chunk"
    document_id = f"{test_run_id}_{scope_type}_{scope_id}_document"

    entities = [
        {
            "entity_id": f"{chunk_id}_entity_{index}",
            "name": name,
            "normalized_name": " ".join(name.casefold().split()),
            "entity_type": "TECHNOLOGY",
        }
        for index, name in enumerate(entity_names)
    ]

    with connection.driver.session(
        database=connection.database
    ) as session:
        session.run(
            """
            CREATE (c:Chunk {
                chunk_id: $chunk_id,
                document_id: $document_id,
                text: $text,
                source: $source,
                page_numbers: [1],
                heading_path: ['Test'],
                chunk_index: 0,
                scope_type: $scope_type,
                scope_id: $scope_id,
                uploaded_by: $uploaded_by,
                test_run_id: $test_run_id
            })

            WITH c
            UNWIND $entities AS item

            CREATE (e:Entity {
                entity_id: item.entity_id,
                name: item.name,
                normalized_name: item.normalized_name,
                entity_type: item.entity_type,
                scope_type: $scope_type,
                scope_id: $scope_id,
                uploaded_by: $uploaded_by,
                test_run_id: $test_run_id
            })

            CREATE (c)-[:MENTIONS]->(e)

            RETURN count(e) AS entities_created
            """,
            chunk_id=chunk_id,
            document_id=document_id,
            text=document_name,
            source=f"{document_name}.txt",
            entities=entities,
            scope_type=scope_type,
            scope_id=scope_id,
            uploaded_by=uploaded_by,
            test_run_id=test_run_id,
        ).consume()

        if relationship:
            relationship_id = f"{chunk_id}_relationship"

            session.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})

                MATCH (source:Entity {
                    entity_id: $source_entity_id
                })

                MATCH (target:Entity {
                    entity_id: $target_entity_id
                })

                CREATE (r:KnowledgeRelationship {
                    relationship_id: $relationship_id,
                    relationship_type: 'PROCESSES',
                    evidence: $evidence,
                    document_id: $document_id,
                    chunk_id: $chunk_id,
                    scope_type: $scope_type,
                    scope_id: $scope_id,
                    uploaded_by: $uploaded_by,
                    test_run_id: $test_run_id
                })

                CREATE (c)-[:SUPPORTS]->(r)
                CREATE (r)-[:SOURCE]->(source)
                CREATE (r)-[:TARGET]->(target)
                """,
                chunk_id=chunk_id,
                source_entity_id=entities[0]["entity_id"],
                target_entity_id=entities[1]["entity_id"],
                relationship_id=relationship_id,
                evidence=(
                    f"{entity_names[0]} processes messages "
                    f"through {entity_names[1]}."
                ),
                document_id=document_id,
                scope_type=scope_type,
                scope_id=scope_id,
                uploaded_by=uploaded_by,
                test_run_id=test_run_id,
            ).consume()

    return chunk_id


def cleanup_test_graph(
    connection: Neo4jConnection,
    test_run_id: str,
) -> None:
    """Remove only temporary graph records created by this test run."""

    with connection.driver.session(
        database=connection.database
    ) as session:
        session.run(
            """
            MATCH (node)
            WHERE node.test_run_id = $test_run_id
            DETACH DELETE node
            """,
            test_run_id=test_run_id,
        ).consume()


def main() -> None:
    """Run graph retrieval and scope-isolation integration tests."""

    print("=" * 60)
    print("NEO4J GRAPH RETRIEVAL TEST")
    print("=" * 60)

    connection = Neo4jConnection()
    test_run_id = f"graph_retrieval_test_{uuid.uuid4().hex}"

    try:
        connection.verify_connection()

        retriever = GraphRetriever(connection)

        # Create three isolated knowledge spaces.
        create_test_graph(
            connection=connection,
            test_run_id=test_run_id,
            scope_type="personal",
            scope_id=f"{test_run_id}_user_a",
            uploaded_by=f"{test_run_id}_user_a",
            document_name=(
                "AWS Lambda processes messages through Amazon SQS."
            ),
            entity_names=["AWS Lambda", "Amazon SQS"],
            relationship=True,
        )

        create_test_graph(
            connection=connection,
            test_run_id=test_run_id,
            scope_type="personal",
            scope_id=f"{test_run_id}_user_b",
            uploaded_by=f"{test_run_id}_user_b",
            document_name=(
                "PostgreSQL is a relational database management system."
            ),
            entity_names=["PostgreSQL"],
        )

        create_test_graph(
            connection=connection,
            test_run_id=test_run_id,
            scope_type="project",
            scope_id=f"{test_run_id}_project_alpha",
            uploaded_by=f"{test_run_id}_project_uploader",
            document_name=(
                "AWS Lambda consumes messages from Amazon SQS."
            ),
            entity_names=["AWS Lambda", "Amazon SQS"],
            relationship=True,
        )

        print("\nThree scoped test knowledge spaces created.")

        user_a = f"{test_run_id}_user_a"
        user_b = f"{test_run_id}_user_b"
        project_alpha = f"{test_run_id}_project_alpha"

        # TEST 1: Retrieve matching personal knowledge.
        user_a_results = retriever.search(
            query="AWS Lambda",
            scope_type="personal",
            scope_id=user_a,
        )

        assert len(user_a_results) == 1
        assert user_a_results[0]["scope_id"] == user_a
        assert "AWS Lambda" in user_a_results[0]["text"]

        print("TEST 1 PASSED: Personal entity retrieval works.")

        # TEST 2: Verify relationship evidence is retrieved.
        relationships = user_a_results[0]["relationships"]

        assert any(
            item is not None
            and item["relationship_type"] == "PROCESSES"
            for item in relationships
        )

        print("TEST 2 PASSED: Relationship evidence is retrieved.")

        # TEST 3: User B must not retrieve User A's AWS information.
        user_b_aws_results = retriever.search(
            query="AWS Lambda",
            scope_type="personal",
            scope_id=user_b,
        )

        assert user_b_aws_results == []

        print("TEST 3 PASSED: Personal scope isolation works.")

        # TEST 4: User B can retrieve their own PostgreSQL information.
        user_b_results = retriever.search(
            query="PostgreSQL",
            scope_type="personal",
            scope_id=user_b,
        )

        assert len(user_b_results) == 1
        assert user_b_results[0]["scope_id"] == user_b
        assert "PostgreSQL" in user_b_results[0]["text"]

        print("TEST 4 PASSED: User B retrieves their own knowledge.")

        # TEST 5: Project retrieval must remain inside Project Alpha.
        project_results = retriever.search(
            query="AWS Lambda",
            scope_type="project",
            scope_id=project_alpha,
        )

        assert len(project_results) == 1
        assert project_results[0]["scope_type"] == "project"
        assert project_results[0]["scope_id"] == project_alpha

        print("TEST 5 PASSED: Project scope retrieval works.")

        # TEST 6: Unknown scopes must not return another scope's data.
        unknown_results = retriever.search(
            query="AWS Lambda",
            scope_type="personal",
            scope_id=f"{test_run_id}_unknown_user",
        )

        assert unknown_results == []

        print("TEST 6 PASSED: Unknown scope returns no results.")

        # TEST 7: Empty queries must be rejected.
        try:
            retriever.search(
                query=" ",
                scope_type="personal",
                scope_id=user_a,
            )
            raise AssertionError("Empty query should have been rejected.")
        except ValueError:
            pass

        print("TEST 7 PASSED: Empty query is rejected.")

        # TEST 8: Invalid scope types must be rejected.
        try:
            retriever.search(
                query="AWS Lambda",
                scope_type="invalid",
                scope_id=user_a,
            )
            raise AssertionError("Invalid scope should have been rejected.")
        except ValueError:
            pass

        print("TEST 8 PASSED: Invalid scope is rejected.")

        print("\nALL GRAPH RETRIEVAL TESTS PASSED")

    finally:
        try:
            cleanup_test_graph(connection, test_run_id)
            print("\nTemporary test graph records cleaned up.")
        finally:
            connection.driver.close()


if __name__ == "__main__":
    main()
