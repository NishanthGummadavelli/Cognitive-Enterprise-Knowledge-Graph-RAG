
"""Test entity and relationship persistence through DocumentManagementService."""

from pathlib import Path
from tempfile import TemporaryDirectory

from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import VectorStore
from src.extraction.schemas import Entity, ExtractionResult, Relationship
from src.ingestion.pipeline import DocumentIngestionPipeline
from src.knowledge_graph.connection import Neo4jConnection
from src.knowledge_graph.graph_store import GraphStore
from src.storage.document_registry import DocumentRegistry
from src.storage.document_service import DocumentManagementService
from src.storage.test_authorization import TestAuthorizationService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_VECTOR_PATH = PROJECT_ROOT / "data" / "chroma_test_knowledge"
TEST_COLLECTION = "document_knowledge_integration_test"


class FakeEntityExtractor:
    """Return deterministic entities and relationships for integration testing."""

    def __init__(self) -> None:
        """Initialize the extraction call counter."""

        self.calls = 0

    def extract(self, text: str) -> ExtractionResult:
        """Extract predefined knowledge supported by the supplied chunk."""

        self.calls += 1
        evidence = "Human Resources uses Workday to manage employee records."

        if evidence not in text:
            return ExtractionResult(entities=[], relationships=[])

        return ExtractionResult(
            entities=[
                Entity(name="Human Resources", entity_type="DEPARTMENT"),
                Entity(name="Workday", entity_type="TECHNOLOGY"),
            ],
            relationships=[
                Relationship(
                    source="Human Resources",
                    relationship_type="USES",
                    target="Workday",
                    evidence=evidence,
                )
            ],
        )


def get_knowledge_graph_counts(
    connection: Neo4jConnection,
    document_id: str,
) -> dict:
    """Retrieve entity and relationship evidence associated with a document."""

    query = """
    MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
    OPTIONAL MATCH (c)-[:SUPPORTS]->(r:KnowledgeRelationship)-[:SOURCE]->(source:Entity)
    OPTIONAL MATCH (r)-[:TARGET]->(target:Entity)
    RETURN
        count(DISTINCT c) AS chunks,
        count(DISTINCT e) AS entities,
        count(DISTINCT r) AS relationships,
        collect(DISTINCT {
            source: source.name,
            relationship_type: r.relationship_type,
            target: target.name,
            evidence: r.evidence
        }) AS relationship_details
    """

    with connection.driver.session(database=connection.database) as session:
        record = session.run(query, document_id=document_id).single()

        if record is None:
            raise RuntimeError("Could not retrieve knowledge graph results.")

        return {
            "chunks": record["chunks"],
            "entities": record["entities"],
            "relationships": record["relationships"],
            "relationship_details": [
                item
                for item in record["relationship_details"]
                if item["relationship_type"] is not None
            ],
        }


def main() -> None:
    """Verify that document service persists extracted knowledge in Neo4j."""

    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        registry = DocumentRegistry(
            db_path=temporary_path / "knowledge_test_registry.db"
        )
        encoder = EmbeddingEncoder()
        vector_store = VectorStore(
            encoder=encoder,
            db_path=TEST_VECTOR_PATH,
            collection_name=TEST_COLLECTION,
        )
        neo4j_connection = Neo4jConnection()
        neo4j_connection.verify_connection()
        graph_store = GraphStore(neo4j_connection)
        entity_extractor = FakeEntityExtractor()

        authorization_service = TestAuthorizationService(
            personal_users={"test_user"},
        )

        service = DocumentManagementService(
            ingestion_pipeline=DocumentIngestionPipeline(),
            registry=registry,
            vector_store=vector_store,
            graph_store=graph_store,
            entity_extractor=entity_extractor,
            authorization_service=authorization_service,
        )

        test_file = temporary_path / "knowledge_test.txt"
        test_file.write_text(
            (
                "Enterprise HR Technology Policy\n\n"
                "Human Resources uses Workday to manage employee records."
            ),
            encoding="utf-8",
        )
        document_id = None

        try:
            print("=" * 65)
            print("DOCUMENT SERVICE KNOWLEDGE INTEGRATION TEST")
            print("=" * 65)

            print("\nTEST 1: DOCUMENT INGESTION AND KNOWLEDGE EXTRACTION")
            result = service.upload_document(
                file_path=test_file,
                user_id="test_user",
                scope_type="personal",
                scope_id="test_user",
            )
            document_id = result.document_id

            print("\nDEBUG: Document service result")
            print(f"Status: {result.status}")
            print(f"Document ID: {result.document_id}")
            print(f"Version: {result.version}")
            print(f"Chunks stored: {result.chunks_stored}")
            print(f"Graph chunks stored: {result.graph_chunks_stored}")
            print(f"Graph entities stored: {result.graph_entities_stored}")
            print(
                "Graph relationships stored: "
                f"{result.graph_relationships_stored}"
            )
            assert result.chunks_stored > 0
            assert result.graph_chunks_stored > 0

            print(f"Document ID: {document_id}")
            print(f"Chunks processed: {result.chunks_stored}")
            print(f"Extractor calls: {entity_extractor.calls}")
            assert entity_extractor.calls == result.chunks_stored

            print("\nTEST 2: NEO4J ENTITY AND RELATIONSHIP PERSISTENCE")
            graph_data = get_knowledge_graph_counts(
                neo4j_connection,
                document_id,
            )
            print(f"Graph chunks: {graph_data['chunks']}")
            print(f"Graph entities: {graph_data['entities']}")
            print(f"Graph relationships: {graph_data['relationships']}")
            assert graph_data["chunks"] > 0
            assert graph_data["entities"] == 2
            assert graph_data["relationships"] == 1

            print("\nTEST 3: RELATIONSHIP EVIDENCE VALIDATION")
            relationship = graph_data["relationship_details"][0]
            assert relationship["source"] == "Human Resources"
            assert relationship["relationship_type"] == "USES"
            assert relationship["target"] == "Workday"
            assert relationship["evidence"] == (
                "Human Resources uses Workday to manage employee records."
            )
            print(
                f"{relationship['source']} "
                f"-[{relationship['relationship_type']}]-> "
                f"{relationship['target']}"
            )
            print(f"Evidence: {relationship['evidence']}")

            print("\nTEST 4: CHROMADB VECTOR PERSISTENCE")
            vector_ids = vector_store.get_document_chunk_ids(document_id)
            assert len(vector_ids) == result.chunks_stored
            print(f"Vectors stored: {len(vector_ids)}")

            print("\n" + "=" * 65)
            print("ALL KNOWLEDGE INTEGRATION TESTS PASSED")
            print("=" * 65)

        finally:
            if document_id is not None:
                try:
                    graph_store.delete_document(document_id)
                except Exception as error:
                    print(f"Warning: Neo4j cleanup failed: {error}")

                try:
                    chunk_ids = vector_store.get_document_chunk_ids(document_id)
                    if chunk_ids:
                        vector_store.delete_chunks(chunk_ids)
                except Exception as error:
                    print(f"Warning: ChromaDB cleanup failed: {error}")

            neo4j_connection.close()


if __name__ == "__main__":
    main()
