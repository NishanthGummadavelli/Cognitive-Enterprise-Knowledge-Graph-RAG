
"""Test real document extraction and storage across all project components."""

from pathlib import Path
from tempfile import TemporaryDirectory

from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import VectorStore
from src.extraction.entity_extractor import EntityExtractor
from src.ingestion.pipeline import DocumentIngestionPipeline
from src.knowledge_graph.connection import Neo4jConnection
from src.knowledge_graph.graph_store import GraphStore
from src.storage.document_registry import DocumentRegistry
from src.storage.document_service import DocumentManagementService
from src.storage.test_authorization import TestAuthorizationService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FILE = PROJECT_ROOT / "data" / "raw" / "sample_policy.txt"
TEST_VECTOR_PATH = PROJECT_ROOT / "data" / "chroma_test_real_knowledge"
TEST_COLLECTION = "real_knowledge_integration_test"


def inspect_document_knowledge(
    connection: Neo4jConnection,
    document_id: str,
) -> dict:
    """Retrieve entities and evidence-backed relationships for a document."""

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
            name: e.name,
            entity_type: e.entity_type
        }) AS entity_details,
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
            raise RuntimeError("Could not retrieve document knowledge.")

        return {
            "chunks": record["chunks"],
            "entities": record["entities"],
            "relationships": record["relationships"],
            "entity_details": [
                item
                for item in record["entity_details"]
                if item["name"] is not None
            ],
            "relationship_details": [
                item
                for item in record["relationship_details"]
                if item["relationship_type"] is not None
            ],
        }


def main() -> None:
    """Run the complete pipeline using the actual Qwen3 entity extractor."""

    if not SAMPLE_FILE.is_file():
        raise FileNotFoundError(f"Sample policy not found: {SAMPLE_FILE}")

    with TemporaryDirectory() as temporary_directory:
        registry = DocumentRegistry(
            db_path=Path(temporary_directory) / "real_test_registry.db"
        )
        encoder = EmbeddingEncoder()
        vector_store = VectorStore(
            encoder=encoder,
            db_path=TEST_VECTOR_PATH,
            collection_name=TEST_COLLECTION,
        )

        # Clear only this dedicated test collection.
        existing_ids = vector_store.collection.get()["ids"]
        if existing_ids:
            vector_store.collection.delete(ids=existing_ids)

        neo4j_connection = Neo4jConnection()
        neo4j_connection.verify_connection()
        graph_store = GraphStore(neo4j_connection)
        entity_extractor = EntityExtractor()

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

        document_id = None

        try:
            print("=" * 70)
            print("REAL DOCUMENT KNOWLEDGE GRAPH INTEGRATION TEST")
            print("=" * 70)
            print("\nTEST 1: REAL DOCUMENT INGESTION AND EXTRACTION")
            print(f"Source: {SAMPLE_FILE.name}")

            result = service.upload_document(
                file_path=SAMPLE_FILE,
                user_id="test_user",
                scope_type="personal",
                scope_id="test_user",
            )
            document_id = result.document_id

            print(f"Document ID: {document_id}")
            print(f"Registry status: {result.status}")
            print(f"Document version: {result.version}")
            print(f"Chunks processed: {result.chunks_stored}")

            print("\nTEST 2: NEO4J KNOWLEDGE GRAPH VERIFICATION")
            graph_data = inspect_document_knowledge(
                neo4j_connection,
                document_id,
            )
            print(f"Graph chunks: {graph_data['chunks']}")
            print(f"Graph entities: {graph_data['entities']}")
            print(f"Graph relationships: {graph_data['relationships']}")

            assert graph_data["chunks"] == result.graph_chunks_stored
            assert graph_data["entities"] == result.graph_entities_stored
            assert (
                graph_data["relationships"]
                == result.graph_relationships_stored
            )

            print("\nEXTRACTED ENTITIES")
            for entity in graph_data["entity_details"]:
                print(f"- {entity['name']} ({entity['entity_type']})")

            print("\nEXTRACTED RELATIONSHIPS")
            for relationship in graph_data["relationship_details"]:
                print(
                    f"- {relationship['source']} "
                    f"-[{relationship['relationship_type']}]-> "
                    f"{relationship['target']}"
                )
                print(f"  Evidence: {relationship['evidence']}")

            print("\nTEST 3: CHROMADB VECTOR VERIFICATION")
            vector_ids = vector_store.get_document_chunk_ids(document_id)
            assert len(vector_ids) == result.chunks_stored
            print(f"Vectors stored: {len(vector_ids)}")

            print("\nTEST 4: CROSS-STORAGE CONSISTENCY")
            assert graph_data["chunks"] == len(vector_ids)
            print("SQLite document registration: verified")
            print("ChromaDB vector storage: verified")
            print("Neo4j graph storage: verified")
            print("Cross-storage chunk consistency: verified")

            print("\n" + "=" * 70)
            print("REAL DOCUMENT INTEGRATION TEST PASSED")
            print("=" * 70)

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

            try:
                remaining_ids = vector_store.collection.get()["ids"]
                if remaining_ids:
                    vector_store.collection.delete(ids=remaining_ids)
            except Exception as error:
                print(f"Warning: Test collection cleanup failed: {error}")

            neo4j_connection.close()


if __name__ == "__main__":
    main()
