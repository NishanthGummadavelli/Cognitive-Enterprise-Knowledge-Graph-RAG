
"""Manage secure connections to the Neo4j AuraDB instance."""

import os
import ssl
from pathlib import Path

import certifi
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase


# Load environment variables from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Neo4jConnection:
    """Create, verify, and close the Neo4j database connection."""

    def __init__(self) -> None:
        """Initialize the Neo4j driver using environment credentials."""

        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        # Validate required configuration.
        if not all([self.uri, self.username, self.password]):
            raise ValueError(
                "Missing Neo4j configuration. Check your .env file."
            )

        # Use a verified CA bundle instead of the default certificate paths.
        ssl_context = ssl.create_default_context(
            cafile=certifi.where()
        )

        # Use explicit TLS configuration with certificate verification.
        # neo4j:// is required when supplying a custom SSL context.
        secure_uri = self.uri.replace("neo4j+s://", "neo4j://", 1)

        self.driver: Driver = GraphDatabase.driver(
            secure_uri,
            auth=(self.username, self.password),
            encrypted=True,
            ssl_context=ssl_context,
        )

    def verify_connection(self) -> bool:
        """Verify connectivity and database authentication."""

        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close the driver and release database resources."""

        self.driver.close()
