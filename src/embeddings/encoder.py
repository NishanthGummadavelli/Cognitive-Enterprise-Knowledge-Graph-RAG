
"""Generate semantic text embeddings using a BGE sentence-transformer model."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


# Selected lightweight English embedding model for semantic retrieval.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE recommends this instruction format for retrieval queries.
QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingEncoder:
    """Convert document text and user queries into semantic vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Load the selected embedding model for reuse.

        Args:
            model_name: Hugging Face model identifier.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """

        try:
            # Load the model once instead of reloading it for every request.
            self.model = SentenceTransformer(model_name)
        except Exception as error:
            raise RuntimeError(
                f"Could not load embedding model '{model_name}': {error}"
            ) from error

        self.model_name = model_name

    def encode_documents(
        self, texts: Sequence[str]
    ) -> NDArray[np.float32]:
        """Convert document chunks into normalized embedding vectors.

        Args:
            texts: Non-empty document chunks.

        Returns:
            A two-dimensional NumPy array of document embeddings.

        Raises:
            ValueError: If the input contains no usable text.
            RuntimeError: If embedding generation fails.
        """

        # Validate every document before sending it to the model.
        if not texts or any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Document texts must contain non-empty strings."
            )

        try:
            embeddings = self.model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Document embedding generation failed: {error}"
            ) from error

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Convert a user query into a normalized retrieval vector.

        Args:
            query: User's search question.

        Returns:
            A one-dimensional NumPy embedding vector.

        Raises:
            ValueError: If the query is empty.
            RuntimeError: If embedding generation fails.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        # Apply the BGE retrieval instruction to the query only.
        instructed_query = QUERY_INSTRUCTION + query.strip()

        try:
            embedding = self.model.encode(
                instructed_query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Query embedding generation failed: {error}"
            ) from error

        return np.asarray(embedding, dtype=np.float32)
