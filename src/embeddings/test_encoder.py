
"""Test BGE document and query embedding generation."""

import numpy as np

from src.embeddings.encoder import EmbeddingEncoder


def main() -> None:
    """Generate and inspect document and query embeddings."""

    # Initialize the model once.
    encoder = EmbeddingEncoder()

    # Use a few sample enterprise document chunks.
    documents = [
        "Employees are entitled to annual leave according to company policy.",
        "The HR department manages employee recruitment and attendance.",
        "The IT team maintains servers and network infrastructure.",
    ]

    # Generate embeddings for all document chunks.
    document_vectors = encoder.encode_documents(documents)

    # Generate an embedding for a sample user question.
    query = "How much annual leave can employees take?"
    query_vector = encoder.encode_query(query)

    print("=" * 60)
    print("BGE EMBEDDING TEST")
    print("=" * 60)

    print(f"Model: {encoder.model_name}")
    print(f"Number of documents: {len(documents)}")
    print(f"Document embedding shape: {document_vectors.shape}")
    print(f"Query embedding shape: {query_vector.shape}")
    print(f"Vector data type: {document_vectors.dtype}")

    # Verify that the generated vectors are normalized.
    print(
        "Document vector norms:",
        np.linalg.norm(document_vectors, axis=1),
    )
    print("Query vector norm:", np.linalg.norm(query_vector))

    # Calculate cosine similarities using normalized vectors.
    similarities = document_vectors @ query_vector

    print("\nQUERY SIMILARITY RESULTS")
    print("-" * 60)

    for index, similarity in enumerate(similarities):
        print(f"Document {index + 1}: {similarity:.4f}")
        print(f"Text: {documents[index]}\n")

    print("EMBEDDING TEST COMPLETED")


if __name__ == "__main__":
    main()
