
"""Provide access to the trained enterprise document classifier."""

from pathlib import Path
from typing import Any

import joblib


# Resolve the project root independently of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "document_classifier.joblib"
)


class DocumentClassifier:
    """Load the saved classification pipeline and predict document categories."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        """Initialize the classifier using an existing trained model.

        Args:
            model_path: Path to the saved scikit-learn pipeline.

        Raises:
            FileNotFoundError: If the trained model does not exist.
            RuntimeError: If the model cannot be loaded.
        """

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Trained classification model not found: {model_path}"
            )

        try:
            self.model: Any = joblib.load(model_path)
        except Exception as error:
            raise RuntimeError(
                f"Could not load classification model: {error}"
            ) from error

    def predict(self, text: str) -> dict[str, str | float]:
        """Predict the enterprise category for a complete document.

        Args:
            text: Cleaned text extracted from the complete document.

        Returns:
            Predicted category and its highest model probability.

        Raises:
            ValueError: If the input text is empty or contains only whitespace.
            RuntimeError: If prediction fails.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Document text cannot be empty.")

        try:
            predicted_category = str(self.model.predict([text])[0])

            # Obtain the model's probability for each supported category.
            probabilities = self.model.predict_proba([text])[0]
            confidence = float(max(probabilities))

        except Exception as error:
            raise RuntimeError(
                f"Document classification failed: {error}"
            ) from error

        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
        }
