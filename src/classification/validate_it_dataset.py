
"""Validate the collected IT documentation training dataset."""

import csv
import re
from collections import Counter
from pathlib import Path

from src.classification.categories import CATEGORY_SET


# Dataset location and expected structure.
DATASET_PATH = Path("data/training/it_documents.csv")

REQUIRED_COLUMNS = {
    "text",
    "label",
    "source_type",
    "source_name",
    "source_url",
    "license",
    "document_id",
    "document_type",
    "locale",
    "subject_area",
    "source_site",
    "topic",
}

VALID_SUBJECT_AREAS = {
    "Security & Secrets",
    "Monitoring & Observability",
    "Networking",
    "Databases & Caching",
    "AI & Machine Learning",
    "Operating Systems",
    "Containers & Orchestration",
    "Cloud & Infrastructure",
    "DevOps & CI/CD",
    "General IT Documentation",
}


def normalize_text(text: str) -> str:
    """Normalize document text for duplicate detection."""

    return re.sub(r"\s+", " ", text).strip().lower()


def validate_it_dataset() -> bool:
    """Check the IT dataset for missing, invalid and duplicate records."""

    errors: list[str] = []
    warnings: list[str] = []

    # Check that the dataset file exists.
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return False

    # Load the CSV and check its column names.
    with DATASET_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            print("ERROR: Dataset has no header row.")
            return False

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            print(f"ERROR: Missing columns: {sorted(missing_columns)}")
            return False

        rows = list(reader)

    if not rows:
        print("ERROR: Dataset contains no records.")
        return False

    # Track IDs, document text and category distributions.
    document_ids: set[str] = set()
    text_hashes: set[str] = set()

    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=2):
        text = row.get("text", "").strip()
        label = row.get("label", "").strip()
        document_id = row.get("document_id", "").strip()
        source_site = row.get("source_site", "").strip()
        subject_area = row.get("subject_area", "").strip()

        # Check required values.
        required_values = {
            "text": text,
            "label": label,
            "document_id": document_id,
            "source_site": source_site,
            "subject_area": subject_area,
            "topic": row.get("topic", "").strip(),
            "source_url": row.get("source_url", "").strip(),
            "license": row.get("license", "").strip(),
        }

        for field, value in required_values.items():
            if not value:
                errors.append(
                    f"Row {row_number}: Missing value in '{field}'."
                )

        # Validate the official classifier category.
        if label not in CATEGORY_SET:
            errors.append(
                f"Row {row_number}: Invalid classification label '{label}'."
            )
        elif label != "IT":
            errors.append(
                f"Row {row_number}: Expected IT label, found '{label}'."
            )

        # Validate the subject-area metadata.
        if subject_area and subject_area not in VALID_SUBJECT_AREAS:
            errors.append(
                f"Row {row_number}: Invalid subject area '{subject_area}'."
            )

        # Check document length.
        if text and len(text) < 500:
            warnings.append(
                f"Row {row_number}: Document text is shorter than 500 characters."
            )

        # Detect duplicate document IDs.
        if document_id:
            if document_id in document_ids:
                errors.append(
                    f"Row {row_number}: Duplicate document ID '{document_id}'."
                )
            document_ids.add(document_id)

        # Detect exact duplicate document text.
        if text:
            normalized = normalize_text(text)

            if normalized in text_hashes:
                warnings.append(
                    f"Row {row_number}: Duplicate document text detected."
                )
            text_hashes.add(normalized)

        label_counts[label] += 1
        source_counts[source_site] += 1
        subject_counts[subject_area] += 1

    # Display the validation report.
    print("\nIT DATASET VALIDATION REPORT")
    print("=" * 45)
    print(f"Total documents: {len(rows)}")
    print(f"Unique document IDs: {len(document_ids)}")
    print(f"Unique document texts: {len(text_hashes)}")

    print("\nClassification distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("\nDocumentation source distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    print("\nSubject-area distribution:")
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count}")

    print(f"\nErrors: {len(errors)}")
    for error in errors:
        print(f"  ERROR: {error}")

    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings:
        print(f"  WARNING: {warning}")

    # Validation passes only when no errors are present.
    if errors:
        print("\nIT DATASET VALIDATION FAILED.")
        return False

    print("\nIT DATASET VALIDATION PASSED.")
    return True


if __name__ == "__main__":
    validate_it_dataset()
