"""Validate the collected Projects & Management classification dataset."""

import csv
from collections import Counter
from pathlib import Path

from src.classification.categories import CATEGORIES


DATASET_PATH = Path("data/training/project_management.csv")

EXPECTED_DOCUMENT_TYPES = {
    "project_proposal": 20,
    "project_report": 20,
    "meeting_minutes": 15,
    "change_request": 10,
    "lessons_learned": 10,
    "risk_assessment": 10,
    "feasibility_study": 5,
}

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
    "original_file",
    "issuer",
    "issuer_country",
    "recipient",
    "document_date",
    "blueprint_id",
    "original_document_number",
}


def validate_project_dataset() -> bool:
    """Check dataset structure, labels, metadata, uniqueness and distribution.

    Returns:
        True if no validation errors are found; otherwise False.
    """

    # Allow long document text fields to be read from the CSV.
    csv.field_size_limit(100_000_000)

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return False

    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            print("ERROR: CSV file has no header.")
            return False

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            print(f"ERROR: Missing columns: {sorted(missing_columns)}")
            return False

        rows = list(reader)

    errors: list[str] = []
    warnings: list[str] = []

    document_ids = [row["document_id"].strip() for row in rows]
    document_types = Counter(row["document_type"].strip() for row in rows)
    labels = Counter(row["label"].strip() for row in rows)
    source_types = Counter(row["source_type"].strip() for row in rows)
    locales = Counter(row["locale"].strip() for row in rows)

    # Check required text and metadata fields.
    required_values = [
        "text",
        "document_id",
        "document_type",
        "source_name",
        "source_url",
        "license",
        "original_file",
        "issuer",
        "blueprint_id",
    ]

    for row_number, row in enumerate(rows, start=2):
        for column in required_values:
            if not row.get(column, "").strip():
                errors.append(
                    f"Row {row_number}: Missing value in '{column}'."
                )

        if row.get("label", "").strip() != "Projects & Management":
            errors.append(f"Row {row_number}: Incorrect classification label.")

        if row.get("source_type", "").strip() != "external_dataset":
            errors.append(f"Row {row_number}: Incorrect source type.")

        if row.get("locale", "").strip() != "en":
            errors.append(f"Row {row_number}: Non-English or missing locale.")

        if row.get("document_type", "").strip() not in EXPECTED_DOCUMENT_TYPES:
            errors.append(
                f"Row {row_number}: Unexpected document type "
                f"'{row.get('document_type', '')}'."
            )

    # Check duplicate IDs and duplicate document text.
    if len(document_ids) != len(set(document_ids)):
        errors.append("Duplicate document IDs found.")

    normalized_texts = [
        " ".join(row["text"].lower().split())
        for row in rows
    ]

    if len(normalized_texts) != len(set(normalized_texts)):
        errors.append("Duplicate document text found.")

    # Check that the collected distribution matches the planned targets.
    if len(rows) != 90:
        errors.append(f"Expected 90 records, but found {len(rows)}.")

    if dict(document_types) != EXPECTED_DOCUMENT_TYPES:
        errors.append(
            "Document type distribution does not match the expected targets."
        )

    if labels != Counter({"Projects & Management": 90}):
        errors.append("Classification label distribution is incorrect.")

    if source_types != Counter({"external_dataset": 90}):
        errors.append("Source type distribution is incorrect.")

    if locales != Counter({"en": 90}):
        errors.append("Locale distribution is incorrect.")

    # Display the validation report.
    print("\n" + "=" * 55)
    print("PROJECT MANAGEMENT DATASET VALIDATION")
    print("=" * 55)

    print(f"\nTotal records: {len(rows)}")
    print(f"Unique document IDs: {len(set(document_ids))}")
    print(f"Unique document texts: {len(set(normalized_texts))}")

    print("\nDocument type distribution:")
    for document_type, count in sorted(document_types.items()):
        print(f"  {document_type:<25} {count}")

    print("\nClassification labels:")
    for label, count in labels.items():
        print(f"  {label:<25} {count}")

    print("\nSource types:")
    for source_type, count in source_types.items():
        print(f"  {source_type:<25} {count}")

    print("\nErrors:", len(errors))
    for error in errors:
        print("  -", error)

    print("\nWarnings:", len(warnings))
    for warning in warnings:
        print("  -", warning)

    if errors:
        print("\nPROJECT DATASET VALIDATION FAILED")
        return False

    print("\nPROJECT DATASET VALIDATION PASSED")
    return True


if __name__ == "__main__":
    validate_project_dataset()
