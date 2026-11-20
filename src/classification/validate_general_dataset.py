"""Validate the General & Administration classification dataset."""

import csv
import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path("data/training/general_administration.csv")

EXPECTED_TYPES = {
    "contact_card": 20,
    "meeting_invite": 20,
    "event_invitation": 15,
    "lost_and_found_report": 15,
    "school_notice": 10,
    "car_rental_booking": 5,
    "flight_booking": 5,
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
    "source_record_id",
    "structured_extraction",
}


def validate_general_dataset() -> bool:
    """Validate records, metadata, uniqueness and document-type distribution.

    Returns:
        True if all validation checks pass; otherwise False.
    """

    # Allow CSV fields containing long document text.
    csv.field_size_limit(100_000_000)

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return False

    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            print("ERROR: CSV has no header.")
            return False

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            print("ERROR: Missing columns:", sorted(missing_columns))
            return False

        rows = list(reader)

    errors: list[str] = []

    document_ids = [row["document_id"].strip() for row in rows]
    source_ids = [row["source_record_id"].strip() for row in rows]
    document_types = Counter(row["document_type"].strip() for row in rows)
    labels = Counter(row["label"].strip() for row in rows)
    locales = Counter(row["locale"].strip() for row in rows)
    source_types = Counter(row["source_type"].strip() for row in rows)

    required_values = [
        "text",
        "document_id",
        "source_record_id",
        "document_type",
        "source_name",
        "source_url",
        "license",
        "topic",
        "structured_extraction",
    ]

    # Validate required metadata and structured extraction JSON.
    for row_number, row in enumerate(rows, start=2):
        for column in required_values:
            if not row.get(column, "").strip():
                errors.append(
                    f"Row {row_number}: Missing value in '{column}'."
                )

        if row.get("label", "").strip() != "General & Administration":
            errors.append(f"Row {row_number}: Incorrect classification label.")

        if row.get("source_type", "").strip() != "external_dataset":
            errors.append(f"Row {row_number}: Incorrect source type.")

        if row.get("locale", "").strip() != "en":
            errors.append(f"Row {row_number}: Locale is not English.")

        if row.get("subject_area", "").strip() != "General Administration":
            errors.append(f"Row {row_number}: Incorrect subject area.")

        try:
            json.loads(row["structured_extraction"])
        except (json.JSONDecodeError, TypeError):
            errors.append(
                f"Row {row_number}: Invalid structured extraction JSON."
            )

        if row.get("document_type", "").strip() not in EXPECTED_TYPES:
            errors.append(
                f"Row {row_number}: Unexpected document type."
            )

    # Normalize document text before checking for duplicates.
    normalized_texts = [
        " ".join(row["text"].lower().split())
        for row in rows
    ]

    if len(rows) != 90:
        errors.append(f"Expected 90 records, found {len(rows)}.")

    if len(document_ids) != len(set(document_ids)):
        errors.append("Duplicate document IDs found.")

    if len(source_ids) != len(set(source_ids)):
        errors.append("Duplicate source record IDs found.")

    if len(normalized_texts) != len(set(normalized_texts)):
        errors.append("Duplicate document text found.")

    if document_types != Counter(EXPECTED_TYPES):
        errors.append("Document type distribution does not match targets.")

    if labels != Counter({"General & Administration": 90}):
        errors.append("Classification label distribution is incorrect.")

    if locales != Counter({"en": 90}):
        errors.append("Locale distribution is incorrect.")

    if source_types != Counter({"external_dataset": 90}):
        errors.append("Source type distribution is incorrect.")

    # Print the validation report.
    print("\n" + "=" * 55)
    print("GENERAL & ADMINISTRATION DATASET VALIDATION")
    print("=" * 55)

    print(f"\nTotal records: {len(rows)}")
    print(f"Unique document IDs: {len(set(document_ids))}")
    print(f"Unique source record IDs: {len(set(source_ids))}")
    print(f"Unique document texts: {len(set(normalized_texts))}")

    print("\nDocument type distribution:")
    for document_type, count in sorted(document_types.items()):
        print(f"  {document_type:<28} {count}")

    print("\nClassification labels:", dict(labels))
    print("Source types:", dict(source_types))
    print("Locales:", dict(locales))

    print(f"\nErrors: {len(errors)}")
    for error in errors:
        print("  -", error)

    if errors:
        print("\nGENERAL DATASET VALIDATION FAILED")
        return False

    print("\nGENERAL DATASET VALIDATION PASSED")
    return True


if __name__ == "__main__":
    validate_general_dataset()
