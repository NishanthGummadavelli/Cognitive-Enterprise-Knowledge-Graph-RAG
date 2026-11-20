"""Validate the combined external Legal & Compliance dataset."""

import csv
import hashlib
from collections import Counter
from pathlib import Path


DATASET_PATH = Path("data/training/legal_contracts.csv")

EXPECTED_COLUMNS = [
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
    "annotation_count",
]

EXPECTED_SOURCE_COUNTS = {
    "CUAD": 35,
    "DOJ Guidance": 15,
    "Federal Register": 25,
    "Terms of Service": 15,
}

VALID_LABEL = "Legal & Compliance"


def normalize_text(text: str) -> str:
    """Normalize text for consistent duplicate detection."""

    return " ".join(text.lower().split())


def calculate_text_hash(text: str) -> str:
    """Generate a SHA-256 hash of normalized document text."""

    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def identify_source(document_id: str) -> str:
    """Identify the dataset source using its document ID prefix."""

    if document_id.startswith("pile_doj_"):
        return "DOJ Guidance"

    if document_id.startswith("pile_fr_"):
        return "Federal Register"

    if document_id.startswith("pile_tos_"):
        return "Terms of Service"

    return "CUAD"


def validate_legal_dataset() -> bool:
    """Validate document content, metadata, uniqueness and source counts."""

    errors = []
    warnings = []

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return False

    # Large legal documents require a higher CSV field-size limit.
    csv.field_size_limit(100_000_000)

    document_ids = set()
    text_hashes = set()
    source_counts = Counter()
    document_types = Counter()

    total_records = 0

    with DATASET_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        # Verify that the CSV structure is correct.
        if reader.fieldnames != EXPECTED_COLUMNS:
            print("ERROR: CSV column structure is incorrect.")
            print(f"Expected: {EXPECTED_COLUMNS}")
            print(f"Found: {reader.fieldnames}")
            return False

        for row_number, row in enumerate(reader, start=2):
            total_records += 1

            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            document_id = row.get("document_id", "").strip()
            document_type = row.get("document_type", "").strip()
            source_url = row.get("source_url", "").strip()
            source_type = row.get("source_type", "").strip()
            source_name = row.get("source_name", "").strip()
            license_name = row.get("license", "").strip()
            locale = row.get("locale", "").strip()

            # Check mandatory document fields.
            if not text:
                errors.append(f"Row {row_number}: Empty document text.")

            if label != VALID_LABEL:
                errors.append(
                    f"Row {row_number}: Invalid label '{label}'."
                )

            if not document_id:
                errors.append(f"Row {row_number}: Missing document ID.")

            if not document_type:
                errors.append(f"Row {row_number}: Missing document type.")

            if not source_url:
                errors.append(f"Row {row_number}: Missing source URL.")

            if not source_name:
                errors.append(f"Row {row_number}: Missing source name.")

            if not license_name:
                errors.append(f"Row {row_number}: Missing license metadata.")

            if source_type != "external_dataset":
                errors.append(
                    f"Row {row_number}: Unexpected source type '{source_type}'."
                )

            if locale != "en":
                errors.append(
                    f"Row {row_number}: Unexpected locale '{locale}'."
                )

            # Check document ID uniqueness.
            if document_id:
                if document_id in document_ids:
                    errors.append(
                        f"Row {row_number}: Duplicate ID '{document_id}'."
                    )
                document_ids.add(document_id)

            # Check exact duplicate document text.
            if text:
                current_hash = calculate_text_hash(text)

                if current_hash in text_hashes:
                    errors.append(
                        f"Row {row_number}: Duplicate document text."
                    )

                text_hashes.add(current_hash)

            # Count records by their original dataset.
            source = identify_source(document_id)
            source_counts[source] += 1
            document_types[document_type] += 1

    # Validate expected source distributions.
    for source, expected_count in EXPECTED_SOURCE_COUNTS.items():
        actual_count = source_counts[source]

        if actual_count != expected_count:
            errors.append(
                f"{source}: Expected {expected_count}, found {actual_count}."
            )

    expected_total = sum(EXPECTED_SOURCE_COUNTS.values())

    if total_records != expected_total:
        errors.append(
            f"Expected {expected_total} total records, found {total_records}."
        )

    # Display validation results.
    print("\n" + "=" * 65)
    print("LEGAL & COMPLIANCE DATASET VALIDATION")
    print("=" * 65)

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Total external documents: {total_records}")
    print(f"Unique document IDs: {len(document_ids)}")
    print(f"Unique document texts: {len(text_hashes)}")

    print("\nSource distribution:")

    for source, count in sorted(source_counts.items()):
        print(f"  {source:<25} {count:>5}")

    print("\nDocument type distribution:")

    for document_type, count in sorted(document_types.items()):
        print(f"  {document_type:<30} {count:>5}")

    print(f"\nErrors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\nVALIDATION ERRORS:")

        for error in errors[:30]:
            print(f"  - {error}")

        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more errors.")

        print("\nLEGAL DATASET VALIDATION FAILED.")
        return False

    print("\nLEGAL DATASET VALIDATION PASSED.")
    return True


if __name__ == "__main__":
    validate_legal_dataset()
