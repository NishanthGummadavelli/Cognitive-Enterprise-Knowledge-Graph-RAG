
"""Validate the collected Company Documents v2 training dataset."""

import csv
from collections import Counter
from pathlib import Path

from src.classification.document_type_mapping import DOCUMENT_TYPE_MAPPING


DATASET_PATH = Path("data/training/company_documents_v2.csv")

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
}


def validate_external_dataset() -> bool:
    """Validate external records, labels, metadata and duplicate IDs.

    Returns:
        True when no validation errors are found; otherwise False.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return False

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        # Verify the expected dataset structure.
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns

        if missing_columns:
            print(f"ERROR: Missing columns: {sorted(missing_columns)}")
            return False

        rows = list(reader)

    category_counts: Counter[str] = Counter()
    document_type_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_text_hashes: set[str] = set()

    import hashlib

    for row_number, row in enumerate(rows, start=2):
        text = row["text"].strip()
        label = row["label"].strip()
        document_id = row["document_id"].strip()
        document_type = row["document_type"].strip()

        if not text:
            errors.append(f"Row {row_number}: Empty document text")

        if label not in set(DOCUMENT_TYPE_MAPPING.values()):
            errors.append(f"Row {row_number}: Invalid category '{label}'")

        if DOCUMENT_TYPE_MAPPING.get(document_type) != label:
            errors.append(
                f"Row {row_number}: Document type/category mismatch"
            )

        if not document_id:
            errors.append(f"Row {row_number}: Missing document ID")
        elif document_id in seen_ids:
            errors.append(f"Row {row_number}: Duplicate document ID")
        else:
            seen_ids.add(document_id)

        if row["source_type"] != "synthetic":
            errors.append(f"Row {row_number}: Unexpected source type")

        if not row["source_name"].strip() or not row["source_url"].strip():
            errors.append(f"Row {row_number}: Missing source information")

        if not row["license"].strip():
            warnings.append(f"Row {row_number}: Missing license information")

        if row["locale"].lower().split("-")[0] != "en":
            errors.append(f"Row {row_number}: Non-English document")

        if text:
            text_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            if text_hash in seen_text_hashes:
                warnings.append(f"Row {row_number}: Duplicate document text")
            else:
                seen_text_hashes.add(text_hash)

        category_counts[label] += 1
        document_type_counts[document_type] += 1

    print("\n========== EXTERNAL DATASET VALIDATION ==========")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total examples: {len(rows)}")

    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print("\nDocument type distribution:")
    for document_type, count in sorted(document_type_counts.items()):
        print(f"  {document_type}: {count}")

    print(f"\nErrors: {len(errors)}")
    for error in errors:
        print(f"  ERROR: {error}")

    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings:
        print(f"  WARNING: {warning}")

    if not rows:
        print("\nDataset validation FAILED: No records found.")
        return False

    print(
        "\nDataset validation PASSED."
        if not errors
        else "\nDataset validation FAILED."
    )

    return not errors


if __name__ == "__main__":
    if not validate_external_dataset():
        raise SystemExit(1)
