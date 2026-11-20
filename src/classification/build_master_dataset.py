"""Combine the eight-category training datasets into one master CSV."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

from src.classification.categories import CATEGORIES


INPUT_FILES = [
    Path("data/training/classification_dataset.csv"),
    Path("data/training/company_documents_v2.csv"),
    Path("data/training/it_documents.csv"),
    Path("data/training/legal_contracts.csv"),
    Path("data/training/project_management.csv"),
    Path("data/training/general_administration.csv"),
]

OUTPUT_PATH = Path("data/training/master_classification_dataset.csv")

EXPECTED_TOTAL = 800
EXPECTED_PER_CATEGORY = 100

# Common columns appear first; additional columns are retained afterward.
BASE_COLUMNS = [
    "text",
    "label",
    "document_id",
    "source_type",
    "source_name",
    "source_url",
    "license",
    "document_type",
    "locale",
]


def calculate_text_hash(text: str) -> str:
    """Generate a normalized SHA-256 hash for duplicate detection."""

    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_source_datasets() -> tuple[list[dict[str, str]], list[str]]:
    """Load all source CSVs and construct a unified column structure."""

    csv.field_size_limit(100_000_000)

    records: list[dict[str, str]] = []
    additional_columns: list[str] = []

    for input_path in INPUT_FILES:
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset not found: {input_path}")

        print(f"Reading: {input_path}")

        with input_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError(f"Missing CSV header: {input_path}")

            required = {"text", "label", "document_id"}
            missing = required - set(reader.fieldnames)

            if missing:
                raise ValueError(
                    f"{input_path} is missing columns: {sorted(missing)}"
                )

            for column in reader.fieldnames:
                if column not in BASE_COLUMNS and column not in additional_columns:
                    additional_columns.append(column)

            source_rows = list(reader)

            # Retain the original filename for traceability.
            for row in source_rows:
                row["source_dataset_file"] = input_path.name
                records.append(row)

            print(f"  Loaded: {len(source_rows)} records")

    fieldnames = (
        BASE_COLUMNS
        + additional_columns
        + ["source_dataset_file"]
    )

    return records, fieldnames


def audit_records(records: list[dict[str, str]]) -> None:
    """Validate counts, categories, required values and duplicate records."""

    errors: list[str] = []
    category_counts = Counter(row["label"].strip() for row in records)

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for index, row in enumerate(records, start=1):
        document_id = row.get("document_id", "").strip()
        text = row.get("text", "").strip()
        label = row.get("label", "").strip()

        if not document_id:
            errors.append(f"Record {index}: Missing document ID.")
        elif document_id in seen_ids:
            errors.append(f"Record {index}: Duplicate ID: {document_id}")
        else:
            seen_ids.add(document_id)

        if not text:
            errors.append(f"Record {index}: Empty document text.")
        else:
            text_hash = calculate_text_hash(text)

            if text_hash in seen_hashes:
                errors.append(f"Record {index}: Duplicate document text.")
            else:
                seen_hashes.add(text_hash)

        if label not in CATEGORIES:
            errors.append(f"Record {index}: Invalid category: {label}")

    if len(records) != EXPECTED_TOTAL:
        errors.append(
            f"Expected {EXPECTED_TOTAL} records, found {len(records)}."
        )

    for category in CATEGORIES:
        count = category_counts.get(category, 0)

        if count != EXPECTED_PER_CATEGORY:
            errors.append(
                f"{category}: Expected {EXPECTED_PER_CATEGORY}, found {count}."
            )

    print("\n" + "=" * 55)
    print("MASTER DATASET AUDIT")
    print("=" * 55)

    print(f"\nTotal records: {len(records)}")
    print(f"Unique document IDs: {len(seen_ids)}")
    print(f"Unique document texts: {len(seen_hashes)}")

    print("\nCategory distribution:")
    for category in CATEGORIES:
        print(f"  {category:<28} {category_counts.get(category, 0)}")

    print(f"\nValidation errors: {len(errors)}")

    for error in errors:
        print("  -", error)

    if errors:
        raise ValueError(
            "Master dataset audit failed. No output CSV was created."
        )

    print("\nMASTER DATASET AUDIT PASSED")


def save_master_dataset(
    records: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write the audited records to the master CSV."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file to avoid leaving a partial output.
    temporary_path = OUTPUT_PATH.with_suffix(".tmp")

    with temporary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Replace the final file only after writing succeeds.
    temporary_path.replace(OUTPUT_PATH)

    print(f"\nMaster dataset saved: {OUTPUT_PATH}")
    print(f"Total columns: {len(fieldnames)}")


def main() -> None:
    """Build and audit the complete classification master dataset."""

    print("Starting master dataset construction...\n")

    records, fieldnames = load_source_datasets()

    print(f"\nTotal records loaded: {len(records)}")
    print(f"Unified columns: {len(fieldnames)}")

    # Audit before writing the final CSV.
    audit_records(records)

    save_master_dataset(records, fieldnames)

    print("\nMASTER DATASET CONSTRUCTION COMPLETED")


if __name__ == "__main__":
    main()
