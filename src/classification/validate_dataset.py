
"""Validate the quality and consistency of the classification dataset."""

import csv
import re
from collections import Counter
from pathlib import Path

from src.classification.categories import CATEGORIES, CATEGORY_SET


REQUIRED_COLUMNS = {
    "text",
    "label",
    "source_type",
    "source_name",
    "source_url",
    "license",
    "document_id",
}


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection.

    Args:
        text: Original document text.

    Returns:
        Lowercase text with normalized whitespace.
    """

    # Ignore case and repeated whitespace when comparing document content.
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_dataset(dataset_path: Path) -> bool:
    """Check dataset structure, labels, duplicates and category distribution.

    Args:
        dataset_path: Path to the classification CSV file.

    Returns:
        True if no validation errors are found, otherwise False.
    """

    if not dataset_path.is_file():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        return False

    errors: list[str] = []
    warnings: list[str] = []

    with dataset_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        # Check that the CSV contains all required columns.
        if reader.fieldnames is None:
            print("ERROR: CSV file is empty or has no header.")
            return False

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            print(f"ERROR: Missing columns: {sorted(missing_columns)}")
            return False

        rows = list(reader)

    if not rows:
        print("ERROR: Dataset contains no examples.")
        return False

    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen_text: dict[str, int] = {}
    seen_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        text = (row.get("text") or "").strip()
        label = (row.get("label") or "").strip()
        source_type = (row.get("source_type") or "").strip().lower()
        source_name = (row.get("source_name") or "").strip()
        source_url = (row.get("source_url") or "").strip()
        license_name = (row.get("license") or "").strip()
        document_id = (row.get("document_id") or "").strip()

        # Check essential fields.
        if not text:
            errors.append(f"Row {row_number}: Missing document text.")

        if not label:
            errors.append(f"Row {row_number}: Missing category label.")
        elif label not in CATEGORY_SET:
            errors.append(
                f"Row {row_number}: Invalid category '{label}'."
            )
        else:
            category_counts[label] += 1

        if not document_id:
            errors.append(f"Row {row_number}: Missing document ID.")
        elif document_id in seen_ids:
            errors.append(
                f"Row {row_number}: Duplicate document ID '{document_id}'."
            )
        else:
            seen_ids.add(document_id)

        if not source_type:
            errors.append(f"Row {row_number}: Missing source type.")
        else:
            source_counts[source_type] += 1

            # Real examples must have traceable source information.
            if source_type == "real":
                if not source_name or not source_url or not license_name:
                    errors.append(
                        f"Row {row_number}: Real example is missing "
                        "source name, source URL or licence."
                    )
            elif source_type != "synthetic":
                errors.append(
                    f"Row {row_number}: Unknown source type '{source_type}'."
                )

        # Check that document content is not duplicated.
        if text:
            normalized = normalize_text(text)

            if normalized in seen_text:
                first_row = seen_text[normalized]
                warnings.append(
                    f"Row {row_number}: Duplicate text found "
                    f"(first appeared in row {first_row})."
                )
            else:
                seen_text[normalized] = row_number

        if not source_name:
            warnings.append(f"Row {row_number}: Source name is empty.")

    # Identify official categories that have no examples.
    for category in CATEGORIES:
        if category_counts[category] == 0:
            errors.append(f"Category '{category}' has no examples.")

    # Display the validation report.
    print("\n========== DATASET VALIDATION REPORT ==========")
    print(f"Dataset: {dataset_path}")
    print(f"Total examples: {len(rows)}")

    print("\nCategory distribution:")
    for category in CATEGORIES:
        print(f"  {category}: {category_counts[category]}")

    print("\nSource distribution:")
    for source_type, count in sorted(source_counts.items()):
        print(f"  {source_type}: {count}")

    print(f"\nErrors: {len(errors)}")
    for error in errors:
        print(f"  ERROR: {error}")

    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings:
        print(f"  WARNING: {warning}")

    if errors:
        print("\nDataset validation FAILED.")
        return False

    print("\nDataset validation PASSED.")
    return True


if __name__ == "__main__":
    dataset_path = Path("data/training/classification_dataset.csv")

    # Return a failure exit code when validation finds errors.
    if not validate_dataset(dataset_path):
        raise SystemExit(1)
