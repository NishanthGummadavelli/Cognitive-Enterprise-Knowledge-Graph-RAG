
"""Incrementally collect Company Documents v2 training records."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

from datasets import Pdf, load_dataset

from src.classification.document_type_mapping import DOCUMENT_TYPE_MAPPING


DATASET_NAME = "AyoubChLin/Company-document-dataset-v2"
DATASET_URL = (
    "https://huggingface.co/datasets/"
    "AyoubChLin/Company-document-dataset-v2"
)

OUTPUT_PATH = Path("data/training/company_documents_v2.csv")

# We already have 30 external records per category.
# The target is 90 external + 10 original synthetic = 100 total.
TARGET_PER_CATEGORY = 90

MAX_RECORDS_TO_SCAN = 20000

LICENSE_NOTE = (
    "Apache-2.0 declared by dataset card; "
    "verify underlying source terms"
)

FIELDNAMES = [
    "text",
    "label",
    "source_type",
    "source_name",
    "source_url",
    "license",
    "document_id",
    "document_type",
    "locale",
]


def collect_company_documents() -> None:
    """Append missing documents until each category reaches its target."""

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Existing dataset not found: {OUTPUT_PATH}. "
            "Run the initial collection first."
        )

    category_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_text_hashes: set[str] = set()

    # Load existing records to prevent overwriting or collecting duplicates.
    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames or not set(FIELDNAMES).issubset(
            reader.fieldnames
        ):
            raise ValueError("Existing CSV does not have the required columns.")

        for row in reader:
            category_counts[row["label"]] += 1
            seen_ids.add(row["document_id"])

            text_hash = hashlib.sha256(
                row["text"].strip().encode("utf-8")
            ).hexdigest()

            seen_text_hashes.add(text_hash)

    categories = set(DOCUMENT_TYPE_MAPPING.values())

    print("Existing external records:")
    for category in sorted(categories):
        print(
            f"  {category}: "
            f"{category_counts[category]}/{TARGET_PER_CATEGORY}"
        )

    if all(
        category_counts[category] >= TARGET_PER_CATEGORY
        for category in categories
    ):
        print("\nAll category targets have already been reached.")
        return

    # Stream records without decoding original PDF objects.
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    dataset = dataset.cast_column("pdf", Pdf(decode=False))
    dataset = dataset.remove_columns(["pdf"])
    dataset = dataset.shuffle(seed=42, buffer_size=500)

    scanned_records = 0
    added_records = 0

    print("\nStarting incremental collection...\n")

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)

        for record in dataset:
            scanned_records += 1

            if scanned_records % 500 == 0:
                print(
                    f"Scanned: {scanned_records} | "
                    f"New records: {added_records}",
                    flush=True,
                )

            locale = str(record.get("locale", "")).lower()

            if locale.split("-")[0].split("_")[0] != "en":
                continue

            document_type = str(record.get("document_type", ""))
            category = DOCUMENT_TYPE_MAPPING.get(document_type)

            if category is None:
                continue

            if category_counts[category] >= TARGET_PER_CATEGORY:
                continue

            text = str(record.get("file_content") or "").strip()

            if len(text) < 50:
                continue

            external_id = str(
                record.get("doc_id") or record.get("file_name", "")
            )

            if not external_id:
                continue

            document_id = f"companydocs_{external_id}"

            if document_id in seen_ids:
                continue

            text_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            if text_hash in seen_text_hashes:
                continue

            writer.writerow(
                {
                    "text": text,
                    "label": category,
                    "source_type": "synthetic",
                    "source_name": DATASET_NAME,
                    "source_url": DATASET_URL,
                    "license": LICENSE_NOTE,
                    "document_id": document_id,
                    "document_type": document_type,
                    "locale": locale,
                }
            )

            # Update tracking immediately after writing each record.
            seen_ids.add(document_id)
            seen_text_hashes.add(text_hash)
            category_counts[category] += 1
            added_records += 1

            print(
                f"Added {category}: "
                f"{category_counts[category]}/{TARGET_PER_CATEGORY}",
                flush=True,
            )

            if all(
                category_counts[item] >= TARGET_PER_CATEGORY
                for item in categories
            ):
                break

            if scanned_records >= MAX_RECORDS_TO_SCAN:
                print("\nReached the maximum scan limit.")
                break

    print("\n========== INCREMENTAL COLLECTION REPORT ==========")
    print(f"Records scanned: {scanned_records}")
    print(f"New records added: {added_records}")
    print(f"Total external records: {sum(category_counts.values())}")

    print("\nCategory distribution:")
    for category in sorted(categories):
        print(
            f"  {category}: "
            f"{category_counts[category]}/{TARGET_PER_CATEGORY}"
        )

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    collect_company_documents()
