"""Collect English administrative documents for classification training."""

import csv
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "LiquidAI/structured-extraction"
DATASET_URL = "https://huggingface.co/datasets/LiquidAI/structured-extraction"
OUTPUT_PATH = Path("data/training/general_administration.csv")

LABEL = "General & Administration"
MIN_TEXT_LENGTH = 200

TARGETS = {
    "contact_card": 20,
    "meeting_invite": 20,
    "event_invitation": 15,
    "lost_and_found_report": 15,
    "school_notice": 10,
    "car_rental_booking": 5,
    "flight_booking": 5,
}

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
    "subject_area",
    "source_site",
    "topic",
    "source_record_id",
    "structured_extraction",
]


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[^\S\n]+", " ", line).strip()
        for line in text.split("\n")
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def calculate_text_hash(text: str) -> str:
    """Return a SHA-256 hash for duplicate-text detection."""

    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_existing_data() -> tuple[set[str], set[str]]:
    """Load existing IDs and text hashes to prevent duplicate collection."""

    existing_ids: set[str] = set()
    existing_hashes: set[str] = set()

    if not OUTPUT_PATH.exists():
        return existing_ids, existing_hashes

    with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                "Existing CSV has an unexpected schema. "
                "Inspect it before collecting more records."
            )

        for row in reader:
            existing_ids.add(row["document_id"])
            existing_hashes.add(calculate_text_hash(row["text"]))

    return existing_ids, existing_hashes


def collect_candidates() -> dict[str, list[dict[str, str]]]:
    """Stream the dataset and collect eligible English document candidates."""

    print("Connecting to the Hugging Face dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    candidates = {document_type: [] for document_type in TARGETS}
    total_scanned = 0

    for record in dataset:
        total_scanned += 1

        document_type = record.get("doc_type")
        if document_type not in TARGETS:
            continue

        if record.get("language") != "en":
            continue

        text = record.get("document")
        if not isinstance(text, str):
            continue

        text = normalize_text(text)

        if len(text) < MIN_TEXT_LENGTH:
            continue

        record_id = str(record.get("id", "")).strip()
        if not record_id:
            continue

        extraction = record.get("extraction", {})
        if not isinstance(extraction, dict):
            extraction = {}

        candidates[document_type].append(
            {
                "record_id": record_id,
                "text": text,
                "extraction": json.dumps(
                    extraction,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    print(f"\nTotal records scanned: {total_scanned}")

    for document_type, records in candidates.items():
        print(f"{document_type:<28} {len(records):>4} eligible records")

    return candidates


def build_dataset_rows(
    candidates: dict[str, list[dict[str, str]]],
    existing_ids: set[str],
    existing_hashes: set[str],
) -> list[dict[str, str]]:
    """Select the target distribution and prepare CSV rows."""

    rng = random.Random(42)
    selected_rows: list[dict[str, str]] = []

    used_ids = set(existing_ids)
    used_hashes = set(existing_hashes)

    for document_type, target in TARGETS.items():
        records = candidates[document_type].copy()
        rng.shuffle(records)

        selected = 0

        for record in records:
            record_id = record["record_id"]
            text = record["text"]
            text_hash = calculate_text_hash(text)

            if record_id in used_ids or text_hash in used_hashes:
                continue

            extraction = json.loads(record["extraction"])

            # Use a meaningful extracted title or subject when available.
            topic = (
                extraction.get("title")
                or extraction.get("event_name")
                or extraction.get("subject")
                or extraction.get("school_name")
                or extraction.get("item")
                or document_type.replace("_", " ").title()
            )

            row = {
                "text": text,
                "label": LABEL,
                "source_type": "external_dataset",
                "source_name": DATASET_NAME,
                "source_url": DATASET_URL,
                "license": (
                    "Refer to the dataset card for its declared licence; "
                    "verify underlying source terms before redistribution"
                ),
                "document_id": f"liquidai_general_{record_id}",
                "document_type": document_type,
                "locale": "en",
                "subject_area": "General Administration",
                "source_site": "LiquidAI Structured Extraction Dataset",
                "topic": str(topic),
                "source_record_id": record_id,
                "structured_extraction": record["extraction"],
            }

            selected_rows.append(row)
            used_ids.add(record_id)
            used_hashes.add(text_hash)
            selected += 1

            if selected == target:
                break

        print(f"\n{document_type:<28} {selected}/{target} selected")

        if selected < target:
            raise ValueError(
                f"Could only select {selected}/{target} records for "
                f"{document_type}. No CSV changes have been made."
            )

    return selected_rows


def save_dataset(rows: list[dict[str, str]]) -> None:
    """Append validated collection rows to the output CSV."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_PATH.exists()

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if write_header:
            writer.writeheader()

        writer.writerows(rows)

    print(f"\nSaved {len(rows)} new records to: {OUTPUT_PATH}")


def main() -> None:
    """Run the General & Administration document collection workflow."""

    print("Starting General & Administration document collection...")

    existing_ids, existing_hashes = load_existing_data()
    print(f"Existing document IDs: {len(existing_ids)}")

    candidates = collect_candidates()

    rows = build_dataset_rows(
        candidates,
        existing_ids,
        existing_hashes,
    )

    save_dataset(rows)

    print("\n" + "=" * 55)
    print("GENERAL & ADMINISTRATION COLLECTION COMPLETED")
    print("=" * 55)
    print(f"New documents added: {len(rows)}")


if __name__ == "__main__":
    main()
