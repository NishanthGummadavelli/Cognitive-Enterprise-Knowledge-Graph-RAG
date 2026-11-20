"""Collect project-management documents from the Sinabis dataset."""

import csv
import hashlib
import random
import re
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "sinabis-group/sinabis-10k-synthetic-docx"
DATASET_URL = "https://huggingface.co/datasets/sinabis-group/sinabis-10k-synthetic-docx"

OUTPUT_PATH = Path("data/training/project_management.csv")

RANDOM_SEED = 42
MIN_TEXT_LENGTH = 1000

# Number of documents required from each project-management theme.
TARGETS = {
    "project_proposal": 20,
    "project_report": 20,
    "meeting_minutes": 15,
    "change_request": 10,
    "lessons_learned": 10,
    "risk_assessment": 10,
    "feasibility_study": 5,
}

# Map original themes to broader project-management subject areas.
SUBJECT_AREAS = {
    "project_proposal": "Project Planning & Proposals",
    "project_report": "Project Execution & Monitoring",
    "meeting_minutes": "Project Meetings & Coordination",
    "change_request": "Project Scope & Change Control",
    "lessons_learned": "Project Evaluation & Learning",
    "risk_assessment": "Project Risk Management",
    "feasibility_study": "Project Feasibility & Business Case",
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
    "annotation_count",
    "original_file",
    "issuer",
    "issuer_country",
    "recipient",
    "document_date",
    "blueprint_id",
    "original_document_number",
]

LABEL = "Projects & Management"

LICENSE = (
    "CC BY 4.0 declared by dataset card; "
    "verify dataset terms before redistribution"
)


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def calculate_text_hash(text: str) -> str:
    """Generate a stable SHA-256 hash for duplicate detection."""

    normalized = normalize_text(text).lower()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_existing_data() -> tuple[set[str], set[str]]:
    """Load existing IDs and text hashes to prevent duplicate additions."""

    existing_ids = set()
    existing_hashes = set()

    if not OUTPUT_PATH.exists():
        return existing_ids, existing_hashes

    csv.field_size_limit(100_000_000)

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                "Existing project dataset has an unexpected CSV schema."
            )

        for row in reader:
            existing_ids.add(row["document_id"])

            if row["text"].strip():
                existing_hashes.add(
                    calculate_text_hash(row["text"])
                )

    return existing_ids, existing_hashes


def collect_candidates() -> dict[str, list[dict]]:
    """Stream the dataset and collect eligible English project documents."""

    print("Connecting to the Hugging Face dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    candidates = {
        theme: [] for theme in TARGETS
    }

    scanned_records = 0

    for record in dataset:
        scanned_records += 1

        theme = record.get("theme", "")
        language = record.get("lang", "")
        text = record.get("text", "")

        # Keep only the seven selected themes.
        if theme not in TARGETS:
            continue

        # The classification dataset is English-only.
        if language != "en":
            continue

        if not isinstance(text, str):
            continue

        text = normalize_text(text)

        if len(text) < MIN_TEXT_LENGTH:
            continue

        # Original dataset index and filename help identify records.
        original_index = record.get("idx")
        original_file = record.get("file", "")

        if original_index is None or not original_file:
            continue

        candidates[theme].append(
            {
                "idx": original_index,
                "file": original_file,
                "text": text,
                "title": record.get("title", ""),
                "issuer": record.get("issuer", ""),
                "issuer_country": record.get("issuer_country", ""),
                "recipient": record.get("recipient", ""),
                "date": record.get("date", ""),
                "blueprint_id": record.get("blueprint_id", ""),
                "number": record.get("number", ""),
            }
        )

    print(f"\nTotal records scanned: {scanned_records:,}")

    for theme, records in candidates.items():
        print(f"{theme:<25} {len(records):>5} eligible records")

    return candidates


def build_dataset_rows(
    candidates: dict[str, list[dict]],
    existing_ids: set[str],
    existing_hashes: set[str],
) -> list[dict[str, str]]:
    """Select unique records according to the predefined theme targets."""

    rng = random.Random(RANDOM_SEED)

    selected_rows = []
    used_ids = set(existing_ids)
    used_hashes = set(existing_hashes)

    for theme, target in TARGETS.items():
        records = candidates[theme].copy()

        # Shuffle records reproducibly to reduce ordering bias.
        rng.shuffle(records)

        selected_count = 0

        for record in records:
            if selected_count >= target:
                break

            document_text = record["text"]
            document_hash = calculate_text_hash(document_text)

            if document_hash in used_hashes:
                continue

            document_id = f"sinabis_pm_{int(record['idx']):06d}"

            if document_id in used_ids:
                continue

            row = {
                "text": document_text,
                "label": LABEL,
                "source_type": "external_dataset",
                "source_name": DATASET_NAME,
                "source_url": DATASET_URL,
                "license": LICENSE,
                "document_id": document_id,
                "document_type": theme,
                "locale": "en",
                "subject_area": SUBJECT_AREAS[theme],
                "source_site": "Sinabis Synthetic Business Documents",
                "topic": record["title"],
                "annotation_count": "",
                "original_file": record["file"],
                "issuer": record["issuer"],
                "issuer_country": record["issuer_country"],
                "recipient": record["recipient"],
                "document_date": str(record["date"]),
                "blueprint_id": record["blueprint_id"],
                "original_document_number": record["number"],
            }

            selected_rows.append(row)

            used_ids.add(document_id)
            used_hashes.add(document_hash)
            selected_count += 1

        print(f"\n{theme}: {selected_count}/{target} selected")

        # Never write a partially completed collection.
        if selected_count < target:
            raise ValueError(
                f"Insufficient unique documents for {theme}. "
                f"Required {target}, selected {selected_count}. "
                "No new records have been written."
            )

    return selected_rows


def save_dataset(rows: list[dict[str, str]]) -> None:
    """Write the selected documents to the project-management CSV."""

    file_exists = OUTPUT_PATH.exists()

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("PROJECT DOCUMENT COLLECTION COMPLETED")
    print("=" * 60)
    print(f"New documents added: {len(rows)}")
    print(f"Output file: {OUTPUT_PATH}")


def main() -> None:
    """Run the complete project-document collection workflow."""

    print("Starting project-management document collection...")

    existing_ids, existing_hashes = load_existing_data()

    print(f"Existing document IDs: {len(existing_ids)}")

    candidates = collect_candidates()

    rows = build_dataset_rows(
        candidates,
        existing_ids,
        existing_hashes,
    )

    save_dataset(rows)


if __name__ == "__main__":
    main()
