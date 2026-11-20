"""Collect additional Legal & Compliance documents from Pile of Law."""

import csv
import hashlib
import json
import lzma
import random
import re
from pathlib import Path
import sys


# Dataset locations.
INPUT_FOLDER = Path("data/training/legal/pile_of_law")
OUTPUT_PATH = Path("data/training/legal_contracts.csv")

# Reproducible selection.
RANDOM_SEED = 42

# Keep documents large enough to contain meaningful legal information.
MIN_TEXT_LENGTH = 1000

# Avoid extremely large records that may contain entire collections.
MAX_TEXT_LENGTH = 200_000

# Target number of new documents from each source.
TARGETS = {
    "train.doj_guidance.jsonl.xz": 15,
    "train.federal_register.jsonl.xz": 25,
    "train.tos.jsonl.xz": 15,
}

# Consistent CSV structure shared with the existing CUAD dataset.
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
]

LABEL = "Legal & Compliance"
SOURCE_NAME = "Pile of Law"
LICENSE = (
    "CC BY-NC-SA 4.0 declared by Pile of Law; "
    "verify underlying source terms and restrictions"
)


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving the document's full content."""

    # Remove unnecessary whitespace without deleting paragraph boundaries.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def text_hash(text: str) -> str:
    """Generate a stable SHA-256 hash for duplicate detection."""

    normalized = normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def looks_english(text: str, url: str) -> bool:
    """Apply a conservative heuristic to exclude obvious non-English text."""

    # Exclude URLs explicitly identifying Spanish documents.
    lowered_url = url.lower()

    if "spanish" in lowered_url or "espanol" in lowered_url:
        return False

    # Compare common English and Spanish words in the document sample.
    sample = re.sub(r"[^a-zA-Z\s]", " ", text[:8000]).lower()
    words = set(sample.split())

    english_markers = {
        "the", "and", "of", "to", "is", "are", "for",
        "this", "that", "with", "shall", "under", "any",
        "such", "may", "must", "from", "by", "section",
    }

    spanish_markers = {
        "el", "la", "los", "las", "de", "del", "que",
        "para", "por", "una", "uno", "con", "como",
        "derecho", "votacion", "estado", "ley",
    }

    english_count = len(words.intersection(english_markers))
    spanish_count = len(words.intersection(spanish_markers))

    # Require sufficient English evidence and reject likely Spanish text.
    return english_count >= 5 and english_count > spanish_count * 1.5



def load_existing_data() -> tuple[set[str], set[str]]:
    """Read existing IDs and text hashes to prevent duplicate additions."""

    # Allow large legal documents to be read from the CSV.
    csv.field_size_limit(100_000_000)

    existing_ids = set()
    existing_hashes = set()

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Existing legal dataset not found: {OUTPUT_PATH}"
        )

    with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                "Existing CSV columns do not match the expected schema.\n"
                f"Expected: {FIELDNAMES}\n"
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            existing_ids.add(row["document_id"])

            if row["text"].strip():
                existing_hashes.add(text_hash(row["text"]))

    return existing_ids, existing_hashes



def collect_candidates(file_path: Path) -> list[dict[str, str]]:
    """Read and filter eligible documents from one Pile of Law collection."""

    candidates = []

    with lzma.open(file_path, "rt", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue

            record = json.loads(line)

            # Extract the original document text and source URL.
            text = record.get("text", "")
            url = record.get("url", "")

            if not isinstance(text, str) or not isinstance(url, str):
                continue

            text = normalize_text(text)

            if not MIN_TEXT_LENGTH <= len(text) <= MAX_TEXT_LENGTH:
                continue

            if not looks_english(text, url):
                continue

            candidates.append(
                {
                    "text": text,
                    "source_url": url,
                    "original_index": str(index),
                }
            )

    return candidates


def build_dataset_rows(
    existing_ids: set[str],
    existing_hashes: set[str],
) -> list[dict[str, str]]:
    """Select the requested number of unique documents from each source."""

    rng = random.Random(RANDOM_SEED)
    selected_rows = []

    source_metadata = {
        "train.doj_guidance.jsonl.xz": {
            "document_type": "government_guidance",
            "subject_area": "Government Legal Guidance",
            "source_site": "U.S. Department of Justice",
        },
        "train.federal_register.jsonl.xz": {
            "document_type": "regulatory_notice",
            "subject_area": "Regulations and Government Notices",
            "source_site": "U.S. Federal Register",
        },
        "train.tos.jsonl.xz": {
            "document_type": "terms_of_service",
            "subject_area": "Terms and Conditions",
            "source_site": "Terms of Service Corpus",
        },
    }

    # Track duplicates across all selected sources.
    used_hashes = set(existing_hashes)
    used_ids = set(existing_ids)

    for filename, target in TARGETS.items():
        matches = list(INPUT_FOLDER.rglob(filename))

        if not matches:
            raise FileNotFoundError(
                f"Could not find downloaded collection: {filename}"
            )

        file_path = matches[0]
        metadata = source_metadata[filename]

        print(f"\nInspecting: {filename}")

        candidates = collect_candidates(file_path)

        # Shuffle candidates consistently to avoid always selecting
        # only the earliest documents in a source.
        rng.shuffle(candidates)

        source_selected = 0

        for candidate in candidates:
            if source_selected >= target:
                break

            document_text = candidate["text"]
            document_hash = text_hash(document_text)

            if document_hash in used_hashes:
                continue

            # Use source-specific IDs to keep records identifiable.
            prefix = {
                "train.doj_guidance.jsonl.xz": "pile_doj",
                "train.federal_register.jsonl.xz": "pile_fr",
                "train.tos.jsonl.xz": "pile_tos",
            }[filename]

            document_id = (
                f"{prefix}_{int(candidate['original_index']):06d}"
            )

            if document_id in used_ids:
                continue

            row = {
                "text": document_text,
                "label": LABEL,
                "source_type": "external_dataset",
                "source_name": SOURCE_NAME,
                "source_url": candidate["source_url"],
                "license": LICENSE,
                "document_id": document_id,
                "document_type": metadata["document_type"],
                "locale": "en",
                "subject_area": metadata["subject_area"],
                "source_site": metadata["source_site"],
                "topic": "",
                "annotation_count": "",
            }

            selected_rows.append(row)
            used_hashes.add(document_hash)
            used_ids.add(document_id)
            source_selected += 1

        print(f"Eligible candidates: {len(candidates)}")
        print(f"Selected: {source_selected}/{target}")

        # Do not append an incomplete collection.
        if source_selected < target:
            raise ValueError(
                f"Insufficient eligible unique records in {filename}. "
                f"Needed {target}, found {source_selected}. "
                "No new records have been written."
            )

    return selected_rows


def save_dataset(rows: list[dict[str, str]]) -> None:
    """Append the selected documents to the existing legal CSV."""

    # Append only after every source has met its target.
    with OUTPUT_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writerows(rows)

    print("\nCollection completed successfully.")
    print(f"New documents added: {len(rows)}")
    print(f"Updated dataset: {OUTPUT_PATH}")


def main() -> None:
    """Run the Pile of Law collection workflow."""

    print("Starting Pile of Law legal document collection...")

    existing_ids, existing_hashes = load_existing_data()

    print(f"Existing document IDs: {len(existing_ids)}")

    rows = build_dataset_rows(existing_ids, existing_hashes)

    save_dataset(rows)


if __name__ == "__main__":
    main()
