
"""Collect a balanced set of IT documentation records from Hugging Face."""

import csv
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


# Dataset and output configuration.
DATASET_NAME = "saidsef/tech-docs"
DATASET_URL = "https://huggingface.co/datasets/saidsef/tech-docs"
OUTPUT_PATH = Path("data/training/it_documents.csv")

TARGET_DOCUMENTS = 90
MIN_TEXT_LENGTH = 500
RANDOM_SEED = 42

# The dataset card declares Apache-2.0; individual upstream sources
# may have separate terms that need to be checked.
LICENSE_NOTE = (
    "Apache-2.0 declared by dataset card; "
    "verify individual upstream source terms"
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
    "subject_area",
    "source_site",
    "topic",
]


def normalize_text(text: str) -> str:
    """Normalize text for reliable duplicate detection."""

    return re.sub(r"\s+", " ", text).strip().lower()


def identify_subject_area(site: str, topic: str) -> str:
    """Assign a broad IT subject area using source and topic clues."""

    combined = f"{site} {topic}".lower()

    # Check more specific areas before broader infrastructure terms.
    if any(word in combined for word in ("vault", "secret", "security")):
        return "Security & Secrets"

    if any(word in combined for word in ("prometheus", "grafana", "monitor")):
        return "Monitoring & Observability"

    if any(word in combined for word in ("istio", "cilium", "ingress", "network")):
        return "Networking"

    if any(word in combined for word in ("redis", "database", "sql")):
        return "Databases & Caching"

    if any(word in combined for word in ("scikit", "machine learning", "ml framework")):
        return "AI & Machine Learning"

    if any(word in combined for word in ("linux", "operating system")):
        return "Operating Systems"

    if any(word in combined for word in ("docker", "kubernetes", "helm")):
        return "Containers & Orchestration"

    if any(word in combined for word in ("terraform", "gcp", "eks", "cloud")):
        return "Cloud & Infrastructure"

    if any(word in combined for word in ("argocd", "tekton", "ci/cd", "pipeline")):
        return "DevOps & CI/CD"

    return "General IT Documentation"


def collect_it_documents() -> None:
    """Select and save a balanced, reproducible IT document dataset."""

    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Total source records: {len(dataset)}")

    # Group eligible records by their original documentation source.
    records_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_hashes: set[str] = set()

    for record in dataset:
        site = str(record.get("site") or "").strip()
        topic = str(record.get("questions") or "").strip()
        text = str(record.get("answers_cleaned") or "").strip()

        # Skip incomplete or very short records.
        if not site or not topic or len(text) < MIN_TEXT_LENGTH:
            continue

        # Remove exact duplicate document text.
        text_hash = hashlib.sha256(
            normalize_text(text).encode("utf-8")
        ).hexdigest()

        if text_hash in seen_hashes:
            continue

        seen_hashes.add(text_hash)

        records_by_source[site].append(
            {
                "site": site,
                "topic": topic,
                "text": text,
                "text_hash": text_hash,
            }
        )

    if not records_by_source:
        raise ValueError("No eligible IT documentation records were found.")

    print(f"Eligible documentation sources: {len(records_by_source)}")
    print(
        "Eligible unique records:",
        sum(len(records) for records in records_by_source.values()),
    )

    # Shuffle within each source using a fixed seed for reproducibility.
    rng = random.Random(RANDOM_SEED)
    for records in records_by_source.values():
        rng.shuffle(records)

    # Select one record per source in each round to improve source balance.
    selected: list[dict[str, str]] = []
    source_names = sorted(records_by_source)
    source_positions = {source: 0 for source in source_names}

    while len(selected) < TARGET_DOCUMENTS:
        added_in_round = False

        for source in source_names:
            position = source_positions[source]
            source_records = records_by_source[source]

            if position < len(source_records):
                selected.append(source_records[position])
                source_positions[source] += 1
                added_in_round = True

                if len(selected) == TARGET_DOCUMENTS:
                    break

        # Stop if all source records have been exhausted.
        if not added_in_round:
            break

    if len(selected) < TARGET_DOCUMENTS:
        raise ValueError(
            f"Only {len(selected)} eligible records were found; "
            f"{TARGET_DOCUMENTS} were requested."
        )

    # Write the selected records using a stable project dataset schema.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for record in selected:
            document_id = f"it_{record['text_hash'][:16]}"

            writer.writerow(
                {
                    "text": record["text"],
                    "label": "IT",
                    "source_type": "external_dataset",
                    "source_name": DATASET_NAME,
                    "source_url": DATASET_URL,
                    "license": LICENSE_NOTE,
                    "document_id": document_id,
                    "document_type": "technical_documentation",
                    "locale": "en",
                    "subject_area": identify_subject_area(
                        record["site"], record["topic"]
                    ),
                    "source_site": record["site"],
                    "topic": record["topic"],
                }
            )

    # Display a summary so the collection can be checked immediately.
    source_counts: dict[str, int] = defaultdict(int)
    subject_counts: dict[str, int] = defaultdict(int)

    for record in selected:
        source_counts[record["site"]] += 1
        subject = identify_subject_area(record["site"], record["topic"])
        subject_counts[subject] += 1

    print("\nIT document collection completed.")
    print(f"Documents collected: {len(selected)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nDocuments by source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    print("\nDocuments by subject area:")
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count}")


if __name__ == "__main__":
    collect_it_documents()
