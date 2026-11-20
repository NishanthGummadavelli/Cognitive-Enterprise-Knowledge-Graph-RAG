
"""Collect a balanced set of complete contracts from the CUAD dataset."""

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


# Input and output configuration.
INPUT_PATH = Path("data/training/legal/CUAD_v1.json")
OUTPUT_PATH = Path("data/training/legal_contracts.csv")

TARGET_DOCUMENTS = 35
MIN_TEXT_LENGTH = 1000
RANDOM_SEED = 42

DATASET_NAME = "Contract Understanding Atticus Dataset (CUAD)"
DATASET_URL = "https://www.atticusprojectai.org/cuad/"

# Record the declared dataset licence without assuming every underlying
# contract has identical reuse permissions.
LICENSE_NOTE = (
    "CC BY 4.0 declared by CUAD dataset; "
    "verify underlying contract reuse terms"
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
    "annotation_count",
]


def normalize_text(text: str) -> str:
    """Normalize contract text for duplicate detection."""

    return re.sub(r"\s+", " ", text).strip().lower()


def identify_contract_type(title: str) -> str:
    """Infer a broad agreement type from its contract title."""

    normalized_title = title.lower()

    # Check specific agreement types before more general terms.
    patterns = [
        (r"non.?disclosure|confidentiality", "Non-Disclosure Agreement"),
        (r"employment|employee", "Employment Agreement"),
        (r"distribution|distributor", "Distribution Agreement"),
        (r"supply|supplier", "Supply Agreement"),
        (r"hosting|website hosting", "Hosting Agreement"),
        (r"consulting|consultant", "Consulting Agreement"),
        (r"joint venture", "Joint Venture Agreement"),
        (r"franchise", "Franchise Agreement"),
        (r"intellectual property|\bip agreement\b", "Intellectual Property Agreement"),
        (r"license|licensing", "Licensing Agreement"),
        (r"service agreement", "Service Agreement"),
        (r"reseller", "Reseller Agreement"),
        (r"marketing|promotion", "Marketing Agreement"),
        (r"endorsement", "Endorsement Agreement"),
        (r"agency", "Agency Agreement"),
        (r"collaboration|cooperation|strategic alliance", "Collaboration Agreement"),
        (r"remarketing", "Remarketing Agreement"),
        (r"sponsorship", "Sponsorship Agreement"),
        (r"affiliate", "Affiliate Agreement"),
        (r"co.?branding", "Co-Branding Agreement"),
        (r"content license", "Content Licensing Agreement"),
    ]

    for pattern, agreement_type in patterns:
        if re.search(pattern, normalized_title):
            return agreement_type

    return "Other Commercial Agreement"


def collect_contracts() -> None:
    """Select, validate and save a balanced CUAD contract collection."""

    # Verify that the source JSON exists.
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"CUAD JSON file not found: {INPUT_PATH}"
        )

    print(f"Reading CUAD dataset from: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    source_records = dataset.get("data", [])

    if not source_records:
        raise ValueError("The CUAD JSON contains no contract records.")

    print(f"Total source contracts: {len(source_records)}")

    # Group complete contracts by inferred agreement type.
    contracts_by_type: dict[str, list[dict]] = defaultdict(list)
    seen_hashes: set[str] = set()

    for record in source_records:
        title = str(record.get("title") or "").strip()

        # CUAD stores the full contract text in paragraph context.
        paragraphs = record.get("paragraphs", [])
        contract_texts = [
            str(paragraph.get("context") or "").strip()
            for paragraph in paragraphs
            if paragraph.get("context")
        ]

        # Join contexts if a contract contains multiple paragraphs.
        full_text = "\n\n".join(contract_texts)

        if not title or len(full_text) < MIN_TEXT_LENGTH:
            continue

        # Remove exact duplicate contract text.
        text_hash = hashlib.sha256(
            normalize_text(full_text).encode("utf-8")
        ).hexdigest()

        if text_hash in seen_hashes:
            continue

        seen_hashes.add(text_hash)

        agreement_type = identify_contract_type(title)

        # Count annotations for traceability, not as separate documents.
        annotation_count = sum(
            len(paragraph.get("qas", []))
            for paragraph in paragraphs
        )

        contracts_by_type[agreement_type].append(
            {
                "title": title,
                "text": full_text,
                "text_hash": text_hash,
                "agreement_type": agreement_type,
                "annotation_count": annotation_count,
            }
        )

    eligible_count = sum(
        len(records) for records in contracts_by_type.values()
    )

    print(f"Eligible unique contracts: {eligible_count}")
    print(f"Agreement types identified: {len(contracts_by_type)}")

    if eligible_count < TARGET_DOCUMENTS:
        raise ValueError(
            f"Only {eligible_count} eligible contracts found; "
            f"{TARGET_DOCUMENTS} required."
        )

    # Shuffle each agreement group reproducibly.
    rng = random.Random(RANDOM_SEED)

    agreement_types = sorted(contracts_by_type)

    for agreement_type in agreement_types:
        rng.shuffle(contracts_by_type[agreement_type])

    # Round-robin selection prevents one agreement type from dominating.
    selected: list[dict] = []
    positions = {agreement_type: 0 for agreement_type in agreement_types}

    while len(selected) < TARGET_DOCUMENTS:
        added_in_round = False

        for agreement_type in agreement_types:
            position = positions[agreement_type]
            records = contracts_by_type[agreement_type]

            if position < len(records):
                selected.append(records[position])
                positions[agreement_type] += 1
                added_in_round = True

                if len(selected) == TARGET_DOCUMENTS:
                    break

        if not added_in_round:
            break

    if len(selected) != TARGET_DOCUMENTS:
        raise ValueError(
            f"Expected {TARGET_DOCUMENTS} contracts, "
            f"but selected {len(selected)}."
        )

    # Save the selected contracts using the project's dataset schema.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for contract in selected:
            writer.writerow(
                {
                    "text": contract["text"],
                    "label": "Legal & Compliance",
                    "source_type": "external_dataset",
                    "source_name": DATASET_NAME,
                    "source_url": DATASET_URL,
                    "license": LICENSE_NOTE,
                    "document_id": (
                        f"legal_{contract['text_hash'][:16]}"
                    ),
                    "document_type": contract["agreement_type"],
                    "locale": "en",
                    "subject_area": "Commercial Contracts",
                    "source_site": "SEC-filed contract via CUAD",
                    "topic": contract["title"],
                    "annotation_count": contract["annotation_count"],
                }
            )

    # Print the final collection summary.
    type_counts: dict[str, int] = defaultdict(int)

    for contract in selected:
        type_counts[contract["agreement_type"]] += 1

    print("\nCUAD COLLECTION COMPLETED")
    print("=" * 45)
    print(f"Contracts collected: {len(selected)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nAgreement type distribution:")
    for agreement_type, count in sorted(type_counts.items()):
        print(f"  {agreement_type}: {count}")

    print("\nTotal contract text characters:")
    print(f"  {sum(len(c['text']) for c in selected):,}")


if __name__ == "__main__":
    collect_contracts()
