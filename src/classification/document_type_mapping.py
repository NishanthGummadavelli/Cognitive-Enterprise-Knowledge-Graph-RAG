
"""Map external document types to project classification categories."""

from src.classification.categories import CATEGORIES


# Each document type is assigned according to its primary business purpose.
DOCUMENT_TYPE_MAPPING: dict[str, str] = {
    "account_statement": "Finance",
    "invoice": "Finance",
    "receipt": "Finance",
    "credit_note": "Finance",
    "payslip": "HR",
    "employment_certificate": "HR",
    "shipping_order": "Operations",
    "work_order": "Operations",
    "purchase_order": "Operations",
    "packing_slip": "Operations",
    "quote": "Sales & Marketing",
}


def validate_document_type_mapping() -> None:
    """Ensure every mapped document type uses an official category."""

    invalid_categories = {
        category
        for category in DOCUMENT_TYPE_MAPPING.values()
        if category not in CATEGORIES
    }

    if invalid_categories:
        raise ValueError(
            f"Invalid classification categories: {invalid_categories}"
        )


if __name__ == "__main__":
    validate_document_type_mapping()

    print("Document type mapping validated successfully.")
    print(f"Total mapped document types: {len(DOCUMENT_TYPE_MAPPING)}")

    for document_type, category in DOCUMENT_TYPE_MAPPING.items():
        print(f"{document_type:<25} -> {category}")
