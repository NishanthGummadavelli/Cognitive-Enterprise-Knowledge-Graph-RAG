
"""Define the official enterprise document categories used by the project."""

# Keep the category names consistent across training, prediction and APIs.
CATEGORIES: tuple[str, ...] = (
    "HR",
    "Finance",
    "IT",
    "Legal & Compliance",
    "Operations",
    "Sales & Marketing",
    "Projects & Management",
    "General & Administration",
)

# A set provides efficient category membership checks during validation.
CATEGORY_SET = frozenset(CATEGORIES)
