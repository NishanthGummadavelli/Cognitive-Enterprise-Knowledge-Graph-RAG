
import re

def clean_text(text: str) -> str:
    """Clean extracted text while preserving paragraph boundaries.

    Args:
        text: Raw text extracted from a document.

    Returns:
        Cleaned text with normalized whitespace and preserved paragraphs.
    """

    # Normalize Windows and old-style line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unnecessary spaces and tabs within individual lines.
    # Preserve newline characters to retain paragraph boundaries.
    lines = [
        re.sub(r"[^\S\n]+", " ", line).strip()
        for line in text.split("\n")
    ]

    # Reassemble the cleaned lines without losing their original separation.
    text = "\n".join(lines)

    # Reduce three or more consecutive newlines to a maximum of two.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove unnecessary whitespace at the beginning and end of the document.
    return text.strip()
