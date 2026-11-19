
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document

from src.ingestion.cleaner import clean_text


@dataclass
class DocumentElement:
    """Represent a structured piece of an extracted document.

    Attributes:
        text: Extracted and cleaned text.
        element_type: Type of element, such as heading or paragraph.
        source: Original document filename.
        page_number: PDF page number, if available.
        heading_level: Heading hierarchy level, if available.
    """

    text: str
    element_type: str
    source: str
    page_number: int | None = None
    heading_level: int | None = None


def load_pdf(file_path: str) -> list[DocumentElement]:
    """Extract text blocks from a PDF while preserving page numbers.

    Args:
        file_path: Path to the PDF document.

    Returns:
        A list of structured document elements.
    """

    path = Path(file_path)
    elements = []

    # Open the PDF safely using a context manager.
    with pymupdf.open(file_path) as pdf:

        # Process each page separately to retain its page number.
        for page_number, page in enumerate(pdf, start=1):

            # Extract text blocks instead of flattening the entire page.
            blocks = page.get_text("blocks")

            for block in blocks:

                # Ignore non-text blocks, such as images.
                if len(block) > 6 and block[6] != 0:
                    continue

                block_text = clean_text(block[4])

                # Skip empty blocks.
                if not block_text:
                    continue

                elements.append(
                    DocumentElement(
                        text=block_text,
                        element_type="paragraph",
                        source=path.name,
                        page_number=page_number,
                    )
                )

    return elements


def load_docx(file_path: str) -> list[DocumentElement]:
    """Extract DOCX paragraphs while preserving heading styles.

    Args:
        file_path: Path to the DOCX document.

    Returns:
        A list of structured document elements.
    """

    path = Path(file_path)
    document = Document(file_path)
    elements = []

    # Process paragraphs in their original document order.
    for paragraph in document.paragraphs:

        paragraph_text = clean_text(paragraph.text)

        # Ignore empty paragraphs.
        if not paragraph_text:
            continue

        style_name = paragraph.style.name

        # Identify headings using Word's built-in heading styles.
        if style_name.startswith("Heading "):
            try:
                heading_level = int(style_name.split()[-1])
            except ValueError:
                heading_level = None

            element_type = "heading"

        elif style_name == "Title":
            heading_level = 1
            element_type = "heading"

        else:
            heading_level = None
            element_type = "paragraph"

        elements.append(
            DocumentElement(
                text=paragraph_text,
                element_type=element_type,
                source=path.name,
                heading_level=heading_level,
            )
        )

    return elements



def load_txt(file_path: str) -> list[DocumentElement]:
    """Extract TXT content while identifying Markdown headings and paragraphs.

    Args:
        file_path: Path to the TXT document.

    Returns:
        A list of structured document elements.
    """

    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")

    # Normalize line endings before processing individual lines.
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    elements = []
    paragraph_lines = []

    # Process each line to distinguish headings from paragraph content.
    for line in raw_text.split("\n"):

        cleaned_line = clean_text(line)

        # Skip blank lines after storing any accumulated paragraph.
        if not cleaned_line:
            if paragraph_lines:
                paragraph_text = clean_text("\n".join(paragraph_lines))

                elements.append(
                    DocumentElement(
                        text=paragraph_text,
                        element_type="paragraph",
                        source=path.name,
                    )
                )

                paragraph_lines = []

            continue

        # Recognize Markdown headings from individual lines.
        heading_match = re.match(
            r"^(#{1,6})\s+(.+)$",
            cleaned_line,
        )

        if heading_match:

            # Store any paragraph content appearing before the heading.
            if paragraph_lines:
                paragraph_text = clean_text("\n".join(paragraph_lines))

                elements.append(
                    DocumentElement(
                        text=paragraph_text,
                        element_type="paragraph",
                        source=path.name,
                    )
                )

                paragraph_lines = []

            # Extract the heading text and its hierarchy level.
            heading_text = heading_match.group(2).strip()
            heading_level = len(heading_match.group(1))

            elements.append(
                DocumentElement(
                    text=heading_text,
                    element_type="heading",
                    source=path.name,
                    heading_level=heading_level,
                )
            )

        else:
            # Accumulate consecutive ordinary lines into one paragraph.
            paragraph_lines.append(cleaned_line)

    # Store the final paragraph if the document doesn't end with a blank line.
    if paragraph_lines:
        paragraph_text = clean_text("\n".join(paragraph_lines))

        elements.append(
            DocumentElement(
                text=paragraph_text,
                element_type="paragraph",
                source=path.name,
            )
        )

    return elements


def load_document(file_path: str) -> list[DocumentElement]:
    """Load a supported document into structured elements.

    Args:
        file_path: Path to a PDF, DOCX, or TXT document.

    Returns:
        Structured elements extracted from the document.

    Raises:
        ValueError: If the file extension is unsupported.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    # Select the appropriate loader based on the file extension.
    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
    }

    loader = loaders.get(extension)

    if loader is None:
        raise ValueError(f"Unsupported file type: {extension}")

    return loader(str(path))
