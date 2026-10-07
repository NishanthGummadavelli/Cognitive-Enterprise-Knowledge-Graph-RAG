from pathlib import Path

import  pymupdf
from docx import Document


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    pdf = pymupdf.open(file_path)

    text = ""

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


def load_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

    return text


def load_txt(file_path: str) -> str:
    """Read text from a TXT file."""
    return Path(file_path).read_text(encoding="utf-8")


def load_document(file_path: str) -> str:
    """Load a supported document based on its file extension."""
    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    elif extension == ".txt":
        return load_txt(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )
