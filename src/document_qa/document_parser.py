import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

MIN_EXTRACTABLE_CHARACTERS = 50


class UnsupportedFileTypeError(ValueError):
    pass


class PDFParsingError(ValueError):
    pass


class DocumentDecodingError(ValueError):
    pass


class DocumentHasNoTextError(ValueError):
    pass


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    # Rejoin words split by a hyphen at a line break.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Remove repeated spaces and tabs, preserving line breaks for now.
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove spaces around line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Squeeze repeated blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_document(file_path: str | Path) -> list[dict[str, int | str]]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: '{path.name}'")

    if suffix in {".txt", ".md"}:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise DocumentDecodingError(
                f"'{path.name}' is not valid UTF-8 text."
            ) from e

        text = clean_text(text)
        if not text:
            raise DocumentHasNoTextError(f"'{path.name}' is empty.")

        return [{"page_number": 1, "text": text}]

    if suffix != ".pdf":
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix or '(no extension)'}"
        )

    try:
        reader = PdfReader(path)
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            pages.append({"page_number": page_number, "text": text})

    except Exception as e:
        # raises error while preserving the original exception for debugging
        raise PDFParsingError(f"Could not read PDF '{path.name}': {e}") from e

    # check doc as a whole, not each page; a very short legitimate PDF could trip this
    total_text = "".join(page["text"] for page in pages)

    if len(total_text.strip()) < MIN_EXTRACTABLE_CHARACTERS:
        raise DocumentHasNoTextError(
            f"PDF '{path.name}' contains almost no extractable text. "
            "It may be scanned or empty."
        )

    return pages
