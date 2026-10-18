from pathlib import Path

import pytest
from fpdf import FPDF

from document_qa.document_parser import (
    PDFHasNoTextError,
    PDFParsingError,
    UnsupportedFileTypeError,
    clean_text,
    parse_document,
)


def test_txt_is_one_page(tmp_path: Path):
    file_path = tmp_path / "policy.txt"
    file_path.write_text("Annual leave is 21 days.", encoding="utf-8")

    assert parse_document(file_path) == [
        {
            "text": "Annual leave is 21 days.",
            "page_number": 1,
        }
    ]


def test_markdown_is_one_page(tmp_path: Path):
    file_path = tmp_path / "policy.md"
    file_path.write_text("# Leave policy\n21 days.", encoding="utf-8")

    pages = parse_document(file_path)

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert pages[0]["text"] == "# Leave policy\n21 days."


def test_unsupported_extension(tmp_path: Path):
    file_path = tmp_path / "policy.docx"
    file_path.write_text("Not supported", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        parse_document(file_path)


def create_test_pdf(path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text="First page text: annual leave is 21 days.")

    pdf.add_page()
    pdf.cell(0, 10, text="Second page text: sick leave is 10 days.")

    pdf.output(str(path))


def test_clean_text():
    text = "The ﬁrst inter-\nnational example"
    cleaned = clean_text(text)

    assert "The first international example" == cleaned


def test_garbage_pdf_raises_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_text("This is not a real PDF")

    with pytest.raises(PDFParsingError):
        parse_document(path)


def test_pdf_keeps_page_numbers(tmp_path: Path):
    file_path = tmp_path / "sample.pdf"
    create_test_pdf(file_path)

    pages = parse_document(file_path)

    assert len(pages) == 2
    assert [page["page_number"] for page in pages] == [1, 2]
    assert "First page text" in pages[0]["text"]
    assert "Second page text" in pages[1]["text"]


def test_blank_pdf_raises_no_text_error(tmp_path: Path):
    path = tmp_path / "blank.pdf"

    pdf = FPDF()
    pdf.add_page()  # creates a valid PDF page w/o text
    pdf.output(str(path))

    with pytest.raises(PDFHasNoTextError, match="almost no extractable text"):
        parse_document(path)
