from itertools import pairwise

from document_qa.chunker import chunk_pages


def test_no_chunk_exceeds_size_limit():
    pages = [{"page_number": 1, "text": "Python is easy to learn and use"}]

    chunks = chunk_pages(
        pages,
        document_name="guide.pdf",
        chunk_size=10,
    )

    assert all(len(chunk["text"]) <= 10 for chunk in chunks)


def test_chunks_never_cross_pages():
    pages = [
        {"page_number": 1, "text": "First page has some text"},
        {"page_number": 2, "text": "Second page has more text"},
    ]

    chunks = chunk_pages(
        pages,
        document_name="guide.pdf",
        chunk_size=10,
    )

    assert all(
        chunk["page_number"] == 1
        for chunk in chunks
        if chunk["text"].startswith("First")
    )
    assert all(
        chunk["page_number"] == 2
        for chunk in chunks
        if chunk["text"].startswith("Second")
    )


def test_chunks_contain_required_metadata():
    pages = [{"page_number": 1, "text": "Hello world"}]

    chunks = chunk_pages(
        pages,
        document_name="guide.pdf",
        chunk_size=20,
    )

    assert chunks == [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Hello world",
        }
    ]


def test_chunks_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_pages(
        [{"page_number": 1, "text": text}],
        document_name="guide.pdf",
        chunk_size=10,
        overlap=3,
    )

    assert len(chunks) > 1

    for previous, current in pairwise(chunks):
        assert previous["text"][-3:] == current["text"][:3]
