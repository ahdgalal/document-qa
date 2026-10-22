import json

import numpy as np

from document_qa.indexer import find_pdfs, collect_chunks, save_index


def test_find_pdfs_returns_only_pdf_files(tmp_path):
    (tmp_path / "a.pdf").touch()
    (tmp_path / "b.pdf").touch()
    (tmp_path / "notes.txt").touch()

    result = find_pdfs(tmp_path)

    assert [path.name for path in result] == ["a.pdf", "b.pdf"]


def test_collect_chunks_combines_documents(tmp_path, monkeypatch):
    for name in ["a.pdf", "b.pdf"]:
        (tmp_path / name).touch()

    def fake_index_document(pdf_path):
        return [{"document_name": pdf_path.name, "text": "example"}]

    monkeypatch.setattr(
        "document_qa.indexer.index_document",
        fake_index_document,
    )

    result = collect_chunks(tmp_path)

    assert len(result) == 2
    assert result[0]["document_name"] == "a.pdf"
    assert result[1]["document_name"] == "b.pdf"


def test_save_index_writes_jsonl_and_numpy(tmp_path):
    chunks = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Hello world",
        }
    ]
    embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    save_index(chunks, embeddings, tmp_path)

    with (tmp_path / "chunks.jsonl").open(encoding="utf-8") as file:
        saved_chunks = [json.loads(line) for line in file]

    saved_embeddings = np.load(tmp_path / "embeddings.npy")

    assert saved_chunks == chunks
    np.testing.assert_array_equal(saved_embeddings, embeddings)



def test_collect_chunks_skips_failed_document(tmp_path, monkeypatch, capsys):
    for name in ["bad.pdf", "good.pdf"]:
        (tmp_path / name).touch()

    def fake_index_document(pdf_path):
        if pdf_path.name == "bad.pdf":
            raise ValueError("No text found")
        return [{"document_name": pdf_path.name, "text": "example"}]

    monkeypatch.setattr(
        "document_qa.indexer.index_document",
        fake_index_document,
    )

    result = collect_chunks(tmp_path)

    assert len(result) == 1
    assert result[0]["document_name"] == "good.pdf"
    assert "bad.pdf: skipped: No text found" in capsys.readouterr().out

