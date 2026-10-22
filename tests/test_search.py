import json

import numpy as np

from document_qa.search import load_index
import pytest

from document_qa.search import Index



class FakeModel:
    def embed(self, texts):
        assert texts == ["Tell me about cats"]
        yield np.array([1.0, 0.0], dtype=np.float32)


def test_search_returns_top_chunks(tmp_path):
    chunks = [
        {"text": "About cats", "page_number": 1},
        {"text": "About dogs", "page_number": 2},
        {"text": "About birds", "page_number": 3},
    ]

    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk) + "\n")

    np.save(
        tmp_path / "embeddings.npy",
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    index = load_index(data_dir=tmp_path, model=FakeModel())

    results = index.search(
        "Tell me about cats",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == "About cats"
    assert results[0]["score"] == 1.0
    assert results[1]["text"] == "About birds"
    assert results[1]["score"] == np.float32(0.8).item()

def test_search_rejects_invalid_top_k():
    index = Index(
        chunks=[],
        embeddings=np.empty((0, 384), dtype=np.float32),
        model=FakeModel(),
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search("question", top_k=0)
