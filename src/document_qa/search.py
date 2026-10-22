import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


class Index:
    def __init__(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        model: TextEmbedding,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model

    def search(self, question: str, top_k: int = 5, ) -> list[dict]:
        
        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not self.chunks:
            return []

        question_embedding = np.asarray(next(self.model.embed([question])), dtype=np.float32)

        scores = self.embeddings @ question_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            result = self.chunks[int(index)].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results


def load_index(data_dir: Path, model: TextEmbedding) -> Index:
    """Load the persisted index using an injected embedding model"""
    chunks_path = data_dir / "chunks.jsonl"
    embeddings_path = data_dir / "embeddings.npy"

    with chunks_path.open(encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    embeddings = np.load(embeddings_path)

    return Index(
        chunks=chunks,
        embeddings=embeddings,
        model=model,
    )
