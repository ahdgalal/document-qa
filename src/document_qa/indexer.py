from pathlib import Path
from document_qa.document_parser import parse_document
from document_qa.chunker import chunk_pages
from document_qa.settings import Settings
import json
import numpy as np
from fastembed import TextEmbedding

def find_pdfs(data_dir: Path) -> list[Path]:
    # makes the order predictable across runs
    return sorted(data_dir.glob("*.pdf"))


def index_document(pdf_path: Path) -> list[dict]:
    pages = parse_document(pdf_path)

    chunks = chunk_pages(
        pages,
        document_name=pdf_path.name,
    )

    print(f"{pdf_path.name}: {len(chunks)} chunks")

    return chunks



def collect_chunks(data_dir: Path) -> list[dict]:
    all_chunks = []

    for pdf_path in find_pdfs(data_dir):
        try:
            chunks = index_document(pdf_path)
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"{pdf_path.name}: skipped: {exc}")

    return all_chunks



def embed_chunks(chunks: list[dict]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)
    
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = list(model.embed(texts, batch_size=8))

    return np.array(embeddings, dtype=np.float32)

def save_index(
    chunks: list[dict],
    embeddings: np.ndarray,
    data_dir: Path,
) -> None:
    
    chunks_path = data_dir / "chunks.jsonl"
    embeddings_path = data_dir / "embeddings.npy"

    with chunks_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    np.save(embeddings_path, embeddings)

def main() -> None:
    settings = Settings()
    data_dir = Path(settings.data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    chunks = collect_chunks(data_dir)
    embeddings = embed_chunks(chunks)
    save_index(chunks, embeddings, data_dir)

    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
