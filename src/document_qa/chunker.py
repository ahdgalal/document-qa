import re


def chunk_pages(
    pages: list[dict],
    document_name: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[dict]:
    
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must not be negative")

    if overlap >= chunk_size:
        overlap = min(overlap, chunk_size - 1)

    chunks = []

    for page in pages:
        text = page["text"].strip()
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            if end < len(text):
                cut = text.rfind(" ", start, end + 1)
                if cut > start:
                    end = cut

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "document_name": document_name,
                    "page_number": page["page_number"],
                    "chunk_index": len(chunks),
                    "text": chunk,
                })

            if end >= len(text):
                break

            # while retaining overlap with the previous chunk
            start = max(start + 1, end - overlap)

    return chunks
