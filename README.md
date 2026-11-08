# Document QA

A small RAG (retrieval-augmented generation) project. It finds the parts of your PDFs that match a question, then asks an LLM to answer using only those parts. If the documents don't contain the answer, it says `Answer not found.`

It is tested on six NIST publications about AI and security, listed in `docs/corpus.md`.

## How it works

1. **Read:** extract the text of each PDF, page by page.
2. **Split:** cut each page into small chunks of about 1,200 characters.
3. **Embed:** turn each chunk into a vector with a local model (`bge-small-en-v1.5`).
4. **Search:** when a question comes in, find the 5 chunks most similar to it.
5. **Answer:** send those chunks to an LLM through OpenRouter, with the rule "answer only from these sources and cite them like [1]".

If no chunk is similar enough, the app answers `Answer not found.` without calling the LLM.

## Run it

You need Python 3.12, uv and OpenRouter API Key.

```bash
git clone https://github.com/ahdgalal/document-qa.git
cd document-qa
uv sync
cp .env.example .env
```

1. Open `.env` and replace `Your_OpenRouter_API_Key` with your OpenRouter key.
2. Put your PDFs in a folder called `data/`.
3. Build the index, then start the API:

```bash
uv run python -m document_qa.indexer
uv run uvicorn document_qa.app:create_app --factory
```

Open <http://127.0.0.1:8000/docs> and try `POST /ask` with:

```json
{"question": "What four functions make up the AI RMF Core?"}
```

The answer comes back with citations, each listing a document and a page number.

## Tests

```bash
uv run pytest
```

The tests use fake models, so they don't need internet access or an API key.

## Limitations

- Only works with PDFs that contain text (scanned PDFs are not supported).
- English only.
- Tables in PDFs are read as plain text, so answers from tables may be missed.

## License

MIT