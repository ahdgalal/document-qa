from fastapi.testclient import TestClient

from document_qa.app import create_app
from document_qa.settings import Settings


class FakeIndex:
    def __init__(self, results):
        self.results = results

    def search(self, question, top_k):
        return self.results


def create_test_app(monkeypatch, results):
    monkeypatch.setattr("document_qa.app.TextEmbedding", lambda model: None)
    monkeypatch.setattr(
        "document_qa.app.load_index", lambda data_dir, model: FakeIndex(results)
    )

    return create_app()


def test_health_returns_ok():
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_max_upload_mb_from_environment(monkeypatch):
    monkeypatch.setenv("DOCUMENT_QA_MAX_UPLOAD_MB", "25")

    settings = Settings()

    assert settings.max_upload_mb == 25


def test_invalid_citation_is_dropped(monkeypatch):
    app = create_test_app(
        monkeypatch,
        [
            {
                "text": "RAG stands for Retrieval Augmented Generation.",
                "document_name": "RAG.pdf",
                "page_number": 1,
                "score": 0.9,
            }
        ],
    )

    monkeypatch.setattr(
        "document_qa.app.generate_answer",
        lambda messages, settings: {
            "answer": "RAG stands for Retrieval Augmented Generation [1]. Invalid source [9].",
            "input_tokens": 10,
            "output_tokens": 8,
        },
    )

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert response.json()["citations"] == [{"id": 1, "document": "RAG.pdf", "page": 1}]


def test_low_score_does_not_call_llm(monkeypatch):
    app = create_test_app(
        monkeypatch,
        [
            {
                "text": "Unrelated information.",
                "document_name": "other.pdf",
                "page_number": 1,
                "score": 0.1,
            }
        ],
    )

    def fail_if_called(messages, settings):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr("document_qa.app.generate_answer", fail_if_called)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert response.json()["answer"] == ("Answer not found.")
    assert response.json()["citations"] == []


def test_sources_are_inside_source_tags(monkeypatch):
    app = create_test_app(
        monkeypatch,
        [
            {
                "text": "RAG is Retrieval Augmented Generation.",
                "document_name": "RAG.pdf",
                "page_number": 3,
                "score": 0.9,
            }
        ],
    )

    captured = {}

    def fake_generate_answer(messages, settings):
        captured["messages"] = messages
        return {
            "answer": "RAG stands for Retrieval Augmented Generation [1].",
            "input_tokens": 10,
            "output_tokens": 5,
        }

    monkeypatch.setattr("document_qa.app.generate_answer", fake_generate_answer)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is RAG?"})

    user_message = captured["messages"][1]["content"]

    assert response.status_code == 200
    assert '<source id="1" document="RAG.pdf" page="3">' in user_message
    assert "RAG is Retrieval Augmented Generation." in user_message
    assert "</source>" in user_message
