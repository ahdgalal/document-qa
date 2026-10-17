from fastapi.testclient import TestClient

from document_qa.app import create_app
from document_qa.settings import Settings


def test_health_returns_ok():
    app = create_app(Settings())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_max_upload_mb_from_environment(monkeypatch):
    monkeypatch.setenv("DOCUMENT_QA_MAX_UPLOAD_MB", "25")
    settings = Settings()
    assert settings.max_upload_mb == 25
