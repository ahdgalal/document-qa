import pytest

from document_qa.settings import Settings


@pytest.fixture(autouse=True)
def ignore_dotenv(monkeypatch):
    monkeypatch.setattr(
        Settings,
        "model_config",
        {
            **Settings.model_config,
            "env_file": None,
        },
    )
