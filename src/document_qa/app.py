from fastapi import FastAPI

from document_qa.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    app = FastAPI(title="Document QA API")

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    return app
