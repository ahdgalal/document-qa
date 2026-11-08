from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# reads configuration from environment variables and, when configured, a .env file
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUMENT_QA_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    max_upload_mb: int = 10

    search_threshold: float = 0.6
    llm_model: str = "google/gemini-3.1-flash-lite"

    openrouter_api_key: str = Field(
        default="",
        validation_alias="OPENROUTER_API_KEY",
    )
