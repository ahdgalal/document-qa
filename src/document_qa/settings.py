from pathlib import Path

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
