from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Athlete Lens"
    app_env: str = "development"
    secret_key: str = "dev-only-change-me"
    database_url: str = "sqlite:////tmp/athlete_lens.db"
    cors_origins: str = "http://localhost:5173,http://localhost:8000,http://127.0.0.1:5173"
    model_dir: str = "/tmp/athlete_lens_models"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
