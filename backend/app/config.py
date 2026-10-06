import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# On Vercel (and most serverless hosts) only /tmp is writable.
ON_VERCEL = bool(os.environ.get("VERCEL"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Athlete Lens"
    app_env: str = "production" if ON_VERCEL else "development"
    version: str = "2.0.0"
    database_url: str = "sqlite:////tmp/athlete_lens_v2.db"
    cors_origins: str = "http://localhost:5173,http://localhost:8000,http://127.0.0.1:5173"
    model_dir: str = "/tmp/athlete_lens_models"
    seed_demo_data: bool = True
    max_upload_bytes: int = 2_000_000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def platform(self) -> str:
        return "vercel" if ON_VERCEL else "server"


settings = Settings()
