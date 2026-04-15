from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str

    # gpt-4o for writing; gpt-4o-mini for cheap deterministic tasks
    openai_model_writer: str = "gpt-4o"
    openai_model_fast: str = "gpt-4o-mini"

    # Optional: set APP_API_KEY in .env to require X-API-Key header on /tailor
    app_api_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
