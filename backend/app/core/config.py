# currently unused: legacy app package configuration is not imported by main.py.
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "GenResearch"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Single instance used across the entire app
settings = Settings()
