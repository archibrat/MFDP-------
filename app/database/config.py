from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASS: str = "postgres"
    DB_NAME: str = "medical_db"
    SECRET_KEY: str = "ef7cb39b5097a96e496ee87383cf9d853706151497015a826120db8e25084ef3"
    COOKIE_NAME: str = "session"
    APP_NAME: str = "medical_ml"
    APP_DESCRIPTION: str = "Medical ML API"
    API_VERSION: str = "1.0.0"
    redis_password: str = ""
    ml_models_path: str = "/app/models"
    DEBUG: bool = False

    @property
    def DATABASE_URL_psycopg(self):
        return f'postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow"  
    )
