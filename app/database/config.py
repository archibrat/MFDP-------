"""
Конфигурация базы данных для PostgreSQL
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Настройки подключения к базе данных PostgreSQL"""
    
    # PostgreSQL настройки
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "medical_db")
    
    # Тестовая база данных
    TEST_POSTGRES_DB: str = os.getenv("TEST_POSTGRES_DB", "medical_db")
    
    # Настройки приложения
    SECRET_KEY: str = os.getenv("SECRET_KEY", "ef7cb39b5097a96e496ee87383cf9d853706151497015a826120db8e25084ef3")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    COOKIE_NAME: str = os.getenv("COOKIE_NAME", "access_token")
    
    # Настройки ML
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "./models")
    
    @property
    def database_url(self) -> str:
        """Формирует URL для подключения к базе данных"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @property
    def test_database_url(self) -> str:
        """Формирует URL для подключения к тестовой базе данных"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.TEST_POSTGRES_DB}"
        )
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


# Глобальный экземпляр настроек
_settings: Optional[DatabaseSettings] = None


def get_settings() -> DatabaseSettings:
    """Получение настроек базы данных"""
    global _settings
    if _settings is None:
        _settings = DatabaseSettings()
    return _settings


# Экспорт настроек для обратной совместимости
settings = get_settings()
