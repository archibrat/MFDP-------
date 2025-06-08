from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # Настройки базы данных
    DB_HOST: Optional[str] = None    # Хост базы данных
    DB_PORT: Optional[int] = None    # Порт базы данных
    DB_USER: Optional[str] = None    # Имя пользователя БД
    DB_PASS: Optional[str] = None    # Пароль пользователя БД
    DB_NAME: Optional[str] = None    # Название базы данных
    COOKIE_NAME: Optional[str] = None # Название cookie
    SECRET_KEY: Optional[str] = None  # Секретный ключ
    
    # Настройки приложения
    APP_NAME: Optional[str] = None        # Название приложения
    APP_DESCRIPTION: Optional[str] = None # Описание приложения
    DEBUG: Optional[bool] = None          # Режим отладки
    API_VERSION: Optional[str] = None     # Версия API
    
    @property
    def DATABASE_URL_asyncpg(self):
        """Формирует URL подключения для asyncpg"""
        return f'postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'
    
    @property
    def DATABASE_URL_psycopg(self):
        """Формирует URL подключения для psycopg"""
        return f'postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    def validate(self) -> None:
        """Проверка критических параметров конфигурации"""
        if not all([self.DB_HOST, self.DB_USER, self.DB_PASS, self.DB_NAME]):
            raise ValueError("Отсутствуют необходимые параметры конфигурации базы данных")

@lru_cache()
def get_settings() -> Settings:
    """Получение настроек приложения с кэшированием"""
    settings = Settings()
    settings.validate()
    return settings
