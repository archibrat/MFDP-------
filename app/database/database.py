"""
Подключение к базе данных PostgreSQL
"""

from sqlalchemy import create_engine, event
from sqlmodel import SQLModel
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import logging
from sqlalchemy import text

from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Создание движка базы данных
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Установить True для отладки SQL запросов
)

# Создание сессии
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_session() -> Generator[Session, None, None]:
    """
    Создание сессии базы данных для dependency injection
    
    Yields:
        Session: Сессия SQLAlchemy
    """
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


async def init_database():
    """Инициализация базы данных"""
    try:
        # Импортируем все модели для регистрации в SQLModel
        from app.models import user, event, patient, prediction, mltask, medialog
        
        # Создание всех таблиц
        SQLModel.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_database():
    """Закрытие соединения с базой данных"""
    try:
        engine.dispose()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")


class DatabaseManager:
    """Менеджер для работы с базой данных"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self) -> Session:
        """Получение новой сессии"""
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """Проверка состояния подключения к базе данных"""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1")).fetchone()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False