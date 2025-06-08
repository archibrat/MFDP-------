from sqlmodel import SQLModel, Session, create_engine 
from sqlalchemy.engine import Engine
from typing import Optional
from database.config import get_settings

def get_database_engine():
    """
    Создание и настройка движка SQLAlchemy.
    
    Возвращает:
        Engine: Настроенный движок SQLAlchemy
    """
    settings = get_settings()
    
    # Создаем движок с настройками подключения к базе данных
    engine = create_engine(
        url=settings.DATABASE_URL_psycopg,
        echo=settings.DEBUG,
        pool_size=5,  # Размер пула соединений
        max_overflow=10,  # Максимальное количество дополнительных соединений
        pool_pre_ping=True,  # Проверка соединения перед использованием
        pool_recycle=3600  # Время жизни соединения в секундах
    )
    return engine

# Инициализация движка базы данных
engine = get_database_engine()

def get_session():
    # Создание сессии базы данных
    with Session(engine) as session:
        yield session

# class DatabaseManager:
#     _engine: Optional[Engine] = None
    
#     @classmethod
#     def get_engine(cls) -> Engine:
#         if cls._engine is None:
#             settings = get_settings()
#             cls._engine = create_engine(
#                 url=settings.DATABASE_URL_psycopg,
#                 echo=settings.DEBUG,
#                 pool_size=5,
#                 max_overflow=10,
#                 pool_pre_ping=True,
#                 pool_recycle=3600
#             )
#         return cls._engine
    
#     @classmethod
#     def get_session(cls):
#         with Session(cls.get_engine()) as session:
#             yield session
            
#     @classmethod
#     def close(cls):
#         if cls._engine:
#             cls._engine.dispose()
#             cls._engine = None