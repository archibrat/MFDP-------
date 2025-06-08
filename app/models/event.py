from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING

# Условный импорт для избежания циклических зависимостей
if TYPE_CHECKING:
    from models.user import User
    from models.mltask import MLTask

class EventBase(SQLModel):
    """
    Базовая модель события с общими полями.
    
    Атрибуты:
        title (str): Название события
        image (str): URL или путь к изображению события
        description (str): Описание события
        location (Optional[str]): Место проведения события
    """
    # Основные поля модели
    title: str = Field(..., min_length=1, max_length=100)  # Название события (от 1 до 100 символов)
    image: str = Field(..., min_length=1)  # Ссылка на изображение (не пустая)
    description: str = Field(..., min_length=1, max_length=1000)  # Описание события (от 1 до 1000 символов)

class Event(EventBase, table=True):
    """
    Модель события для хранения в базе данных.
    
    Атрибуты:
        id (Optional[int]): Первичный ключ
        creator_id (Optional[int]): Внешний ключ к таблице пользователей
        creator (Optional[User]): Связь с моделью пользователя
        created_at (datetime): Временная метка создания события
    """
    # Поля для базы данных
    id: Optional[int] = Field(default=None, primary_key=True)  # ID события
    creator_id: Optional[int] = Field(default=None, foreign_key="user.id")  # ID создателя события
    creator: Optional["User"] = Relationship(  # Связь с создателем события
        back_populates="events",
        sa_relationship_kwargs={"lazy": "selectin"}
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)  # Дата и время создания
    
    def __str__(self) -> str:
        """Возвращает строковое представление события"""
        result = (f"Id: {self.id}. Title: {self.title}. Creator: {self.creator.email}")
        return result
    
    @property
    def short_description(self) -> str:
        """Возвращает сокращенное описание для предварительного просмотра"""
        max_length = 100
        return (f"{self.description[:max_length]}..."
                if len(self.description) > max_length
                else self.description)

class EventCreate(EventBase):
    """Схема для создания новых событий"""
    pass

class EventUpdate(EventBase):
    """Схема для обновления существующих событий"""
    # Необязательные поля для обновления
    title: Optional[str] = None  # Необязательное название
    image: Optional[str] = None  # Необязательное изображение
    description: Optional[str] = None  # Необязательное описание

    class Config:
        """Конфигурация модели"""
        validate_assignment = True  # Включение валидации при присваивании значений