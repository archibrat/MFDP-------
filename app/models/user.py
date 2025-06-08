from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
import re

if TYPE_CHECKING:
    from models.event import Event
    from models.mltask import MLTask

class UserBase(SQLModel):
    """
    Базовая модель пользователя с общими полями.
    
    Атрибуты:
        email (str): Электронная почта пользователя
        password (str): Хешированный пароль
    """
    email: str = Field(
        ...,  # Обязательное поле
        unique=True,
        index=True,
        min_length=5,
        max_length=255,
        description="Электронная почта пользователя"
    )
    password: str = Field(
        ..., 
        min_length=4,
        description="Хешированный пароль пользователя"
    )

class User(UserBase, table=True):
    """
    Модель пользователя для хранения в базе данных.
    
    Атрибуты:
        id (int): Первичный ключ
        created_at (datetime): Дата создания аккаунта
        events (List[Event]): Список событий пользователя
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    events: List["Event"] = Relationship(
        back_populates="creator",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "lazy": "selectin"
        }
    )
    ml_tasks: List["MLTask"] = Relationship(
        back_populates="creator",
        sa_relationship_kwargs={"lazy": "selectin"}
    )

    def __str__(self) -> str:
        """Строковое представление пользователя"""
        return f"Id: {self.id}. Email: {self.email}"

    def validate_email(self) -> bool:
        """
        Проверка формата электронной почты.
        
        Возвращает:
            bool: True если формат верный
        
        Вызывает:
            ValueError: Если формат электронной почты неверный
        """
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not pattern.match(self.email):
            raise ValueError("Неверный формат электронной почты")
        return True
    
    @property
    def event_count(self) -> int:
        """Количество событий, связанных с пользователем"""
        return len(self.events)

class UserCreate(UserBase):
    """
    DTO модель для создания нового пользователя.
    
    Наследует все базовые поля от UserBase (email и password),
    не добавляя новых полей.
    """
    pass

