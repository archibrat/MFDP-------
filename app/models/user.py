from datetime import datetime
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship
from pydantic import EmailStr
from email_validator import validate_email, EmailNotValidError

from app.models.event import Event
from app.models.mltask import MLTask
import re

class UserBase(SQLModel):
    """
    Базовая модель пользователя с общими полями.
    """
    username: str = Field(
        ...,
        unique=True,
        index=True,
        min_length=3,
        max_length=50,
        description="Имя пользователя"
    )
    email: str = Field(
        ...,
        unique=True,
        index=True,
        min_length=5,
        max_length=255,
        description="Электронная почта пользователя"
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Полное имя пользователя"
    )
    role: str = Field(
        default="user",
        description="Роль пользователя (admin, doctor, user)"
    )
    is_active: bool = Field(
        default=True,
        description="Активен ли пользователь"
    )

class User(UserBase, table=True):
    """
    Модель пользователя для хранения в базе данных.
    """
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str = Field(
        ..., 
        min_length=4,
        description="Хешированный пароль пользователя"
    )
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
    """
    password: str = Field(
        ..., 
        min_length=4,
        description="Пароль пользователя (незашифрованный)"
    )

class UserUpdate(SQLModel):
    """
    DTO модель для обновления пользователя.
    """
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(None, min_length=5, max_length=255)
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    role: Optional[str] = Field(None)
    is_active: Optional[bool] = Field(None)
    password: Optional[str] = Field(None, min_length=4)

class UserResponse(UserBase):
    """
    DTO модель для ответа с данными пользователя.
    """
    id: int
    created_at: datetime

