from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel

class TimestampMixin(SQLModel):
    """
    Миксин для добавления временных меток к моделям.
    """
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Время создания записи"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Время последнего обновления записи"
    )

class BaseDBModel(TimestampMixin):
    """
    Базовая модель для всех сущностей в системе.
    """
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
        description="Уникальный идентификатор записи"
    )
    
    class Config:
        from_attributes = True
        validate_assignment = True
        use_enum_values = True

# Базовые Pydantic схемы
class BaseResponse(BaseModel):
    """Базовая схема ответа API"""
    success: bool = True
    message: str = "Операция выполнена успешно"
    data: Optional[Any] = Field(default=None)

class PaginationParams(BaseModel):
    """Параметры пагинации"""
    page: int = Field(default=1, ge=1, description="Номер страницы")
    size: int = Field(default=10, ge=1, le=100, description="Размер страницы")

class PaginatedResponse(BaseModel):
    """Схема пагинированного ответа"""
    success: bool = True
    message: str = "Операция выполнена успешно"
    data: list
    total: int
    page: int
    size: int
    pages: int

    @classmethod
    def create(cls, data: list, total: int, page: int, size: int):
        """Создание пагинированного ответа"""
        pages = (total + size - 1) // size
        return cls(
            data=data,
            total=total,
            page=page,
            size=size,
            pages=pages
        )

# Общие валидаторы
class CommonValidators:
    """Общие валидаторы для моделей"""
    
    @staticmethod
    def validate_non_empty_string(v: str, field_name: str = "field") -> str:
        if not v or not v.strip():
            raise ValueError(f"{field_name} не может быть пустым")
        return v.strip()
    
    @staticmethod
    def validate_metadata_size(v: Optional[Dict[str, Any]], max_size: int = 5000) -> Optional[Dict[str, Any]]:
        if v and len(str(v)) > max_size:
            raise ValueError(f"Метаданные слишком большие (максимум {max_size} символов)")
        return v