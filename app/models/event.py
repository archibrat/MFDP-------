from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship
from enum import Enum

# Условный импорт для избежания циклических зависимостей
if TYPE_CHECKING:
    from app.models.user import User
    from app.models.mltask import MLTask

class EventType(str, Enum):
    APPOINTMENT = "appointment"
    CONSULTATION = "consultation"
    EXAMINATION = "examination"
    OPERATION = "operation"
    FOLLOW_UP = "follow_up"

class EventPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventStatus(str, Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"

class EventBase(SQLModel):
    """Базовая модель события"""
    event_type: EventType = Field(description="Тип события")
    patient_id: int = Field(description="ID пациента")
    scheduled_time: datetime = Field(description="Запланированное время")
    description: str = Field(max_length=1000, description="Описание события")
    priority: EventPriority = Field(default=EventPriority.MEDIUM, description="Приоритет")
    status: EventStatus = Field(default=EventStatus.SCHEDULED, description="Статус события")
    sms_received: bool = Field(default=False, description="SMS получено")

class Event(EventBase, table=True):
    """Модель события для хранения в базе данных"""
    __tablename__ = "events"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    creator_id: Optional[int] = Field(default=None, foreign_key="users.id")
    creator: Optional["User"] = Relationship(
        back_populates="events",
        sa_relationship_kwargs={"lazy": "selectin"}
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def __str__(self) -> str:
        """Возвращает строковое представление события"""
        return f"Id: {self.id}. Type: {self.event_type}. Patient: {self.patient_id}"
    
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

class EventUpdate(SQLModel):
    """Схема для обновления существующих событий"""
    event_type: Optional[EventType] = None
    patient_id: Optional[int] = None
    scheduled_time: Optional[datetime] = None
    description: Optional[str] = None
    priority: Optional[EventPriority] = None
    status: Optional[EventStatus] = None
    sms_received: Optional[bool] = None

    class Config:
        """Конфигурация модели"""
        validate_assignment = True