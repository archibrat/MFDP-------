"""
Схемы данных для модуля прогнозирования неявок пациентов
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class RiskLevel(str, Enum):
    """Уровни риска неявки"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PatientProfile(BaseModel):
    """Профиль пациента для анализа неявок"""
    patient_id: int = Field(..., description="ID пациента")
    age: int = Field(..., ge=0, le=120, description="Возраст пациента")
    gender: str = Field(..., description="Пол пациента (M/F)")
    visit_history_count: int = Field(..., ge=0, description="Количество посещений")
    avg_interval_between_visits: float = Field(..., ge=0, description="Средний интервал между посещениями (дни)")
    no_show_history_rate: float = Field(..., ge=0, le=1, description="Исторический процент неявок")
    phone_confirmed: bool = Field(..., description="Подтвержден ли телефон")

    @validator('gender')
    def validate_gender(cls, v):
        if v.upper() not in ['M', 'F']:
            raise ValueError('Пол должен быть M или F')
        return v.upper()


class AppointmentContext(BaseModel):
    """Контекст записи на прием"""
    appointment_id: int = Field(..., description="ID записи")
    patient_id: int = Field(..., description="ID пациента")
    doctor_id: int = Field(..., description="ID врача")
    appointment_time: datetime = Field(..., description="Время записи")
    visit_type: str = Field(..., description="Тип посещения")
    advance_booking_days: int = Field(..., ge=0, description="Дни заблаговременной записи")
    is_repeat_visit: bool = Field(..., description="Повторное посещение")
    reminder_sent: bool = Field(..., description="Отправлено ли напоминание")


class PredictionResult(BaseModel):
    """Результат прогнозирования неявки"""
    appointment_id: int = Field(..., description="ID записи")
    no_show_probability: float = Field(..., ge=0, le=1, description="Вероятность неявки")
    risk_level: RiskLevel = Field(..., description="Уровень риска")
    recommendation: str = Field(..., description="Рекомендация")
    confidence: float = Field(..., ge=0, le=1, description="Уровень уверенности модели")
    features_used: Dict[str, Any] = Field(default_factory=dict, description="Использованные признаки")


class TrainingMetrics(BaseModel):
    """Метрики качества модели"""
    accuracy: float = Field(..., ge=0, le=1, description="Точность")
    precision: float = Field(..., ge=0, le=1, description="Точность для класса 'неявка'")
    recall: float = Field(..., ge=0, le=1, description="Полнота для класса 'неявка'")
    f1_score: float = Field(..., ge=0, le=1, description="F1-мера для класса 'неявка'")
    training_samples: int = Field(..., ge=0, description="Количество образцов для обучения")
    test_samples: int = Field(..., ge=0, description="Количество образцов для тестирования")


class BatchPredictionRequest(BaseModel):
    """Запрос на массовое прогнозирование"""
    appointment_ids: List[int] = Field(..., description="Список ID записей")
    save_predictions: bool = Field(default=True, description="Сохранить прогнозы в БД")

    @validator('appointment_ids')
    def validate_appointment_ids(cls, v):
        if len(v) < 1:
            raise ValueError('Должен быть указан хотя бы один ID записи')
        if len(v) > 100:
            raise ValueError('Максимум 100 записей за раз')
        return v


class BatchPredictionResponse(BaseModel):
    """Ответ на массовое прогнозирование"""
    predictions: List[PredictionResult] = Field(..., description="Результаты прогнозирования")
    total_processed: int = Field(..., ge=0, description="Общее количество обработанных записей")
    successful_predictions: int = Field(..., ge=0, description="Количество успешных прогнозов")
    failed_predictions: int = Field(..., ge=0, description="Количество неудачных прогнозов")
    processing_time: float = Field(..., ge=0, description="Время обработки в секундах") 