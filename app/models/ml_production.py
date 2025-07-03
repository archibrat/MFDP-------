"""
Модели данных для продакшн ML-модуля
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


# Витрины признаков для ML
class NSFeatures(BaseModel):
    """Витрина признаков для прогнозирования неявок"""
    
    planning_id: int
    patient_id: int
    medecin_id: int
    department_id: int
    
    # Временные признаки
    hour_of_day: int
    day_of_week: int
    days_until_appointment: int
    is_weekend: bool
    is_holiday: bool
    
    # Пациентские признаки
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    distance_km: Optional[float] = None
    previous_noshow_count: int = 0
    previous_appointment_count: int = 0
    avg_delay_minutes: Optional[float] = None
    
    # Записные признаки
    appointment_type: Optional[str] = None
    is_emergency: bool = False
    is_first_visit: bool = False
    estimated_duration: Optional[int] = None
    
    # Контекстные признаки
    doctor_workload_pct: Optional[float] = None
    department_queue_length: int = 0
    weather_condition: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LoadStat(BaseModel):
    """Статистика загрузки по отделениям"""
    
    id: Optional[int] = None
    date: date
    department_id: int
    medecin_id: int
    
    scheduled_appointments: int = 0
    completed_appointments: int = 0
    no_show_count: int = 0
    average_duration: Optional[float] = None
    utilization_rate: float = 0.0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PLSnapshot(BaseModel):
    """Снимки состояния планирования для аналитики"""
    
    id: Optional[int] = None
    snapshot_date: datetime
    planning_id: int
    
    original_date: datetime
    current_date: datetime
    status: str
    
    prediction_score: Optional[float] = None
    actual_outcome: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Enums для API
class EventType(str, Enum):
    ARRIVE_DATE = "ARRIVE_DATE"
    CANCELLED = "CANCELLED"
    CONS_DURATION = "CONS_DURATION"
    NO_SHOW = "NO_SHOW"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# API Schemas
class NoShowPredictionRequest(BaseModel):
    """Запрос прогноза неявки"""
    planning_id: int = Field(..., description="ID записи на прием")


class NoShowPredictionResponse(BaseModel):
    """Ответ прогноза неявки"""
    planning_id: int
    probability: float = Field(..., ge=0.0, le=1.0, description="Вероятность неявки")
    risk_level: RiskLevel
    recommendations: List[str] = Field(default_factory=list)
    features_used: Dict[str, Any] = Field(default_factory=dict)


class LoadBalanceRequest(BaseModel):
    """Запрос оптимизации нагрузки"""
    start_date: date
    end_date: date
    department_ids: Optional[List[int]] = None
    target_utilization: float = Field(default=0.8, ge=0.1, le=1.0)


class LoadBalanceResponse(BaseModel):
    """Ответ оптимизации нагрузки"""
    optimized_assignments: Dict[int, int] = Field(default_factory=dict)  # planning_id -> new_medecin_id
    metrics: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class SchedulerEvent(BaseModel):
    """Событие планировщика"""
    event_type: EventType
    planning_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)


class SchedulerResponse(BaseModel):
    """Ответ планировщика"""
    planning_id: int
    old_time: Optional[datetime] = None
    new_time: Optional[datetime] = None
    adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    notifications_sent: List[str] = Field(default_factory=list)


class TrainingRequest(BaseModel):
    """Запрос обучения модели"""
    days_back: int = Field(default=365, ge=30, le=1095)
    model_type: str = Field(default="xgboost")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4)


class TrainingResponse(BaseModel):
    """Ответ обучения модели"""
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    samples_used: int


class PredictionMetrics(BaseModel):
    """Метрики качества предсказаний"""
    date: date
    total_predictions: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    false_positive_rate: Optional[float] = None


class DoctorLoadMetrics(BaseModel):
    """Метрики загрузки врача"""
    medecin_id: int
    medecin_name: str
    current_utilization: float
    target_utilization: float
    scheduled_today: int
    avg_appointment_duration: float
    estimated_no_shows: int


class RealtimeQueueStatus(BaseModel):
    """Статус очереди в реальном времени"""
    department_id: int
    department_name: str
    current_queue_length: int
    estimated_wait_time: int  # minutes
    next_available_slot: Optional[datetime] = None
    available_doctors: List[DoctorLoadMetrics] = Field(default_factory=list)


class OptimizationSuggestion(BaseModel):
    """Предложение по оптимизации"""
    suggestion_type: str
    planning_id: int
    current_medecin_id: int
    suggested_medecin_id: int
    expected_improvement: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)