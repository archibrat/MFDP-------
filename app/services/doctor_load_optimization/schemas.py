"""
Схемы данных для модуля оптимизации нагрузки врачей
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class OptimizationObjective(str, Enum):
    """Цели оптимизации"""
    MINIMIZE_WAIT_TIME = "minimize_wait_time"
    MAXIMIZE_UTILIZATION = "maximize_utilization"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_COST = "minimize_cost"


class DoctorProfile(BaseModel):
    """Профиль врача для оптимизации"""
    doctor_id: int = Field(..., description="ID врача")
    speciality: str = Field(..., description="Специальность")
    department: str = Field(..., description="Отделение")
    current_load: float = Field(..., ge=0.0, le=1.0, description="Текущая загруженность")
    avg_wait_time: float = Field(..., ge=0.0, description="Среднее время ожидания (минуты)")
    utilization_rate: float = Field(..., ge=0.0, le=1.0, description="Коэффициент использования времени")
    patient_satisfaction: float = Field(..., ge=0.0, le=1.0, description="Оценка удовлетворенности")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Сложность случаев")
    experience_years: int = Field(..., ge=0, description="Годы опыта")
    max_patients_per_day: int = Field(..., ge=1, description="Максимум пациентов в день")
    
    @field_validator('current_load', 'utilization_rate', 'patient_satisfaction', 'complexity_score')
    @classmethod
    def validate_probability_fields(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Значение должно быть в диапазоне [0, 1]')
        return v


class ScheduleSlot(BaseModel):
    """Слот расписания"""
    start_time: datetime = Field(..., description="Время начала")
    end_time: datetime = Field(..., description="Время окончания")
    doctor_id: int = Field(..., description="ID врача")
    patient_id: Optional[int] = Field(None, description="ID пациента")
    appointment_type: str = Field(..., description="Тип приема")
    complexity: float = Field(..., ge=0.0, le=1.0, description="Сложность случая")
    is_available: bool = Field(default=True, description="Доступность слота")
    room_id: Optional[int] = Field(None, description="ID кабинета")
    
    @field_validator('end_time')
    @classmethod
    def validate_time_range(cls, v, info):
        if 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError('Время окончания должно быть позже времени начала')
        return v


class OptimizationResult(BaseModel):
    """Результат оптимизации"""
    optimized_schedule: Dict[int, List[ScheduleSlot]] = Field(..., description="Оптимизированное расписание")
    load_distribution: Dict[int, float] = Field(..., description="Распределение нагрузки")
    wait_time_reduction: float = Field(..., description="Сокращение времени ожидания (%)")
    utilization_improvement: float = Field(..., description="Улучшение использования ресурсов (%)")
    recommendations: List[str] = Field(..., description="Рекомендации")
    efficiency_score: float = Field(..., ge=0.0, le=1.0, description="Оценка эффективности")
    cost_savings: float = Field(..., description="Экономия затрат")
    processing_time: float = Field(..., ge=0.0, description="Время обработки (секунды)")


class ResourceAllocationRequest(BaseModel):
    """Запрос на распределение ресурсов"""
    date: datetime = Field(..., description="Дата")
    speciality: str = Field(..., description="Специальность")
    appointment_type: str = Field(..., description="Тип приема")
    patient_count: int = Field(..., ge=1, description="Количество пациентов")
    complexity_level: float = Field(..., ge=0.0, le=1.0, description="Уровень сложности")
    duration_minutes: int = Field(..., ge=15, le=480, description="Длительность приема (минуты)")


class ResourceAllocationResult(BaseModel):
    """Результат распределения ресурсов"""
    allocated_doctors: List[int] = Field(..., description="Назначенные врачи")
    allocated_rooms: List[int] = Field(..., description="Назначенные кабинеты")
    schedule_slots: List[ScheduleSlot] = Field(..., description="Слоты расписания")
    efficiency_score: float = Field(..., ge=0.0, le=1.0, description="Оценка эффективности")
    cost_estimate: float = Field(..., description="Оценка затрат")
    recommendations: List[str] = Field(..., description="Рекомендации")


class PerformanceMetrics(BaseModel):
    """Метрики производительности врача"""
    doctor_id: int = Field(..., description="ID врача")
    period_start: datetime = Field(..., description="Начало периода")
    period_end: datetime = Field(..., description="Конец периода")
    total_appointments: int = Field(..., ge=0, description="Общее количество приемов")
    completed_appointments: int = Field(..., ge=0, description="Завершенные приемы")
    avg_consultation_time: float = Field(..., ge=0.0, description="Среднее время консультации")
    patient_satisfaction: float = Field(..., ge=0.0, le=1.0, description="Удовлетворенность пациентов")
    utilization_rate: float = Field(..., ge=0.0, le=1.0, description="Коэффициент использования")
    efficiency_score: float = Field(..., ge=0.0, le=1.0, description="Оценка эффективности")
    recommendations: List[str] = Field(..., description="Рекомендации по улучшению")


class OptimizationRequest(BaseModel):
    """Запрос на оптимизацию"""
    start_date: datetime = Field(..., description="Начальная дата")
    end_date: datetime = Field(..., description="Конечная дата")
    objective: OptimizationObjective = Field(default=OptimizationObjective.BALANCE_LOAD, description="Цель оптимизации")
    specialities: Optional[List[str]] = Field(None, description="Специальности для оптимизации")
    departments: Optional[List[str]] = Field(None, description="Отделения для оптимизации")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Ограничения")
    
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        if 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError('Конечная дата должна быть позже начальной')
        return v


class BatchOptimizationRequest(BaseModel):
    """Запрос на пакетную оптимизацию"""
    optimization_requests: List[OptimizationRequest] = Field(..., min_length=1, max_length=10, description="Запросы на оптимизацию")
    parallel_processing: bool = Field(default=True, description="Параллельная обработка")
    
    @field_validator('optimization_requests')
    @classmethod
    def validate_requests(cls, v):
        if len(v) > 10:
            raise ValueError('Максимум 10 запросов на оптимизацию')
        return v


class BatchOptimizationResponse(BaseModel):
    """Ответ на пакетную оптимизацию"""
    results: List[OptimizationResult] = Field(..., description="Результаты оптимизации")
    total_processed: int = Field(..., ge=0, description="Общее количество обработанных запросов")
    successful_optimizations: int = Field(..., ge=0, description="Успешные оптимизации")
    failed_optimizations: int = Field(..., ge=0, description="Неудачные оптимизации")
    processing_time: float = Field(..., ge=0.0, description="Время обработки (секунды)")
    summary_metrics: Dict[str, float] = Field(..., description="Сводные метрики") 