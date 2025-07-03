from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class PredictionType(str, Enum):
    """Типы предсказаний"""
    NO_SHOW = "no_show"
    NO_SHOW_RISK = "no_show_risk"
    SCHEDULING_OPTIMIZATION = "scheduling_optimization"


class PatientData(BaseModel):
    """Данные пациента для предсказания"""
    patient_id: int
    appointment_id: int
    gender: str
    age: int
    neighbourhood: str
    scholarship: bool
    hypertension: bool
    diabetes: bool
    alcoholism: bool
    handcap: int
    sms_received: bool
    scheduled_day: str
    appointment_day: str


class PredictionRequest(BaseModel):
    """Запрос на предсказание"""
    patient_data: PatientData
    prediction_type: PredictionType = PredictionType.NO_SHOW
    include_explanation: bool = Field(default=False)


class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    prediction_id: int
    prediction_value: float
    confidence_score: float
    risk_level: str
    recommendations: List[str]
    model_version: str
    created_at: datetime
    explanation: Optional[dict] = None


class PredictionResult(BaseModel):
    """Результат предсказания"""
    probability: float
    confidence: float
    risk_level: str
    recommendations: List[str]


class BatchPredictionRequest(BaseModel):
    """Запрос на пакетное предсказание"""
    patients_data: List[PatientData]
    prediction_type: PredictionType = PredictionType.NO_SHOW_RISK


class ModelStatusResponse(BaseModel):
    """Статус модели"""
    model_version: str
    status: str
    last_update: datetime
    metrics: dict
    active_predictions: int


class PatientDataCreate(BaseModel):
    """Данные для создания пациента (используется в batch-запросах)"""
    patient_id: int
    appointment_id: int
    gender: str
    age: int
    neighbourhood: str
    scholarship: bool
    hypertension: bool
    diabetes: bool
    alcoholism: bool
    handcap: int
    sms_received: bool
    scheduled_day: str
    appointment_day: str
