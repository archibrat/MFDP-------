from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from app.models.prediction import PredictionType

class PatientDataCreate(PatientDataBase):
    """Схема для создания данных пациента"""
    client_id: str = Field(max_length=50)
    booking_id: str = Field(max_length=50)

class PredictionRequest(BaseModel):
    """Запрос на предсказание"""
    patient_data: PatientDataCreate
    prediction_types: List[PredictionType] = Field(default=[PredictionType.NO_SHOW_RISK])
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

class BatchPredictionRequest(BaseModel):
    """Запрос на пакетное предсказание"""
    patients_data: List[PatientDataCreate]
    prediction_type: PredictionType = PredictionType.NO_SHOW_RISK

class ModelStatusResponse(BaseModel):
    """Статус модели"""
    model_version: str
    status: str
    last_update: datetime
    metrics: dict
    active_predictions: int
