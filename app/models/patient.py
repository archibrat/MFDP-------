from datetime import datetime
from enum import Enum
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class PredictionType(str, Enum):
    NO_SHOW_RISK = "no_show_risk"
    PATIENT_FLOW = "patient_flow"
    RESOURCE_OPTIMIZATION = "resource_optimization"

class PredictionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"

class Patient(SQLModel, table=True):
    """Модель пациента"""
    __tablename__ = "legacy_patients"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: int = Field(unique=True, index=True, description="ID пациента")
    gender: str = Field(max_length=1, description="Пол пациента (M/F)")
    age: int = Field(ge=0, le=120, description="Возраст пациента")
    neighbourhood: str = Field(max_length=100, description="Район проживания")
    scholarship: bool = Field(default=False, description="Наличие стипендии")
    hypertension: bool = Field(default=False, description="Гипертония")
    diabetes: bool = Field(default=False, description="Диабет")
    alcoholism: bool = Field(default=False, description="Алкоголизм")
    handcap: int = Field(default=0, description="Степень инвалидности")

class PatientDataBase(SQLModel):
    """Базовая модель для данных пациента"""
    age: int = Field(ge=0, le=120)
    gender: str = Field(max_length=10)
    district: str = Field(max_length=100)
    scholarship: bool = Field(default=False)
    condition_a: bool = Field(default=False)  # Hypertension
    condition_b: bool = Field(default=False)  # Diabetes
    condition_c: bool = Field(default=False)  # Alcoholism
    accessibility_level: int = Field(ge=0, le=4, default=0)
    notification_sent: bool = Field(default=False)
    planned_date: datetime
    session_date: datetime

class PatientData(PatientDataBase, table=True):
    """Модель данных пациента для хранения в БД"""
    __tablename__ = "patient_data"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: str = Field(index=True)
    booking_id: str = Field(unique=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Связи
    predictions: List["PredictionResult"] = Relationship(back_populates="patient_data")

class PredictionResult(SQLModel, table=True):
    """Результат ML-предсказания"""
    __tablename__ = "prediction_results"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Связи с другими сущностями
    task_id: Optional[int] = Field(foreign_key="mltask.id")
    patient_data_id: int = Field(foreign_key="patient_data.id")
    user_id: int = Field(foreign_key="user.id")
    
    # Основные поля предсказания
    prediction_type: PredictionType
    prediction_value: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(max_length=20)
    
    # Метаданные
    model_version: str = Field(max_length=50)
    features_used: str  # JSON строка с использованными признаками
    status: PredictionStatus = Field(default=PredictionStatus.PENDING)
    
    # Временные метки
    created_at: datetime = Field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = Field(default=None)
    
    # Связи
    task: Optional["MLTask"] = Relationship()
    patient_data: PatientData = Relationship(back_populates="predictions")
    creator: Optional["User"] = Relationship()

class ModelMetrics(SQLModel, table=True):
    """Метрики качества модели"""
    __tablename__ = "model_metrics"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    model_version: str = Field(max_length=50)
    metric_name: str = Field(max_length=50)
    metric_value: float
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)
    data_period_start: datetime
    data_period_end: datetime

class FeatureImportance(SQLModel, table=True):
    """Важность признаков модели"""
    __tablename__ = "feature_importance"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    model_version: str = Field(max_length=50)
    feature_name: str = Field(max_length=100)
    importance_value: float
    rank: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
