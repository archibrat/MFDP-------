"""
Модуль прогнозирования неявок пациентов
Оптимизированная архитектура с разделением ответственности
"""

from .schemas import (
    RiskLevel, PatientProfile, AppointmentContext, PredictionResult,
    TrainingMetrics, BatchPredictionRequest, BatchPredictionResponse
)
from .database_connector import DatabaseConnector, MedialogDatabaseConnector
from .feature_extractor import FeatureExtractor
from .predictor import NoShowPredictor
from .risk import RiskAssessment, RiskThresholds, Recommendation, RecommendationType

__all__ = [
    # Схемы
    'RiskLevel', 'PatientProfile', 'AppointmentContext', 'PredictionResult',
    'TrainingMetrics', 'BatchPredictionRequest', 'BatchPredictionResponse',
    
    # База данных
    'DatabaseConnector', 'MedialogDatabaseConnector',
    
    # Признаки
    'FeatureExtractor',
    
    # Модель
    'NoShowPredictor',
    
    # Риски
    'RiskAssessment', 'RiskThresholds', 'Recommendation', 'RecommendationType'
] 