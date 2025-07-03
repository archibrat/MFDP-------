import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models.prediction import PatientData, PredictionResult, PredictionType
from app.models.prediction import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

class MLModelManager:
    """Менеджер ML-моделей с поддержкой версионирования"""
    
    def __init__(self, models_path: str = "./models"):
        self.models_path = models_path
        self.models = {}
        self.feature_processors = {}
        self.model_metadata = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def load_model(self, model_version: str = "latest"):
        """Асинхронная загрузка модели"""
        try:
            loop = asyncio.get_event_loop()
            
            # Загружаем модель в отдельном потоке
            model_data = await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                model_version
            )
            
            self.models[model_version] = model_data['model']
            self.feature_processors[model_version] = model_data['processor']
            self.model_metadata[model_version] = model_data['metadata']
            
            logger.info(f"Model {model_version} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_version}: {str(e)}")
            return False
    
    def _load_model_sync(self, model_version: str) -> Dict:
        """Синхронная загрузка модели"""
        model_path = f"{self.models_path}/{model_version}_pipeline.pkl"
        
        with open(model_path, 'rb') as f:
            model_data = joblib.load(f)
        
        return {
            'model': model_data['model'],
            'processor': model_data.get('feature_processor'),
            'metadata': {
                'version': model_data.get('model_version', model_version),
                'training_date': model_data.get('training_date'),
                'metrics': model_data.get('metrics', {}),
                'feature_names': model_data.get('feature_names', [])
            }
        }

class MLService:
    """Основной ML-сервис"""
    
    def __init__(self):
        self.model_manager = MLModelManager()
        self.feature_engineer = None
        self.interpreter = None
        self._initialize_components()
    
    async def initialize(self):
        """Инициализация сервиса"""
        success = await self.model_manager.load_model("latest")
        if success:
            self._setup_feature_engineering()
            self._setup_interpreter()
        return success
    
    def _initialize_components(self):
        from app.services.ml.better_baseline_model import TimeAwareFeatureEngineering, ModelEvaluator
        
        self.feature_engineer = TimeAwareFeatureEngineering()
        self.model_evaluator = ModelEvaluator()
    
    def _setup_feature_engineering(self):
        """Настройка feature engineering"""
        # Настройка согласно paste.txt
        pass
    
    def _setup_interpreter(self):
        """Настройка интерпретатора модели"""
        latest_model = self.model_manager.models.get("latest")
        if latest_model:
            # Простая реализация интерпретатора без внешних зависимостей
            self.interpreter = {"model": latest_model}
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Основной метод предсказания"""
        try:
            # Подготовка данных
            processed_data = await self._prepare_data(request.patient_data)
            
            # Генерация признаков
            features = await self._generate_features(processed_data)
            
            # Предсказание
            prediction_result = await self._make_prediction(features)
            
            # Интерпретация
            explanation = None
            if request.include_explanation and self.interpreter:
                explanation = await self._generate_explanation(features)
            
            # Формирование ответа
            response = PredictionResponse(
                prediction_id=0,  # Будет установлен при сохранении
                prediction_value=prediction_result['probability'],
                confidence_score=prediction_result['confidence'],
                risk_level=prediction_result['risk_level'],
                recommendations=prediction_result['recommendations'],
                model_version=self.model_manager.model_metadata["latest"]["version"],
                created_at=datetime.utcnow(),
                explanation=explanation
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    async def _prepare_data(self, patient_data) -> pd.DataFrame:
        """Подготовка данных для модели"""
        # Конвертация в DataFrame
        data_dict = patient_data.model_dump()
        df = pd.DataFrame([data_dict])
        
        # Применение преобразований
        loop = asyncio.get_event_loop()
        processed_df = await loop.run_in_executor(
            self.model_manager.executor,
            self._process_data_sync,
            df
        )
        
        return processed_df
    
    def _process_data_sync(self, df: pd.DataFrame) -> pd.DataFrame:
        """Синхронная обработка данных"""
        # Создание временных признаков
        if 'appointment_day' in df.columns and 'scheduled_day' in df.columns:
            df['appointment_day'] = pd.to_datetime(df['appointment_day'])
            df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
            df['days_advance'] = (df['appointment_day'] - df['scheduled_day']).dt.days
            df['hours_advance'] = (df['appointment_day'] - df['scheduled_day']).dt.total_seconds() / 3600
        
        return df
    
    async def _generate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Генерация признаков"""
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not initialized")
            
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            self.model_manager.executor,
            self.feature_engineer.transform,
            data
        )
        return features
    
    async def _make_prediction(self, features: np.ndarray) -> Dict:
        """Выполнение предсказания"""
        model = self.model_manager.models["latest"]
        
        loop = asyncio.get_event_loop()
        probability = await loop.run_in_executor(
            self.model_manager.executor,
            lambda: model.predict_proba(features)[0, 1]
        )
        
        # Определение уровня риска с помощью улучшенной модели
        risk_level = self.model_evaluator.get_risk_level(probability)
        recommendations = self.model_evaluator.get_recommendations(risk_level)
        
        return {
            'probability': float(probability),
            'confidence': float(min(max(probability, 1-probability), 0.95)),
            'risk_level': risk_level,
            'recommendations': recommendations
        }
    
    async def _generate_explanation(self, features: np.ndarray) -> Optional[Dict]:
        """Генерация объяснения предсказания"""
        if not self.interpreter:
            return None
        
        # Простое объяснение на основе значимости признаков
        feature_names = self.model_manager.model_metadata.get("latest", {}).get("feature_names", [])
        
        if hasattr(self.model_manager.models["latest"], 'feature_importances_'):
            importances = self.model_manager.models["latest"].feature_importances_
            
            # Топ-3 самых важных признака
            if len(feature_names) == len(importances):
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                return {
                    "top_features": feature_importance[:3],
                    "explanation": "Предсказание основано на наиболее важных признаках"
                }
        
        return {"explanation": "Объяснение недоступно для данной модели"}

# Синглтон для использования в приложении
ml_service = MLService()

async def get_ml_service() -> MLService:
    """Dependency для получения ML-сервиса"""
    if not ml_service.model_manager.models:
        await ml_service.initialize()
    return ml_service
