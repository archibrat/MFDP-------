"""
Оптимизированный сервис прогнозирования неявок пациентов
Использует модульную архитектуру для лучшей производительности и тестируемости
"""

import logging
import json
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlmodel import Session

from app.models.medialog import NoShowPrediction
from app.models.base import BaseResponse

from app.services.no_show_prediction import (
    MedialogDatabaseConnector, FeatureExtractor, NoShowPredictor, RiskAssessment,
    PatientProfile, AppointmentContext, PredictionResult, TrainingMetrics,
    BatchPredictionRequest, BatchPredictionResponse
)


class NoShowPredictionService:
    """
    Оптимизированный сервис прогнозирования неявок
    Использует модульную архитектуру и batch-обработку
    """

    def __init__(self, db_connector: MedialogDatabaseConnector):
        """
        Инициализация сервиса
        
        Args:
            db_connector: Коннектор к базе данных
        """
        self.db_connector = db_connector
        self.predictor = NoShowPredictor()
        self.risk_assessor = RiskAssessment()
        self.feature_extractor = FeatureExtractor()
        self.logger = logging.getLogger(__name__)

    def train_model(self, days_back: int = 365) -> TrainingMetrics:
        """
        Обучение модели на исторических данных
        
        Args:
            days_back: Количество дней для обучения
            
        Returns:
            Метрики качества модели
        """
        try:
            self.logger.info(f"Начало обучения модели на данных за {days_back} дней")
            start_time = time.time()
            
            # Получение исторических данных
            historical_data = self.db_connector.get_historical_data(days_back)
            
            if historical_data.empty:
                raise ValueError("Недостаточно данных для обучения модели")
                
            self.logger.info(f"Получено {len(historical_data)} записей для обучения")
            
            # Обучение модели
            metrics = self.predictor.train(historical_data)
            
            training_time = time.time() - start_time
            self.logger.info(f"Модель обучена за {training_time:.2f} секунд")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            raise

    def predict_no_show(self, appointment_id: int) -> Optional[PredictionResult]:
        """
        Прогнозирование неявки для конкретной записи
        
        Args:
            appointment_id: ID записи на прием
            
        Returns:
            Результат прогнозирования или None при ошибке
        """
        try:
            start_time = time.time()
            
            # Получение данных
            appointment = self.db_connector.get_appointment_data(appointment_id)
            if not appointment:
                self.logger.warning(f"Запись {appointment_id} не найдена")
                return None

            patient = self.db_connector.get_patient_data(appointment.patient_id)
            if not patient:
                self.logger.warning(f"Пациент не найден для записи {appointment_id}")
                return None

            # Извлечение признаков
            features = self.feature_extractor.extract_features(patient, appointment)

            # Прогнозирование
            probability = self.predictor.predict(features)
            
            # Оценка риска
            risk_level = self.risk_assessor.assess_risk(probability)
            recommendation = self.risk_assessor.get_recommendation(
                risk_level, 
                patient_context=patient.dict()
            )

            prediction_time = time.time() - start_time
            self.logger.info(f"Прогноз для записи {appointment_id} выполнен за {prediction_time:.3f} сек")

            return PredictionResult(
                appointment_id=appointment_id,
                no_show_probability=probability,
                risk_level=risk_level,
                recommendation=recommendation,
                confidence=0.85,  # В реальности должно вычисляться моделью
                features_used=features
            )

        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования для записи {appointment_id}: {e}")
            return None

    def batch_predict(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Массовое прогнозирование для списка записей
        Оптимизировано для batch-обработки
        
        Args:
            request: Запрос на массовое прогнозирование
            
        Returns:
            Результаты массового прогнозирования
        """
        try:
            start_time = time.time()
            appointment_ids = request.appointment_ids
            
            self.logger.info(f"Начало batch-прогнозирования для {len(appointment_ids)} записей")
            
            # Получение данных для всех записей
            patients_data = []
            appointments_data = []
            valid_appointment_ids = []
            
            for appointment_id in appointment_ids:
                appointment = self.db_connector.get_appointment_data(appointment_id)
                if not appointment:
                    self.logger.warning(f"Запись {appointment_id} не найдена")
                    continue
                    
                patient = self.db_connector.get_patient_data(appointment.patient_id)
                if not patient:
                    self.logger.warning(f"Пациент не найден для записи {appointment_id}")
                    continue
                    
                patients_data.append(patient)
                appointments_data.append(appointment)
                valid_appointment_ids.append(appointment_id)
            
            if not patients_data:
                raise ValueError("Нет валидных записей для прогнозирования")
            
            # Batch-извлечение признаков
            features_list = []
            for patient, appointment in zip(patients_data, appointments_data):
                features = self.feature_extractor.extract_features(patient, appointment)
                features_list.append(features)
            
            # Batch-прогнозирование
            probabilities = self.predictor.predict_batch(features_list)
            
            # Формирование результатов
            predictions = []
            successful_predictions = 0
            failed_predictions = 0
            
            for i, (appointment_id, probability, patient, appointment) in enumerate(
                zip(valid_appointment_ids, probabilities, patients_data, appointments_data)
            ):
                try:
                    risk_level = self.risk_assessor.assess_risk(probability)
                    recommendation = self.risk_assessor.get_recommendation(
                        risk_level, 
                        patient_context=patient.dict()
                    )
                    
                    prediction = PredictionResult(
                        appointment_id=appointment_id,
                        no_show_probability=probability,
                        risk_level=risk_level,
                        recommendation=recommendation,
                        confidence=0.85,
                        features_used=features_list[i]
                    )
                    
                    predictions.append(prediction)
                    successful_predictions += 1
                    
                    # Сохранение в БД если требуется
                    if request.save_predictions:
                        self._save_prediction_to_db(prediction)
                        
                except Exception as e:
                    self.logger.error(f"Ошибка обработки прогноза для записи {appointment_id}: {e}")
                    failed_predictions += 1
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Batch-прогнозирование завершено: "
                f"{successful_predictions} успешных, {failed_predictions} неудачных, "
                f"время: {processing_time:.2f} сек"
            )
            
            return BatchPredictionResponse(
                predictions=predictions,
                total_processed=len(appointment_ids),
                successful_predictions=successful_predictions,
                failed_predictions=failed_predictions,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка batch-прогнозирования: {e}")
            raise

    def save_prediction_to_db(self, prediction: PredictionResult, 
                            model_version: str = "1.0") -> bool:
        """
        Сохранение прогноза в базу данных
        
        Args:
            prediction: Результат прогнозирования
            model_version: Версия модели
            
        Returns:
            True при успешном сохранении
        """
        try:
            no_show_prediction = NoShowPrediction(
                appointment_id=prediction.appointment_id,
                prediction_probability=prediction.no_show_probability,
                risk_level=prediction.risk_level.value,
                recommendations=prediction.recommendation,
                model_version=model_version,
                confidence_score=prediction.confidence,
                features_used=json.dumps(prediction.features_used)
            )
            
            return self.db_connector.save_prediction(no_show_prediction)
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения прогноза: {e}")
            return False

    def _save_prediction_to_db(self, prediction: PredictionResult) -> bool:
        """Внутренний метод для сохранения прогноза"""
        return self.save_prediction_to_db(prediction)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Returns:
            Информация о модели
        """
        if not self.predictor.is_trained:
            return {"status": "not_trained"}
            
        return {
            "status": "trained",
            "model_type": self.predictor.model_type,
            "feature_count": len(self.predictor.feature_names),
            "training_metrics": self.predictor.training_metrics.dict() if self.predictor.training_metrics else None,
            "feature_importance": self.predictor.get_feature_importance()
        }

    def update_risk_thresholds(self, low_risk: float, high_risk: float) -> None:
        """
        Обновление порогов риска
        
        Args:
            low_risk: Новый порог низкого риска
            high_risk: Новый порог высокого риска
        """
        from app.services.no_show_prediction.risk import RiskThresholds
        
        new_thresholds = RiskThresholds(low_risk, high_risk)
        self.risk_assessor = RiskAssessment(new_thresholds)
        
        self.logger.info(f"Пороги риска обновлены: low={low_risk}, high={high_risk}") 