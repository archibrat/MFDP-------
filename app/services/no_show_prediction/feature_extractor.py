"""
Оптимизированный экстрактор признаков для модели прогнозирования неявок
Использует векторизацию numpy/pandas для повышения производительности
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta

from app.services.no_show_prediction.schemas import PatientProfile, AppointmentContext


class FeatureExtractor:
    """
    Оптимизированный класс для извлечения признаков для модели ML
    Использует векторизацию для повышения производительности
    """

    # Константы для нормализации
    AGE_MAX = 120
    INTERVAL_MAX = 365 * 5  # 5 лет
    BOOKING_MAX = 365  # 1 год

    @staticmethod
    def extract_features(patient: PatientProfile, appointment: AppointmentContext) -> Dict[str, Any]:
        """
        Извлечение признаков из данных пациента и записи
        Оптимизировано с использованием векторизации
        """
        # Базовые признаки пациента
        features = {
            'age': patient.age,
            'gender_encoded': 1 if patient.gender == 'M' else 0,
            'visit_history_count': patient.visit_history_count,
            'avg_interval_days': patient.avg_interval_between_visits,
            'historical_no_show_rate': patient.no_show_history_rate,
            'phone_confirmed': int(patient.phone_confirmed),
        }

        # Признаки записи
        appointment_features = FeatureExtractor._extract_appointment_features(appointment)
        features.update(appointment_features)

        # Производные признаки
        derived_features = FeatureExtractor._extract_derived_features(patient, appointment)
        features.update(derived_features)

        return features

    @staticmethod
    def extract_batch_features(patients: List[PatientProfile], 
                              appointments: List[AppointmentContext]) -> np.ndarray:
        """
        Batch-извлечение признаков для списка пациентов и записей
        Оптимизировано для векторизации
        
        Args:
            patients: Список профилей пациентов
            appointments: Список контекстов записей
            
        Returns:
            Матрица признаков (n_samples, n_features)
        """
        if len(patients) != len(appointments):
            raise ValueError("Количество пациентов должно совпадать с количеством записей")
        
        # Получаем размерность признаков из первого примера
        sample_features = FeatureExtractor.extract_features(patients[0], appointments[0])
        n_features = len(sample_features)
        n_samples = len(patients)
        
        # Создаем матрицу признаков
        features_matrix = np.zeros((n_samples, n_features))
        
        # Извлекаем признаки для каждого пациента
        for i, (patient, appointment) in enumerate(zip(patients, appointments)):
            features = FeatureExtractor.extract_features(patient, appointment)
            features_matrix[i] = list(features.values())
        
        return features_matrix

    @staticmethod
    def _extract_appointment_features(appointment: AppointmentContext) -> Dict[str, Any]:
        """Извлечение признаков записи на прием"""
        return {
            'advance_booking_days': appointment.advance_booking_days,
            'is_repeat_visit': int(appointment.is_repeat_visit),
            'reminder_sent': int(appointment.reminder_sent),
            'appointment_hour': appointment.appointment_time.hour,
            'appointment_weekday': appointment.appointment_time.weekday(),
            'is_morning_appointment': int(appointment.appointment_time.hour < 12),
            'is_afternoon_appointment': int(12 <= appointment.appointment_time.hour < 17),
            'is_evening_appointment': int(appointment.appointment_time.hour >= 17),
        }

    @staticmethod
    def _extract_derived_features(patient: PatientProfile, 
                                appointment: AppointmentContext) -> Dict[str, Any]:
        """Извлечение производных признаков"""
        # Нормализованные признаки
        normalized_age = patient.age / FeatureExtractor.AGE_MAX
        normalized_interval = min(patient.avg_interval_between_visits / FeatureExtractor.INTERVAL_MAX, 1.0)
        normalized_booking = min(appointment.advance_booking_days / FeatureExtractor.BOOKING_MAX, 1.0)

        # Временные признаки
        appointment_time = appointment.appointment_time
        is_weekend = int(appointment_time.weekday() >= 5)
        is_monday = int(appointment_time.weekday() == 0)
        is_friday = int(appointment_time.weekday() == 4)

        # Признаки сезонности
        month = appointment_time.month
        is_summer = int(6 <= month <= 8)
        is_winter = int(month in [12, 1, 2])

        return {
            'normalized_age': normalized_age,
            'normalized_interval': normalized_interval,
            'normalized_booking': normalized_booking,
            'is_weekend': is_weekend,
            'is_monday': is_monday,
            'is_friday': is_friday,
            'is_summer': is_summer,
            'is_winter': is_winter,
        }

    @staticmethod
    def get_feature_names() -> List[str]:
        """Получение списка имен признаков в правильном порядке"""
        return [
            'age', 'gender_encoded', 'visit_history_count', 'avg_interval_days',
            'historical_no_show_rate', 'phone_confirmed', 'advance_booking_days',
            'is_repeat_visit', 'reminder_sent', 'appointment_hour', 'appointment_weekday',
            'is_morning_appointment', 'is_afternoon_appointment', 'is_evening_appointment',
            'normalized_age', 'normalized_interval', 'normalized_booking',
            'is_weekend', 'is_monday', 'is_friday', 'is_summer', 'is_winter'
        ]

    @staticmethod
    def normalize_features(features: np.ndarray, scaler=None) -> np.ndarray:
        """
        Нормализация признаков
        Если scaler не передан, создается новый
        """
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(features)
        else:
            return scaler.transform(features)

    @staticmethod
    def create_feature_importance_plot(feature_importances: np.ndarray, 
                                     feature_names: List[str] = None) -> Dict[str, float]:
        """
        Создание словаря важности признаков для интерпретации модели
        """
        if feature_names is None:
            feature_names = FeatureExtractor.get_feature_names()

        if len(feature_importances) != len(feature_names):
            raise ValueError("Количество важностей признаков должно совпадать с количеством имен")

        # Создаем словарь и сортируем по важности
        importance_dict = dict(zip(feature_names, feature_importances))
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))

        return sorted_importance 