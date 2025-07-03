"""
Feature Store для ML-модуля
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from app.services.ml_production.dal import MLDataAccessLayer


logger = logging.getLogger(__name__)


class FeatureStore:
    """Хранилище и инженерия признаков для ML"""
    
    def __init__(self, dal: MLDataAccessLayer):
        self.dal = dal

    def extract_features(self, planning_id: int) -> Optional[Dict[str, Any]]:
        """Извлекает и инженерит признаки для planning_id"""
        try:
            raw_features = self.dal.extract_noshow_features(planning_id)
            if not raw_features:
                return None
                
            # Инженерия признаков
            features = self._engineer_features(raw_features)
            
            # Сохранение в витрину
            self.dal.save_features(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Ошибка извлечения признаков для planning_id {planning_id}: {e}")
            return None

    def _engineer_features(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """Инженерия дополнительных признаков"""
        features = raw_features.copy()
        
        # Временные признаки
        features['is_morning'] = 1 if features['hour'] < 12 else 0
        features['is_evening'] = 1 if features['hour'] >= 17 else 0
        features['is_holiday_month'] = 1 if features['month'] in [1, 5, 12] else 0
        
        # Признаки пациента
        if features['patient_age']:
            features['age_group'] = self._get_age_group(features['patient_age'])
            features['is_elderly'] = 1 if features['patient_age'] >= 65 else 0
        else:
            features['age_group'] = 'unknown'
            features['is_elderly'] = 0
            
        # Признаки записи
        features['is_urgent_booking'] = 1 if features['advance_booking_days'] <= 1 else 0
        features['is_far_booking'] = 1 if features['advance_booking_days'] >= 30 else 0
        
        # Контактность
        features['contact_score'] = (
            (1 if features['has_email'] else 0) + 
            (1 if not features['not_send_sms'] else 0)
        )
        
        # Риск на основе истории
        features['noshow_risk_historical'] = min(features['past_noshows_count'] / 5.0, 1.0)
        
        return features

    def _get_age_group(self, age: int) -> str:
        """Группирует возраст"""
        if age < 18:
            return 'child'
        elif age < 35:
            return 'young_adult'
        elif age < 55:
            return 'middle_age'
        elif age < 70:
            return 'senior'
        else:
            return 'elderly'

    def load_training_dataset(self, days_back: int = 365) -> pd.DataFrame:
        """Загружает обучающий датасет"""
        try:
            df = self.dal.load_training_data(days_back)
            
            if df.empty:
                logger.warning("Обучающий датасет пуст")
                return df
                
            # Очистка данных
            df = self._clean_dataset(df)
            
            # Дополнительная инженерия признаков
            df = self._add_engineered_features(df)
            
            logger.info(f"Загружен датасет: {len(df)} записей, {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки обучающего датасета: {e}")
            return pd.DataFrame()

    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка датасета"""
        # Удаление дубликатов
        df = df.drop_duplicates(subset=['planning_id'])
        
        # Заполнение пропусков
        df['patient_age'] = df['patient_age'].fillna(df['patient_age'].median())
        df['patient_gender'] = df['patient_gender'].fillna('U')
        df['doctor_specialization_id'] = df['doctor_specialization_id'].fillna(0)
        df['department_id'] = df['department_id'].fillna(0)
        
        # Фильтрация аномальных значений
        df = df[df['advance_booking_days'] >= 0]
        df = df[df['advance_booking_days'] <= 365]
        df = df[df['patient_age'] >= 0]
        df = df[df['patient_age'] <= 120]
        
        return df

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет инженерные признаки в датасет"""
        # Временные признаки
        df['is_morning'] = (df['hour'] < 12).astype(int)
        df['is_evening'] = (df['hour'] >= 17).astype(int)
        df['is_holiday_month'] = df['month'].isin([1, 5, 12]).astype(int)
        
        # Группы возраста
        df['age_group'] = df['patient_age'].apply(self._get_age_group)
        df['is_elderly'] = (df['patient_age'] >= 65).astype(int)
        
        # Признаки записи
        df['is_urgent_booking'] = (df['advance_booking_days'] <= 1).astype(int)
        df['is_far_booking'] = (df['advance_booking_days'] >= 30).astype(int)
        
        # Контактность
        df['contact_score'] = (
            df['has_email'].astype(int) + 
            (~df['not_send_sms']).astype(int)
        )
        
        # Исторический риск
        df['noshow_risk_historical'] = np.minimum(df['past_noshows_count'] / 5.0, 1.0)
        
        return df

    def get_feature_names(self) -> List[str]:
        """Возвращает список названий признаков"""
        return [
            'weekday', 'hour', 'month', 'is_weekend',
            'patient_age', 'past_noshows_count',
            'advance_booking_days',
            'doctor_specialization_id', 'department_id',
            'not_send_sms', 'has_email',
            'is_morning', 'is_evening', 'is_holiday_month',
            'is_elderly', 'is_urgent_booking', 'is_far_booking',
            'contact_score', 'noshow_risk_historical'
        ]

    def prepare_features_for_prediction(self, features: Dict[str, Any]) -> np.ndarray:
        """Подготавливает признаки для предсказания"""
        feature_names = self.get_feature_names()
        feature_vector = []
        
        for feature_name in feature_names:
            value = features.get(feature_name, 0)
            if isinstance(value, bool):
                value = int(value)
            elif value is None:
                value = 0
            feature_vector.append(float(value))
            
        return np.array(feature_vector).reshape(1, -1) 