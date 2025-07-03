"""
Улучшенная ML модель на основе better_baseline_4.py
Адаптировано для использования в MFDP проекте
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# ML библиотеки
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# Продвинутые модели
try:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    logging.warning("Продвинутые модели (LightGBM, XGBoost, CatBoost) недоступны")

logger = logging.getLogger(__name__)


class AdvancedModelArchitecture:
    """Класс для создания продвинутых ансамблевых моделей"""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.ensemble_model = None
        self.calibrated_models = {}
        
    def create_base_models(self) -> Dict[str, Any]:
        """Создание базовых моделей для ансамбля"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1,
                C=1.0,
                solver='liblinear'
            )
        }
        
        if ADVANCED_MODELS_AVAILABLE:
            models.update({
                'lightgbm': lgb.LGBMClassifier(
                    objective='binary',
                    boosting_type='gbdt',
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=7,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=7,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                ),
                'catboost': CatBoostClassifier(
                    loss_function='Logloss',
                    eval_metric='AUC',
                    iterations=200,
                    learning_rate=0.1,
                    depth=7,
                    random_state=42,
                    verbose=False,
                    thread_count=-1
                )
            })
        
        return models
    
    def create_stacking_ensemble(self, base_models: Dict[str, Any], meta_model=None) -> StackingClassifier:
        """Создание стекинг-ансамбля с временной валидацией"""
        if meta_model is None:
            meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Используем TimeSeriesSplit для стекинга
        stacking_classifier = StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=meta_model,
            cv=TimeSeriesSplit(n_splits=3),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_classifier
    
    def create_voting_ensemble(self, base_models: Dict[str, Any], voting='soft') -> VotingClassifier:
        """Создание voting-ансамбля"""
        voting_classifier = VotingClassifier(
            estimators=list(base_models.items()),
            voting=voting,
            n_jobs=-1
        )
        
        return voting_classifier


class TimeAwareFeatureEngineering:
    """Класс для создания временных признаков"""
    
    def __init__(self):
        self.feature_names = []
        self.label_encoders = {}
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание временных признаков"""
        result_df = df.copy()
        
        # Проверяем наличие временных колонок
        if 'scheduled_day' in df.columns and 'appointment_day' in df.columns:
            # Конвертируем в datetime если еще не сделано
            result_df['scheduled_day'] = pd.to_datetime(result_df['scheduled_day'])
            result_df['appointment_day'] = pd.to_datetime(result_df['appointment_day'])
            
            # Вычисляем разность во времени
            result_df['days_advance'] = (result_df['appointment_day'] - result_df['scheduled_day']).dt.days
            result_df['hours_advance'] = (result_df['appointment_day'] - result_df['scheduled_day']).dt.total_seconds() / 3600
            
            # Дополнительные временные признаки
            result_df['appointment_weekday'] = result_df['appointment_day'].dt.dayofweek
            result_df['appointment_hour'] = result_df['appointment_day'].dt.hour
            result_df['appointment_month'] = result_df['appointment_day'].dt.month
            
            # Признаки планирования
            result_df['is_same_day'] = (result_df['days_advance'] == 0).astype(int)
            result_df['is_weekend_appointment'] = (result_df['appointment_weekday'] >= 5).astype(int)
            result_df['is_early_morning'] = (result_df['appointment_hour'] < 9).astype(int)
            result_df['is_evening'] = (result_df['appointment_hour'] >= 17).astype(int)
            
        return result_df
    
    def create_patient_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков пациента"""
        result_df = df.copy()
        
        # Возрастные группы
        if 'age' in df.columns:
            result_df['age_group'] = pd.cut(
                result_df['age'], 
                bins=[0, 18, 30, 50, 65, 100], 
                labels=['child', 'young', 'adult', 'middle', 'senior']
            )
            
            # Кодируем возрастные группы
            if 'age_group' not in self.label_encoders:
                self.label_encoders['age_group'] = LabelEncoder()
                result_df['age_group_encoded'] = self.label_encoders['age_group'].fit_transform(
                    result_df['age_group'].astype(str)
                )
            else:
                result_df['age_group_encoded'] = self.label_encoders['age_group'].transform(
                    result_df['age_group'].astype(str)
                )
        
        # Кодирование пола
        if 'gender' in df.columns:
            if 'gender' not in self.label_encoders:
                self.label_encoders['gender'] = LabelEncoder()
                result_df['gender_encoded'] = self.label_encoders['gender'].fit_transform(result_df['gender'])
            else:
                result_df['gender_encoded'] = self.label_encoders['gender'].transform(result_df['gender'])
        
        # Суммарный показатель здоровья
        health_cols = ['hypertension', 'diabetes', 'alcoholism']
        available_health_cols = [col for col in health_cols if col in df.columns]
        if available_health_cols:
            result_df['health_conditions_count'] = result_df[available_health_cols].sum(axis=1)
        
        # Показатель риска
        if 'handcap' in df.columns:
            result_df['has_disability'] = (result_df['handcap'] > 0).astype(int)
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Полная обработка признаков"""
        result_df = self.create_temporal_features(df)
        result_df = self.create_patient_features(result_df)
        
        # Запоминаем названия признаков
        self.feature_names = result_df.columns.tolist()
        
        return result_df
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Преобразование данных в массив признаков для модели"""
        processed_df = self.fit_transform(df)
        
        # Выбираем только числовые колонки для модели
        numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
        return processed_df[numeric_columns].fillna(0).values


class PredictionPostprocessor:
    """Класс для постобработки предсказаний модели"""
    
    def __init__(self):
        self.calibrators = {}
        self.optimal_thresholds = {}
        
    def calibrate_probabilities(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                               method='isotonic') -> CalibratedClassifierCV:
        """Калибровка вероятностей модели"""
        calibrated_model = CalibratedClassifierCV(
            model, 
            method=method, 
            cv=TimeSeriesSplit(n_splits=3)
        )
        
        calibrated_model.fit(X_train, y_train)
        return calibrated_model
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, 
                              metric='f1') -> float:
        """Поиск оптимального порога классификации"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            else:
                score = accuracy_score(y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def apply_business_rules(self, predictions: np.ndarray, 
                           patient_data: pd.DataFrame) -> np.ndarray:
        """Применение бизнес-правил к предсказаниям"""
        adjusted_predictions = predictions.copy()
        
        # Правило 1: Высокий риск для пациентов с множественными заболеваниями
        if 'health_conditions_count' in patient_data.columns:
            high_risk_mask = patient_data['health_conditions_count'] >= 2
            adjusted_predictions[high_risk_mask] = np.maximum(
                adjusted_predictions[high_risk_mask], 0.6
            )
        
        # Правило 2: Повышенный риск для позднего времени
        if 'appointment_hour' in patient_data.columns:
            late_appointment_mask = patient_data['appointment_hour'] >= 17
            adjusted_predictions[late_appointment_mask] *= 1.1
        
        # Правило 3: Сниженный риск при SMS напоминании
        if 'sms_received' in patient_data.columns:
            sms_mask = patient_data['sms_received'] == True
            adjusted_predictions[sms_mask] *= 0.9
        
        # Ограничиваем значения в диапазоне [0, 1]
        adjusted_predictions = np.clip(adjusted_predictions, 0, 1)
        
        return adjusted_predictions


class ModelEvaluator:
    """Класс для оценки качества модели"""
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_proba: np.ndarray) -> Dict[str, float]:
        """Расчет комплексных метрик качества"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        return metrics
    
    @staticmethod
    def get_risk_level(probability: float) -> str:
        """Определение уровня риска"""
        if probability < 0.3:
            return "Низкий"
        elif probability < 0.6:
            return "Средний"
        else:
            return "Высокий"
    
    @staticmethod 
    def get_recommendations(risk_level: str) -> List[str]:
        """Получение рекомендаций"""
        if risk_level == "Низкий":
            return ["Стандартные процедуры", "Обычное напоминание"]
        elif risk_level == "Средний":
            return ["SMS-напоминание за день до приема", "Подтверждение записи"]
        else:
            return [
                "Обязательно связаться с пациентом",
                "Телефонный звонок за день до приема"
            ]


def save_model_pipeline(model, feature_processor, model_version: str, 
                       metrics: Dict, file_path: str) -> None:
    """Сохранение полного пайплайна модели"""
    pipeline_data = {
        'model': model,
        'feature_processor': feature_processor,
        'model_version': model_version,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'feature_names': feature_processor.feature_names if hasattr(feature_processor, 'feature_names') else []
    }
    
    with open(file_path, 'wb') as f:
        joblib.dump(pipeline_data, f)
    
    logger.info(f"Модель сохранена: {file_path}")


def load_model_pipeline(file_path: str) -> Dict:
    """Загрузка полного пайплайна модели"""
    try:
        with open(file_path, 'rb') as f:
            pipeline_data = joblib.load(f)
        
        logger.info(f"Модель загружена: {file_path}")
        return pipeline_data
    
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise 