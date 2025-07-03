"""
No-Show Predictor с XGBoost
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from app.models.ml_production import RiskLevel
from app.services.ml_production.feature_store import FeatureStore


logger = logging.getLogger(__name__)


class NoShowPredictor:
    """Предиктор неявок на основе XGBoost"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.model_path = model_path or "models/noshow_model.pkl"
        
        if xgb is None:
            logger.warning("XGBoost не установлен, используется логистическая регрессия")
            
        # Попытка загрузки сохраненной модели
        self._load_model()

    def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """Обучает модель на обучающих данных"""
        try:
            if training_data.empty:
                raise ValueError("Обучающий датасет пуст")
                
            # Подготовка данных
            feature_columns = [col for col in training_data.columns 
                             if col not in ['planning_id', 'target', 'created_at', 'actual_noshow', 'id']]
            
            X = training_data[feature_columns]
            y = training_data['target']
            
            logger.info(f"Обучение модели на {len(X)} записях с {len(feature_columns)} признаками")
            
            # Разделение на train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Обучение модели
            if xgb is not None:
                self.model = xgb.XGBClassifier(
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                )
            else:
                # Fallback на sklearn
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            
            self.model.fit(X_train, y_train)
            self.feature_names = list(X.columns)
            self.is_trained = True
            
            # Оценка качества
            metrics = self._evaluate_model(X_test, y_test)
            
            # Сохранение модели
            self._save_model()
            
            logger.info(f"Модель обучена. Метрики: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            return {}

    def predict(self, features: Dict[str, Any]) -> float:
        """Предсказывает вероятность неявки"""
        try:
            if not self.is_trained:
                logger.warning("Модель не обучена, возвращаем среднюю вероятность")
                return 0.3
                
            # Подготовка признаков
            feature_vector = self._prepare_features(features)
            
            # Предсказание
            probability = self.model.predict_proba(feature_vector)[0][1]
            
            return float(probability)
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return 0.3

    def get_risk_level(self, probability: float) -> RiskLevel:
        """Определяет уровень риска по вероятности"""
        if probability >= 0.7:
            return RiskLevel.HIGH
        elif probability >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def get_recommendations(self, probability: float, features: Dict[str, Any]) -> List[str]:
        """Генерирует рекомендации по снижению риска неявки"""
        recommendations = []
        
        if probability >= 0.8:
            recommendations.append("Критический риск: рассмотреть двойное бронирование")
            
        if probability >= 0.6:
            recommendations.append("Высокий риск: отправить SMS-напоминание")
            
        if features.get('not_send_sms', False):
            recommendations.append("Пациент не получает SMS: рассмотреть звонок")
            
        if features.get('advance_booking_days', 0) > 30:
            recommendations.append("Запись заблаговременно: отправить напоминание накануне")
            
        if features.get('past_noshows_count', 0) > 2:
            recommendations.append("История неявок: требуется подтверждение записи")
            
        if features.get('is_weekend', False):
            recommendations.append("Запись на выходные: дополнительное напоминание")
            
        if not recommendations:
            recommendations.append("Низкий риск: стандартные процедуры")
            
        return recommendations

    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Подготавливает признаки для модели"""
        if not self.feature_names:
            raise ValueError("Модель не обучена или названия признаков не определены")
            
        feature_vector = []
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0)
            if isinstance(value, bool):
                value = int(value)
            elif value is None:
                value = 0
            feature_vector.append(float(value))
            
        return np.array(feature_vector).reshape(1, -1)

    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Оценивает качество модели"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else 0.5
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка оценки модели: {e}")
            return {}

    def _save_model(self) -> None:
        """Сохраняет обученную модель"""
        try:
            model_dir = Path(self.model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Модель сохранена в {self.model_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")

    def _load_model(self) -> None:
        """Загружает сохраненную модель"""
        try:
            if Path(self.model_path).exists():
                model_data = joblib.load(self.model_path)
                
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.is_trained = model_data['is_trained']
                
                logger.info(f"Модель загружена из {self.model_path}")
            else:
                logger.info("Сохраненная модель не найдена")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Возвращает важность признаков"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
            
        try:
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Ошибка получения важности признаков: {e}")
            return {}


class PatientNoShowModel:
    """Основная модель для работы с прогнозированием неявок пациентов"""
    
    def __init__(self, feature_store: FeatureStore, predictor: NoShowPredictor):
        self.feature_store = feature_store
        self.predictor = predictor

    def predict_noshow(self, planning_id: int) -> Optional[Dict[str, Any]]:
        """Предсказывает неявку для planning_id"""
        try:
            # Извлечение признаков
            features = self.feature_store.extract_features(planning_id)
            if not features:
                return None
                
            # Предсказание
            probability = self.predictor.predict(features)
            risk_level = self.predictor.get_risk_level(probability)
            recommendations = self.predictor.get_recommendations(probability, features)
            
            return {
                'planning_id': planning_id,
                'probability': probability,
                'risk_level': risk_level,
                'recommendations': recommendations,
                'features_used': features
            }
            
        except Exception as e:
            logger.error(f"Ошибка предсказания неявки для planning_id {planning_id}: {e}")
            return None

    def train_model(self, days_back: int = 365) -> Dict[str, float]:
        """Обучает модель на исторических данных"""
        try:
            # Загрузка данных
            training_data = self.feature_store.load_training_dataset(days_back)
            
            if training_data.empty:
                logger.warning("Нет данных для обучения")
                return {}
                
            # Обучение
            metrics = self.predictor.train(training_data)
            
            logger.info(f"Модель переобучена на {len(training_data)} записях")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            return {} 