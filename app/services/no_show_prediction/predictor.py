"""
Оптимизированный предиктор неявок пациентов
Поддерживает различные алгоритмы ML с автоматическим выбором лучшего
"""

import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from app.services.no_show_prediction.schemas import TrainingMetrics, PredictionResult, RiskLevel
from app.services.no_show_prediction.feature_extractor import FeatureExtractor


class NoShowPredictor:
    """
    Оптимизированный класс для прогнозирования неявок пациентов
    Поддерживает автоматический выбор лучшего алгоритма
    """

    def __init__(self, model_type: str = "auto"):
        """
        Инициализация предиктора
        
        Args:
            model_type: Тип модели ("rf", "gb", "lr", "auto")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = FeatureExtractor.get_feature_names()
        self.is_trained = False
        self.training_metrics = None
        self.logger = logging.getLogger(__name__)

    def train(self, training_data: pd.DataFrame) -> TrainingMetrics:
        """
        Обучение модели на исторических данных
        Автоматически выбирает лучший алгоритм при model_type="auto"
        """
        try:
            if training_data.empty:
                raise ValueError("Нет данных для обучения")

            # Подготовка данных
            features = training_data.drop(['no_show', 'appointment_id'], axis=1, errors='ignore')
            target = training_data['no_show']

            # Проверяем, что все необходимые признаки присутствуют
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                self.logger.warning(f"Отсутствуют признаки: {missing_features}")
                # Добавляем недостающие признаки с нулевыми значениями
                for feature in missing_features:
                    features[feature] = 0

            # Приводим к правильному порядку признаков
            features = features.reindex(columns=self.feature_names, fill_value=0)

            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )

            # Выбор и обучение модели
            if self.model_type == "auto":
                self.model = self._select_best_model(X_train, y_train)
            else:
                self.model = self._create_model(self.model_type)

            # Создание пайплайна с нормализацией
            self.pipeline = Pipeline([
                ('scaler', self.scaler),
                ('classifier', self.model)
            ])

            # Обучение пайплайна
            self.pipeline.fit(X_train, y_train)

            # Оценка качества модели
            self.training_metrics = self._evaluate_model(X_test, y_test, len(X_train), len(X_test))
            
            self.is_trained = True
            self.logger.info(f"Модель обучена успешно. Точность: {self.training_metrics.accuracy:.3f}")

            return self.training_metrics

        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            raise

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Прогнозирование вероятности неявки
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        try:
            # Подготовка признаков
            feature_vector = self._prepare_features(features)
            
            # Прогнозирование
            probability = self.pipeline.predict_proba([feature_vector])[0][1]
            
            return float(probability)

        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования: {e}")
            raise

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """
        Batch-прогнозирование для списка записей
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        try:
            # Подготовка признаков
            feature_matrix = np.array([
                self._prepare_features(features) for features in features_list
            ])
            
            # Batch-прогнозирование
            probabilities = self.pipeline.predict_proba(feature_matrix)[:, 1]
            
            return [float(p) for p in probabilities]

        except Exception as e:
            self.logger.error(f"Ошибка batch-прогнозирования: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}

        return FeatureExtractor.create_feature_importance_plot(
            self.model.feature_importances_, 
            self.feature_names
        )

    def save_model(self, file_path: str) -> None:
        """
        Сохранение модели в файл
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        try:
            model_data = {
                'pipeline': self.pipeline,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'model_type': self.model_type,
                'trained_at': datetime.utcnow()
            }
            
            joblib.dump(model_data, file_path)
            self.logger.info(f"Модель сохранена в {file_path}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")
            raise

    def load_model(self, file_path: str) -> None:
        """
        Загрузка модели из файла
        """
        try:
            model_data = joblib.load(file_path)
            
            self.pipeline = model_data['pipeline']
            self.feature_names = model_data['feature_names']
            self.training_metrics = model_data['training_metrics']
            self.model_type = model_data['model_type']
            self.is_trained = True
            
            self.logger.info(f"Модель загружена из {file_path}")

        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def _select_best_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        """
        Автоматический выбор лучшего алгоритма
        """
        models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }

        best_score = 0
        best_model = None
        best_name = None

        for name, model in models.items():
            try:
                # Кросс-валидация
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                self.logger.warning(f"Ошибка при оценке модели {name}: {e}")

        if best_model is None:
            self.logger.warning("Не удалось выбрать лучшую модель, используем RandomForest")
            best_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            best_name = 'rf'

        self.logger.info(f"Выбрана модель: {best_name} (ROC-AUC: {best_score:.3f})")
        return best_model

    def _create_model(self, model_type: str) -> BaseEstimator:
        """
        Создание модели указанного типа
        """
        models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        if model_type not in models:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
            
        return models[model_type]

    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                       train_samples: int, test_samples: int) -> TrainingMetrics:
        """
        Оценка качества модели
        """
        predictions = self.pipeline.predict(X_test)
        probabilities = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Метрики
        report = classification_report(y_test, predictions, output_dict=True)
        
        return TrainingMetrics(
            accuracy=float(report['accuracy']),
            precision=float(report['1']['precision']),
            recall=float(report['1']['recall']),
            f1_score=float(report['1']['f1-score']),
            training_samples=train_samples,
            test_samples=test_samples
        )

    def _prepare_features(self, features: Dict[str, Any]) -> List[float]:
        """
        Подготовка признаков для прогнозирования
        """
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0))
        return feature_vector 