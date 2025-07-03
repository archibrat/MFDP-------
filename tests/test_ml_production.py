"""
Тесты для продакшн ML-модуля
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from app.services.ml_production.dal import MLDataAccessLayer
from app.services.ml_production.feature_store import FeatureStore
from app.services.ml_production.no_show_predictor import NoShowPredictor, PatientNoShowModel
from app.services.ml_production.load_balancer import DoctorLoadOptimizer
from app.services.ml_production.real_time_scheduler import RealTimeScheduler
from app.models.ml_production import RiskLevel, EventType


class TestFeatureStore:
    """Тесты Feature Store"""
    
    def test_extract_features(self):
        """Тест извлечения признаков"""
        mock_dal = Mock()
        mock_dal.extract_noshow_features.return_value = {
            'planning_id': 123,
            'weekday': 1,
            'hour': 14,
            'month': 6,
            'is_weekend': False,
            'patient_age': 45,
            'patient_gender': 'M',
            'advance_booking_days': 7,
            'doctor_specialization_id': 1,
            'department_id': 1,
            'not_send_sms': False,
            'has_email': True,
            'past_noshows_count': 2
        }
        
        feature_store = FeatureStore(mock_dal)
        features = feature_store.extract_features(123)
        
        assert features is not None
        assert features['planning_id'] == 123
        assert 'is_morning' in features
        assert 'contact_score' in features
        assert features['contact_score'] == 2  # has_email + not not_send_sms
        
        mock_dal.save_features.assert_called_once()

    def test_load_training_dataset(self):
        """Тест загрузки обучающего датасета"""
        mock_dal = Mock()
        mock_data = pd.DataFrame({
            'planning_id': [1, 2, 3],
            'weekday': [1, 2, 3],
            'hour': [9, 14, 16],
            'patient_age': [25, 45, 65],
            'advance_booking_days': [1, 7, 14],
            'target': [0, 1, 0]
        })
        mock_dal.load_training_data.return_value = mock_data
        
        feature_store = FeatureStore(mock_dal)
        dataset = feature_store.load_training_dataset(365)
        
        assert not dataset.empty
        assert 'is_morning' in dataset.columns
        assert 'is_elderly' in dataset.columns
        assert len(dataset) == 3


class TestNoShowPredictor:
    """Тесты No-Show Predictor"""
    
    def test_predict_without_model(self):
        """Тест предсказания без обученной модели"""
        predictor = NoShowPredictor()
        features = {'weekday': 1, 'hour': 14}
        
        probability = predictor.predict(features)
        
        assert 0.0 <= probability <= 1.0
        assert probability == 0.3  # Значение по умолчанию

    def test_get_risk_level(self):
        """Тест определения уровня риска"""
        predictor = NoShowPredictor()
        
        assert predictor.get_risk_level(0.2) == RiskLevel.LOW
        assert predictor.get_risk_level(0.5) == RiskLevel.MEDIUM
        assert predictor.get_risk_level(0.8) == RiskLevel.HIGH

    def test_get_recommendations(self):
        """Тест генерации рекомендаций"""
        predictor = NoShowPredictor()
        features = {
            'not_send_sms': True,
            'past_noshows_count': 3,
            'advance_booking_days': 45
        }
        
        recommendations = predictor.get_recommendations(0.9, features)
        
        assert len(recommendations) > 0
        assert any('двойное бронирование' in rec for rec in recommendations)
        assert any('звонок' in rec for rec in recommendations)

    @patch('app.services.ml_production.no_show_predictor.xgb')
    def test_train_model(self, mock_xgb):
        """Тест обучения модели"""
        # Мокаем XGBoost
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])
        mock_model.predict.return_value = np.array([0, 1])
        mock_xgb.XGBClassifier.return_value = mock_model
        
        predictor = NoShowPredictor()
        
        # Создаем тестовые данные
        training_data = pd.DataFrame({
            'weekday': [1, 2, 3, 4, 1, 2],
            'hour': [9, 14, 16, 10, 11, 15],
            'patient_age': [25, 45, 65, 30, 55, 40],
            'target': [0, 1, 0, 1, 0, 1]
        })
        
        metrics = predictor.train(training_data)
        
        assert predictor.is_trained
        assert 'accuracy' in metrics
        mock_model.fit.assert_called_once()


class TestPatientNoShowModel:
    """Тесты основной модели неявок"""
    
    def test_predict_noshow(self):
        """Тест предсказания неявки"""
        mock_feature_store = Mock()
        mock_feature_store.extract_features.return_value = {
            'planning_id': 123,
            'weekday': 1,
            'hour': 14
        }
        
        mock_predictor = Mock()
        mock_predictor.predict.return_value = 0.7
        mock_predictor.get_risk_level.return_value = RiskLevel.HIGH
        mock_predictor.get_recommendations.return_value = ['Отправить SMS']
        
        model = PatientNoShowModel(mock_feature_store, mock_predictor)
        result = model.predict_noshow(123)
        
        assert result is not None
        assert result['planning_id'] == 123
        assert result['probability'] == 0.7
        assert result['risk_level'] == RiskLevel.HIGH


class TestDoctorLoadOptimizer:
    """Тесты оптимизатора нагрузки"""
    
    def test_optimize_load_distribution(self):
        """Тест оптимизации распределения нагрузки"""
        mock_dal = Mock()
        mock_dal.get_doctor_profiles.return_value = [
            {
                'medecins_id': 1,
                'specialization_id': 1,
                'department_id': 1,
                'daily_norm': 20,
                'current_load': 0.9,
                'avg_consultation_time': 30
            },
            {
                'medecins_id': 2,
                'specialization_id': 1,
                'department_id': 1,
                'daily_norm': 20,
                'current_load': 0.3,
                'avg_consultation_time': 25
            }
        ]
        
        mock_dal.get_schedule_data.return_value = [
            {
                'planning_id': 1,
                'medecins_id': 1,
                'status': 'scheduled',
                'cito': 'N',
                'vip_groups_id': None
            }
        ]
        
        optimizer = DoctorLoadOptimizer(mock_dal)
        result = optimizer.optimize_load_distribution(
            datetime.utcnow(),
            datetime.utcnow() + timedelta(days=1)
        )
        
        assert 'optimized_assignments' in result
        assert 'metrics' in result
        assert 'recommendations' in result

    def test_generate_nightly_report(self):
        """Тест генерации ночного отчета"""
        mock_dal = Mock()
        mock_dal.get_doctor_profiles.return_value = []
        mock_dal.get_schedule_data.return_value = []
        
        optimizer = DoctorLoadOptimizer(mock_dal)
        
        # Не должно вызывать исключений
        optimizer.generate_nightly_report([1, 2, 3])
        
        mock_dal.save_load_stats.assert_called()


class TestRealTimeScheduler:
    """Тесты планировщика реального времени"""
    
    def test_handle_patient_arrival(self):
        """Тест обработки прибытия пациента"""
        mock_dal = Mock()
        mock_dal.get_waiting_patients.return_value = []
        
        scheduler = RealTimeScheduler(mock_dal)
        
        result = scheduler.handle_event(EventType.ARRIVE_DATE, {
            'planning_id': 123,
            'doctor_id': 1,
            'arrival_time': datetime.utcnow()
        })
        
        assert 'planning_id' in result
        assert result['planning_id'] == 123

    def test_handle_cancellation(self):
        """Тест обработки отмены"""
        mock_dal = Mock()
        mock_dal.get_waiting_patients.return_value = [
            {
                'planning_id': 124,
                'priority': 8,
                'heure': datetime.utcnow() + timedelta(hours=1)
            }
        ]
        
        scheduler = RealTimeScheduler(mock_dal)
        
        result = scheduler.handle_event(EventType.CANCELLED, {
            'planning_id': 123,
            'doctor_id': 1,
            'scheduled_time': datetime.utcnow()
        })
        
        assert result['status'] == 'cancelled'
        assert 'rescheduled_patients' in result

    def test_optimize_waiting_queue(self):
        """Тест оптимизации очереди ожидания"""
        mock_dal = Mock()
        mock_dal.get_waiting_patients.return_value = [
            {
                'planning_id': 1,
                'doctor_id': 1,
                'priority': 5,
                'heure': datetime.utcnow()
            },
            {
                'planning_id': 2,
                'doctor_id': 1,
                'priority': 10,
                'heure': datetime.utcnow()
            }
        ]
        
        scheduler = RealTimeScheduler(mock_dal)
        result = scheduler.optimize_waiting_queue(1)
        
        assert 'reordered' in result
        assert 'queue' in result
        assert len(result['queue']) == 2


@pytest.fixture
def sample_features():
    """Фикстура с примером признаков"""
    return {
        'planning_id': 123,
        'weekday': 1,
        'hour': 14,
        'month': 6,
        'is_weekend': False,
        'patient_age': 45,
        'advance_booking_days': 7,
        'past_noshows_count': 1
    }


@pytest.fixture
def sample_training_data():
    """Фикстура с обучающими данными"""
    return pd.DataFrame({
        'planning_id': range(100),
        'weekday': np.random.randint(0, 7, 100),
        'hour': np.random.randint(8, 18, 100),
        'patient_age': np.random.randint(18, 80, 100),
        'advance_booking_days': np.random.randint(0, 30, 100),
        'target': np.random.randint(0, 2, 100)
    })


class TestIntegration:
    """Интеграционные тесты"""
    
    def test_full_noshow_pipeline(self, sample_features):
        """Тест полного пайплайна прогнозирования неявок"""
        # Мокаем DAL
        mock_dal = Mock()
        mock_dal.extract_noshow_features.return_value = sample_features
        mock_dal.save_features.return_value = None
        
        # Создаем компоненты
        feature_store = FeatureStore(mock_dal)
        predictor = NoShowPredictor()
        model = PatientNoShowModel(feature_store, predictor)
        
        # Тестируем пайплайн
        result = model.predict_noshow(123)
        
        assert result is not None
        assert 'probability' in result
        assert 'risk_level' in result
        assert 'recommendations' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 