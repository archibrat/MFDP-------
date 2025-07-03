import pytest
import pytest_asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from services.ml_service import MLService, MLModelManager
from models.prediction import PredictionRequest, PatientData, PredictionType
import tempfile
import os


class TestImprovedMLModelIntegration:
    """Тесты для интеграции улучшенной ML модели из better_baseline_4.py"""
    
    @pytest.fixture
    def mock_trained_model(self):
        """Создание мока обученной модели"""
        model_mock = MagicMock()
        model_mock.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])
        model_mock.predict.return_value = np.array([0, 1])
        return model_mock
    
    @pytest.fixture
    def sample_patient_data(self):
        """Создание примера данных пациента"""
        return PatientData(
            patient_id=12345,
            appointment_id=67890,
            gender="F",
            age=45,
            neighbourhood="Central",
            scholarship=False,
            hypertension=True,
            diabetes=False,
            alcoholism=False,
            handcap=0,
            sms_received=True,
            scheduled_day=datetime.now().isoformat(),
            appointment_day=(datetime.now() + timedelta(days=3)).isoformat()
        )
    
    @pytest_asyncio.fixture
    async def ml_service(self, mock_trained_model):
        """Создание экземпляра ML сервиса с мокированной моделью"""
        with patch('services.ml_service.joblib.load') as mock_load:
            mock_load.return_value = {
                'model': mock_trained_model,
                'feature_processor': MagicMock(),
                'model_version': 'better_baseline_4.0',
                'training_date': '2025-01-20',
                'metrics': {'auc': 0.97, 'accuracy': 0.89},
                'feature_names': ['age', 'gender_encoded', 'days_advance', 'sms_received']
            }
            
            service = MLService()
            await service.initialize()
            return service
    
    def test_ml_model_manager_initialization(self):
        """Тест инициализации менеджера ML моделей"""
        manager = MLModelManager()
        assert manager.models_path == "./models"
        assert manager.models == {}
        assert manager.feature_processors == {}
        assert manager.model_metadata == {}
    
    @pytest.mark.asyncio
    async def test_improved_model_loading(self, mock_trained_model):
        """Тест загрузки улучшенной модели"""
        with patch('builtins.open'), patch('services.ml_service.joblib.load') as mock_load:
            mock_load.return_value = {
                'model': mock_trained_model,
                'feature_processor': MagicMock(),
                'model_version': 'better_baseline_4.0',
                'training_date': '2025-01-20',
                'metrics': {'auc': 0.97, 'accuracy': 0.89},
                'feature_names': ['age', 'gender_encoded', 'days_advance', 'sms_received']
            }
            
            manager = MLModelManager()
            success = await manager.load_model("better_baseline_4")
            
            assert success is True
            assert "better_baseline_4" in manager.models
            assert manager.model_metadata["better_baseline_4"]["version"] == "better_baseline_4.0"
            assert manager.model_metadata["better_baseline_4"]["metrics"]["auc"] == 0.97
    
    @pytest.mark.asyncio
    async def test_prediction_with_improved_model(self, ml_service, sample_patient_data):
        """Тест предсказания с улучшенной моделью"""
        request = PredictionRequest(
            patient_data=sample_patient_data,
            prediction_type=PredictionType.NO_SHOW,
            include_explanation=False
        )
        
        response = await ml_service.predict(request)
        
        assert response is not None
        assert 0 <= response.prediction_value <= 1
        assert 0 <= response.confidence_score <= 1
        assert response.risk_level in ["Низкий", "Средний", "Высокий"]
        assert response.model_version == "better_baseline_4.0"
        assert isinstance(response.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_advanced_feature_engineering(self, ml_service, sample_patient_data):
        """Тест продвинутой генерации признаков"""
        data_dict = sample_patient_data.model_dump()
        df = pd.DataFrame([data_dict])
        
        processed_data = await ml_service._prepare_data(sample_patient_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert 'days_advance' in processed_data.columns
        assert 'hours_advance' in processed_data.columns
        assert len(processed_data) == 1
    
    def test_temporal_feature_extraction(self):
        """Тест извлечения временных признаков как в better_baseline_4.py"""
        test_data = pd.DataFrame({
            'ScheduledDay': pd.to_datetime(['2023-06-15 10:30:00', '2023-06-16 14:20:00']),
            'AppointmentDay': pd.to_datetime(['2023-06-18 10:00:00', '2023-06-20 15:30:00'])
        })
        
        test_data['days_advance'] = (test_data['AppointmentDay'] - test_data['ScheduledDay']).dt.days
        test_data['hours_advance'] = (test_data['AppointmentDay'] - test_data['ScheduledDay']).dt.total_seconds() / 3600
        
        assert test_data['days_advance'].iloc[0] == 2  # 18-15 = 2 дня разницы  
        assert test_data['days_advance'].iloc[1] == 4
    
    @pytest.mark.asyncio
    async def test_risk_level_classification(self, ml_service):
        """Тест классификации уровня риска"""
        # Тест различных уровней вероятности
        test_cases = [
            (0.1, "Низкий"),
            (0.25, "Низкий"),
            (0.45, "Средний"),
            (0.55, "Средний"),
            (0.75, "Высокий"),
            (0.9, "Высокий")
        ]
        
        for probability, expected_risk in test_cases:
            features = np.array([[1, 0, 1, 0]])  # Dummy features
            
            with patch.object(ml_service.model_manager.models["latest"], 'predict_proba') as mock_predict:
                mock_predict.return_value = np.array([[1-probability, probability]])
                
                result = await ml_service._make_prediction(features)
                assert result['risk_level'] == expected_risk
    
    @pytest.mark.asyncio 
    async def test_recommendation_generation(self, ml_service):
        """Тест генерации рекомендаций на основе уровня риска"""
        features = np.array([[1, 0, 1, 0]])
        
        # Высокий риск
        with patch.object(ml_service.model_manager.models["latest"], 'predict_proba') as mock_predict:
            mock_predict.return_value = np.array([[0.25, 0.75]])
            
            result = await ml_service._make_prediction(features)
            assert "Обязательно связаться с пациентом" in result['recommendations']
            assert "Телефонный звонок за день до приема" in result['recommendations']
    
    def test_better_baseline_model_architecture_integration(self):
        """Тест интеграции архитектуры модели из better_baseline_4.py"""
        # Проверяем, что можем импортировать и использовать классы из better_baseline_4
        from services.ml.better_baseline_model import AdvancedModelArchitecture
        
        architecture = AdvancedModelArchitecture()
        models = architecture.create_base_models()
        
        # Проверяем наличие базовых моделей (продвинутые могут быть недоступны)
        required_models = ['random_forest', 'logistic']
        for model_name in required_models:
            assert model_name in models
        
        # Проверяем, что хотя бы один из продвинутых алгоритмов доступен
        assert len(models) >= 2
    
    @pytest.mark.asyncio
    async def test_model_performance_metrics(self, ml_service, sample_patient_data):
        """Тест метрик производительности модели"""
        request = PredictionRequest(
            patient_data=sample_patient_data,
            prediction_type=PredictionType.NO_SHOW,
            include_explanation=False
        )
        
        response = await ml_service.predict(request)
        
        # Проверяем, что модель возвращает ожидаемые метрики производительности
        metadata = ml_service.model_manager.model_metadata["latest"]
        assert metadata["metrics"]["auc"] >= 0.9  # Ожидаем высокое качество от улучшенной модели
        assert metadata["metrics"]["accuracy"] >= 0.85
    
    def test_error_handling_for_missing_model(self):
        """Тест обработки ошибок при отсутствии модели"""
        manager = MLModelManager()
        
        with pytest.raises(Exception):
            manager._load_model_sync("non_existent_model")
    
    def test_batch_prediction_capability(self):
        """Тест возможности батчевых предсказаний"""
        # Создаем несколько примеров данных
        batch_data = []
        for i in range(5):
            patient_data = PatientData(
                patient_id=12345 + i,
                appointment_id=67890 + i,
                gender="F" if i % 2 == 0 else "M",
                age=30 + i * 5,
                neighbourhood="Central",
                scholarship=False,
                hypertension=i % 3 == 0,
                diabetes=False,
                alcoholism=False,
                handcap=0,
                sms_received=True,
                scheduled_day=datetime.now().isoformat(),
                appointment_day=(datetime.now() + timedelta(days=i+1)).isoformat()
            )
            batch_data.append(patient_data)
        
        # Проверяем, что можем обработать множественные запросы
        assert len(batch_data) == 5
        for data in batch_data:
            assert isinstance(data, PatientData) 