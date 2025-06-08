import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

from main import app
from database.database import get_session
from services.ml_service import get_ml_service
from tests.conftest import override_get_session, override_get_ml_service

@pytest.fixture
def test_patient_data():
    """Тестовые данные пациента"""
    return {
        "client_id": "TEST_001",
        "booking_id": "BOOK_TEST_001",
        "age": 45,
        "gender": "f",
        "district": "test_district",
        "scholarship": False,
        "condition_a": True,
        "condition_b": False,
        "condition_c": False,
        "accessibility_level": 0,
        "notification_sent": True,
        "planned_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
        "session_date": (datetime.utcnow() + timedelta(days=1)).isoformat()
    }

class TestPredictionAPI:
    """Тесты API предсказаний"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Настройка для каждого теста"""
        app.dependency_overrides[get_session] = override_get_session
        app.dependency_overrides[get_ml_service] = override_get_ml_service
        self.client = TestClient(app)
    
    def test_single_prediction_success(self, test_patient_data):
        """Тест успешного единичного предсказания"""
        request_data = {
            "patient_data": test_patient_data,
            "prediction_types": ["no_show_risk"],
            "include_explanation": False
        }
        
        response = self.client.post("/api/predictions/predict", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "prediction_id" in result
        assert "prediction_value" in result
        assert "confidence_score" in result
        assert "risk_level" in result
        assert "recommendations" in result
        assert "model_version" in result
        
        assert 0 <= result["prediction_value"] <= 1
        assert 0 <= result["confidence_score"] <= 1
        assert result["risk_level"] in ["Низкий", "Средний", "Высокий"]
        assert isinstance(result["recommendations"], list)
    
    def test_prediction_with_explanation(self, test_patient_data):
        """Тест предсказания с объяснением"""
        request_data = {
            "patient_data": test_patient_data,
            "prediction_types": ["no_show_risk"],
            "include_explanation": True
        }
        
        response = self.client.post("/api/predictions/predict", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "explanation" in result
        if result["explanation"]:
            assert isinstance(result["explanation"], dict)
    
    def test_invalid_patient_data(self):
        """Тест с некорректными данными пациента"""
        invalid_data = {
            "patient_data": {
                "age": -5,  # Некорректный возраст
                "gender": "invalid"  # Некорректный пол
            },
            "prediction_types": ["no_show_risk"]
        }
        
        response = self.client.post("/api/predictions/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction(self, test_patient_data):
        """Тест пакетного предсказания"""
        batch_data = {
            "patients_data": [test_patient_data, test_patient_data],
            "prediction_type": "no_show_risk"
        }
        
        response = self.client.post("/api/predictions/predict/batch", json=batch_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "task_id" in result
        assert "status" in result
        assert result["status"] == "started"
        assert "total_patients" in result
        assert result["total_patients"] == 2
    
    def test_model_status(self):
        """Тест получения статуса моделей"""
        response = self.client.get("/api/predictions/models/status")
        
        assert response.status_code == 200
        result = response.json()
        
        assert isinstance(result, list)
        if result:
            model = result[0]
            assert "model_version" in model
            assert "status" in model
            assert "metrics" in model
    
    def test_prediction_unauthorized(self, test_patient_data):
        """Тест доступа без авторизации"""
        # Убираем переопределение аутентификации
        if hasattr(app, 'dependency_overrides'):
            app.dependency_overrides.clear()
        
        request_data = {
            "patient_data": test_patient_data,
            "prediction_types": ["no_show_risk"]
        }
        
        response = self.client.post("/api/predictions/predict", json=request_data)
        assert response.status_code == 401

class TestMLService:
    """Тесты ML-сервиса"""
    
    @pytest.mark.asyncio
    async def test_ml_service_initialization(self):
        """Тест инициализации ML-сервиса"""
        from services.ml_service import MLService
        
        ml_service = MLService()
        success = await ml_service.initialize()
        
        assert success is True
        assert ml_service.model_manager.models
        assert "latest" in ml_service.model_manager.models
    
    @pytest.mark.asyncio
    async def test_prediction_generation(self, test_patient_data):
        """Тест генерации предсказания"""
        from services.ml_service import MLService
        from schemas.prediction import PredictionRequest, PatientDataCreate
        
        ml_service = MLService()
        await ml_service.initialize()
        
        patient_data = PatientDataCreate(**test_patient_data)
        request = PredictionRequest(
            patient_data=patient_data,
            prediction_types=["no_show_risk"]
        )
        
        result = await ml_service.predict(request)
        
        assert hasattr(result, 'prediction_value')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'risk_level')
        assert hasattr(result, 'recommendations')
        
        assert 0 <= result.prediction_value <= 1
        assert 0 <= result.confidence_score <= 1
    
    @pytest.mark.asyncio
    async def test_feature_engineering(self, test_patient_data):
        """Тест создания признаков"""
        from services.ml_service import MLService
        
        ml_service = MLService()
        await ml_service.initialize()
        
        # Преобразование в DataFrame
        import pandas as pd
        df = pd.DataFrame([test_patient_data])
        
        # Тест подготовки данных
        processed_data = await ml_service._prepare_data(test_patient_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert not processed_data.empty
        assert 'days_advance' in processed_data.columns

class TestPredictionDatabase:
    """Тесты базы данных предсказаний"""
    
    def test_prediction_model_creation(self):
        """Тест создания модели предсказания"""
        from models.prediction import PredictionResult, PredictionType, PredictionStatus
        
        prediction = PredictionResult(
            task_id=1,
            patient_data_id=1,
            user_id=1,
            prediction_type=PredictionType.NO_SHOW_RISK,
            prediction_value=0.75,
            confidence_score=0.85,
            risk_level="Высокий",
            model_version="v2.0",
            features_used='{"age": 45, "conditions": 1}',
            status=PredictionStatus.COMPLETED
        )
        
        assert prediction.prediction_type == PredictionType.NO_SHOW_RISK
        assert prediction.prediction_value == 0.75
        assert prediction.confidence_score == 0.85
        assert prediction.risk_level == "Высокий"
    
    def test_patient_data_validation(self):
        """Тест валидации данных пациента"""
        from models.prediction import PatientData
        from datetime import datetime
        
        # Корректные данные
        valid_data = PatientData(
            client_id="TEST_001",
            booking_id="BOOK_001",
            age=45,
            gender="f",
            district="test",
            planned_date=datetime.utcnow(),
            session_date=datetime.utcnow()
        )
        
        assert valid_data.age == 45
        assert valid_data.gender == "f"
        
        # Некорректный возраст
        with pytest.raises(ValueError):
            PatientData(
                client_id="TEST_002",
                booking_id="BOOK_002",
                age=-5,  # Некорректно
                gender="f",
                district="test",
                planned_date=datetime.utcnow(),
                session_date=datetime.utcnow()
            )

# tests/test_ml_integration.py
class TestMLIntegration:
    """Интеграционные тесты ML-компонентов"""
    
    @pytest.mark.asyncio
    async def test_full_prediction_flow(self, test_patient_data):
        """Тест полного потока предсказания"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Создание предсказания
            response = await ac.post("/api/predictions/predict", json={
                "patient_data": test_patient_data,
                "prediction_types": ["no_show_risk"],
                "include_explanation": True
            })
            
            assert response.status_code == 200
            prediction_result = response.json()
            prediction_id = prediction_result["prediction_id"]
            
            # Получение истории
            patient_id = test_patient_data["client_id"]
            history_response = await ac.get(f"/api/predictions/history/{patient_id}")
            
            assert history_response.status_code == 200
            history = history_response.json()
            
            assert len(history) > 0
            assert any(p["id"] == prediction_id for p in history)
    
    @pytest.mark.asyncio
    async def test_model_performance_monitoring(self):
        """Тест мониторинга производительности модели"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Получение аналитики
            response = await ac.get("/api/predictions/analytics/performance?days=7")
            
            assert response.status_code == 200
            analytics = response.json()
            
            assert "accuracy" in analytics
            assert "f1_score" in analytics
            assert "dates" in analytics
            
            if analytics["accuracy"]:
                assert all(0 <= acc <= 1 for acc in analytics["accuracy"])
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, test_patient_data):
        """Тест конкурентных предсказаний"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Создание множественных запросов
            tasks = []
            for i in range(10):
                patient_data = test_patient_data.copy()
                patient_data["client_id"] = f"TEST_{i:03d}"
                patient_data["booking_id"] = f"BOOK_{i:03d}"
                
                task = ac.post("/api/predictions/predict", json={
                    "patient_data": patient_data,
                    "prediction_types": ["no_show_risk"]
                })
                tasks.append(task)
            
            # Выполнение всех запросов
            responses = await asyncio.gather(*tasks)
            
            # Проверка результатов
            assert all(r.status_code == 200 for r in responses)
            results = [r.json() for r in responses]
            
            # Проверка уникальности prediction_id
            prediction_ids = [r["prediction_id"] for r in results]
            assert len(set(prediction_ids)) == len(prediction_ids)

# Конфигурация для pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
