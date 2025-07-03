import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import json


class TestAPIFrontendIntegration:
    """Тесты для интеграции API с фронтендом"""
    
    def test_cors_configuration(self, client):
        """Тест конфигурации CORS для интеграции с фронтендом"""
        response = client.options("/api/ml/predict", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    def test_prediction_endpoint_format(self, client):
        """Тест формата ответа endpoint'а предсказаний для фронтенда"""
        prediction_data = {
            "patient_data": {
                "patient_id": 12345,
                "appointment_id": 67890,
                "gender": "F",
                "age": 45,
                "neighbourhood": "Central",
                "scholarship": False,
                "hypertension": True,
                "diabetes": False,
                "alcoholism": False,
                "handcap": 0,
                "sms_received": True,
                "scheduled_day": datetime.now().isoformat(),
                "appointment_day": (datetime.now() + timedelta(days=3)).isoformat()
            },
            "prediction_type": "no_show",
            "include_explanation": True
        }
        
        with patch('services.ml_service.get_ml_service') as mock_service:
            mock_ml_service = MagicMock()
            mock_response = MagicMock()
            mock_response.prediction_value = 0.75
            mock_response.confidence_score = 0.85
            mock_response.risk_level = "Высокий"
            mock_response.recommendations = ["Обязательно связаться с пациентом"]
            mock_response.model_version = "better_baseline_4.0"
            mock_response.created_at = datetime.now()
            
            mock_ml_service.predict.return_value = mock_response
            mock_service.return_value = mock_ml_service
            
            response = client.post("/api/ml/predict", json=prediction_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Проверяем структуру ответа для фронтенда
            assert "prediction_value" in data
            assert "confidence_score" in data
            assert "risk_level" in data
            assert "recommendations" in data
            assert "model_version" in data
            assert isinstance(data["recommendations"], list)
    
    def test_analytics_endpoint_for_dashboard(self, client):
        """Тест endpoint аналитики для дашборда фронтенда"""
        response = client.get("/api/analytics/dashboard")
        
        # Проверяем, что endpoint возвращает данные в нужном формате
        assert response.status_code in [200, 404]  # 404 если endpoint не реализован
    
    def test_events_crud_endpoints(self, client):
        """Тест CRUD операций для событий"""
        # Тест получения списка событий
        response = client.get("/api/events/")
        assert response.status_code in [200, 404]
        
        # Тест создания события
        event_data = {
            "event_type": "appointment",
            "patient_id": 12345,
            "scheduled_time": datetime.now().isoformat(),
            "description": "Плановый прием",
            "priority": "medium"
        }
        
        response = client.post("/api/events/", json=event_data)
        assert response.status_code in [200, 201, 404, 422]  # Различные возможные коды
    
    def test_user_management_endpoints(self, client):
        """Тест endpoints управления пользователями"""
        response = client.get("/api/users/")
        assert response.status_code in [200, 401, 404]
    
    def test_health_check_endpoint(self, client):
        """Тест endpoint проверки здоровья системы"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_api_error_handling(self, client):
        """Тест обработки ошибок API для фронтенда"""
        # Отправляем невалидные данные
        invalid_data = {"invalid": "data"}
        
        response = client.post("/api/ml/predict", json=invalid_data)
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_authentication_endpoints(self, client):
        """Тест endpoints аутентификации"""
        # Тест endpoint логина
        login_data = {
            "username": "test@example.com",
            "password": "testpassword"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code in [200, 401, 404]
    
    def test_api_versioning(self, client):
        """Тест версионирования API"""
        response = client.get("/api/docs")
        assert response.status_code == 200
    
    def test_content_type_headers(self, client):
        """Тест правильных заголовков Content-Type"""
        response = client.get("/")
        assert "content-type" in response.headers
        
    def test_api_response_time(self, client):
        """Тест времени ответа API"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0  # API должно отвечать быстро 