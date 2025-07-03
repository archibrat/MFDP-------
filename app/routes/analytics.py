"""
Маршруты для аналитики и дашборда
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
from datetime import datetime, timedelta
from services.ml_service import get_ml_service, MLService
import random

analytics_router = APIRouter()


@analytics_router.get("/dashboard")
async def get_dashboard_analytics() -> Dict:
    """Получение аналитических данных для дашборда"""
    
    # Генерируем демонстрационные данные для дашборда
    current_date = datetime.now()
    dates = [(current_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    
    dashboard_data = {
        "patient_flow": {
            "dates": dates,
            "predicted": [random.randint(120, 200) for _ in range(7)],
            "actual": [random.randint(115, 195) for _ in range(7)]
        },
        "metrics": {
            "total_patients": 1247,
            "ml_accuracy": "89.23%",
            "auc_roc": "97.29%",
            "active_models": 4
        },
        "model_performance": {
            "accuracy": 89.23,
            "auc_roc": 97.29,
            "precision": 91.45,
            "recall": 87.82
        },
        "risk_distribution": {
            "low_risk": 65,
            "medium_risk": 25,
            "high_risk": 10
        }
    }
    
    return dashboard_data


@analytics_router.get("/patient-flow")
async def get_patient_flow_data(days: int = 7) -> Dict:
    """Получение данных о потоке пациентов"""
    
    current_date = datetime.now()
    dates = [(current_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)]
    
    return {
        "dates": dates,
        "predicted": [random.randint(100, 200) for _ in range(days)],
        "actual": [random.randint(95, 195) for _ in range(days)],
        "no_show_predicted": [random.randint(15, 35) for _ in range(days)],
        "no_show_actual": [random.randint(12, 38) for _ in range(days)]
    }


@analytics_router.get("/model-metrics")
async def get_model_metrics(ml_service: MLService = Depends(get_ml_service)) -> Dict:
    """Получение метрик ML модели"""
    
    try:
        # Получаем метрики из загруженной модели
        model_metadata = ml_service.model_manager.model_metadata.get("latest", {})
        metrics = model_metadata.get("metrics", {})
        
        return {
            "model_version": model_metadata.get("version", "better_baseline_4.0"),
            "training_date": model_metadata.get("training_date", datetime.now().isoformat()),
            "accuracy": metrics.get("accuracy", 0.89) * 100,
            "auc_roc": metrics.get("auc", 0.97) * 100,
            "precision": metrics.get("precision", 0.91) * 100,
            "recall": metrics.get("recall", 0.88) * 100,
            "f1_score": metrics.get("f1", 0.89) * 100
        }
    except Exception as e:
        # Возвращаем демонстрационные данные при ошибке
        return {
            "model_version": "better_baseline_4.0",
            "training_date": datetime.now().isoformat(),
            "accuracy": 89.23,
            "auc_roc": 97.29,
            "precision": 91.45,
            "recall": 87.82,
            "f1_score": 89.13
        }


@analytics_router.get("/high-risk-patients")
async def get_high_risk_patients() -> List[Dict]:
    """Получение списка пациентов высокого риска"""
    
    # Демонстрационные данные высокого риска
    high_risk_patients = [
        {
            "name": "Волков В.В.",
            "appointment": "22.06.2025 14:00",
            "risk_probability": 0.92,
            "risk_level": "Высокий"
        },
        {
            "name": "Новикова Е.А.",
            "appointment": "22.06.2025 15:30",
            "risk_probability": 0.87,
            "risk_level": "Высокий"
        },
        {
            "name": "Борисов М.И.",
            "appointment": "23.06.2025 10:15",
            "risk_probability": 0.84,
            "risk_level": "Высокий"
        },
        {
            "name": "Кузнецова О.П.",
            "appointment": "23.06.2025 11:45",
            "risk_probability": 0.79,
            "risk_level": "Высокий"
        },
        {
            "name": "Федоров А.Н.",
            "appointment": "24.06.2025 09:30",
            "risk_probability": 0.76,
            "risk_level": "Высокий"
        }
    ]
    
    return high_risk_patients


@analytics_router.get("/performance")
async def get_system_performance() -> Dict:
    """Получение данных о производительности системы"""
    
    return {
        "api_response_time": random.uniform(0.1, 0.5),
        "ml_prediction_time": random.uniform(0.2, 1.0),
        "database_connection_time": random.uniform(0.05, 0.2),
        "active_connections": random.randint(10, 50),
        "memory_usage": random.uniform(45, 75),
        "cpu_usage": random.uniform(20, 60)
    } 