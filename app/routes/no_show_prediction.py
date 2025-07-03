"""
Роутер для прогнозирования неявок пациентов
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.models.base import BaseResponse
from app.database.database import get_session
from app.repositories.medialog_repository import MedialogRepository
from app.services.no_show_prediction_service import MedialogDatabaseConnector, NoShowPredictionService
from app.services.no_show_prediction.schemas import (
    BatchPredictionRequest, BatchPredictionResponse, TrainingMetrics
)

router = APIRouter(prefix="/no-show-prediction", tags=["no-show-prediction"])


def get_prediction_service(session: Session = Depends(get_session)) -> NoShowPredictionService:
    """Получение сервиса прогнозирования"""
    db_connector = MedialogDatabaseConnector(session)
    return NoShowPredictionService(db_connector)


def get_medialog_repository(session: Session = Depends(get_session)) -> MedialogRepository:
    """Получение репозитория медиалог"""
    return MedialogRepository(session)


@router.post("/train", response_model=TrainingMetrics)
async def train_model(
    days_back: int = Query(default=365, ge=30, le=1095, description="Количество дней для обучения"),
    prediction_service: NoShowPredictionService = Depends(get_prediction_service)
) -> TrainingMetrics:
    """
    Обучение модели прогнозирования неявок
    
    Args:
        days_back: Количество дней для обучения
        prediction_service: Сервис прогнозирования
        
    Returns:
        Метрики качества обученной модели
    """
    try:
        metrics = prediction_service.train_model(days_back)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения модели: {str(e)}")


@router.post("/predict/{appointment_id}", response_model=BaseResponse)
async def predict_no_show(
    appointment_id: int,
    save_prediction: bool = Query(default=True, description="Сохранить прогноз в БД"),
    prediction_service: NoShowPredictionService = Depends(get_prediction_service)
) -> BaseResponse:
    """
    Прогнозирование неявки для конкретной записи
    
    Args:
        appointment_id: ID записи на прием
        save_prediction: Сохранить прогноз в БД
        prediction_service: Сервис прогнозирования
        
    Returns:
        Результат прогнозирования
    """
    try:
        result = prediction_service.predict_no_show(appointment_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Запись {appointment_id} не найдена")
        
        if save_prediction:
            prediction_service.save_prediction_to_db(result)
        
        return BaseResponse(
            success=True,
            message="Прогноз выполнен успешно",
            data={
                "appointment_id": result.appointment_id,
                "no_show_probability": result.no_show_probability,
                "risk_level": result.risk_level.value,
                "recommendation": result.recommendation,
                "confidence": result.confidence
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прогнозирования: {str(e)}")


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_no_show(
    request: BatchPredictionRequest,
    prediction_service: NoShowPredictionService = Depends(get_prediction_service)
) -> BatchPredictionResponse:
    """
    Массовое прогнозирование неявок
    
    Args:
        request: Запрос на массовое прогнозирование
        prediction_service: Сервис прогнозирования
        
    Returns:
        Результаты массового прогнозирования
    """
    try:
        result = prediction_service.batch_predict(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка массового прогнозирования: {str(e)}")


@router.get("/predictions/{appointment_id}", response_model=BaseResponse)
async def get_predictions_for_appointment(
    appointment_id: int,
    repository: MedialogRepository = Depends(get_medialog_repository)
) -> BaseResponse:
    """
    Получение прогнозов для записи
    
    Args:
        appointment_id: ID записи
        repository: Репозиторий медиалог
        
    Returns:
        Список прогнозов для записи
    """
    try:
        predictions = repository.get_predictions_by_appointment(appointment_id)
        
        prediction_data = []
        for pred in predictions:
            prediction_data.append({
                "prediction_id": pred.id,
                "appointment_id": pred.appointment_id,
                "probability": pred.prediction_probability,
                "risk_level": pred.risk_level,
                "recommendations": pred.recommendations,
                "model_version": pred.model_version,
                "confidence": pred.confidence_score,
                "prediction_date": pred.prediction_date.isoformat()
            })
        
        return BaseResponse(
            success=True,
            message=f"Найдено {len(predictions)} прогнозов для записи {appointment_id}",
            data=prediction_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения прогнозов: {str(e)}")


@router.get("/statistics", response_model=BaseResponse)
async def get_no_show_statistics(
    start_date: datetime = Query(description="Начальная дата"),
    end_date: datetime = Query(description="Конечная дата"),
    group_by: str = Query(default="day", regex="^(day|week|month)$", description="Группировка"),
    repository: MedialogRepository = Depends(get_medialog_repository)
) -> BaseResponse:
    """
    Получение статистики неявок
    
    Args:
        start_date: Начальная дата
        end_date: Конечная дата
        group_by: Тип группировки
        repository: Репозиторий медиалог
        
    Returns:
        Статистика неявок
    """
    try:
        statistics = repository.get_no_show_statistics(start_date, end_date, group_by)
        
        return BaseResponse(
            success=True,
            message=f"Статистика неявок за период с {start_date.date()} по {end_date.date()}",
            data={
                "period": f"{start_date.date()} - {end_date.date()}",
                "group_by": group_by,
                "statistics": statistics,
                "total_records": len(statistics)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")


@router.get("/risk-profiles", response_model=BaseResponse)
async def get_patient_risk_profiles(
    limit: int = Query(default=50, ge=1, le=200, description="Количество профилей"),
    repository: MedialogRepository = Depends(get_medialog_repository)
) -> BaseResponse:
    """
    Получение профилей риска пациентов
    
    Args:
        limit: Количество профилей
        repository: Репозиторий медиалог
        
    Returns:
        Профили риска пациентов
    """
    try:
        risk_profiles = repository.get_patient_risk_profiles(limit)
        
        # Расчет распределения рисков
        risk_distribution = {
            "low": len([p for p in risk_profiles if p["risk_level"] == "low"]),
            "medium": len([p for p in risk_profiles if p["risk_level"] == "medium"]),
            "high": len([p for p in risk_profiles if p["risk_level"] == "high"])
        }
        
        return BaseResponse(
            success=True,
            message=f"Получено {len(risk_profiles)} профилей риска",
            data={
                "risk_profiles": risk_profiles,
                "total_count": len(risk_profiles),
                "risk_distribution": {
                    "low": risk_distribution["low"],
                    "medium": risk_distribution["medium"],
                    "high": risk_distribution["high"]
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения профилей риска: {str(e)}")


@router.get("/appointments/{appointment_id}/details", response_model=BaseResponse)
async def get_appointment_details(
    appointment_id: int,
    repository: MedialogRepository = Depends(get_medialog_repository)
) -> BaseResponse:
    """
    Получение детальной информации о записи
    
    Args:
        appointment_id: ID записи
        repository: Репозиторий медиалог
        
    Returns:
        Детальная информация о записи
    """
    try:
        appointment = repository.get_appointment_by_id(appointment_id)
        
        if not appointment:
            raise HTTPException(status_code=404, detail=f"Запись {appointment_id} не найдена")
        
        # Получение данных пациента
        patient = repository.get_patient_by_id(appointment.patients_id)
        
        # Получение прогнозов
        predictions = repository.get_predictions_by_appointment(appointment_id)
        
        appointment_details = {
            "appointment_id": appointment.appointment_id,
            "appointment_time": appointment.appointment_time.isoformat(),
            "visit_type": appointment.visit_type,
            "status": appointment.status,
            "no_show_flag": appointment.no_show_flag,
            "patient": {
                "patient_id": patient.patients_id if patient else None,
                "name": f"{patient.surname} {patient.name}" if patient else "Неизвестно",
                "age": (datetime.utcnow() - patient.birth_date).days // 365 if patient else None,
                "gender": patient.gender if patient else None
            } if patient else None,
            "predictions": [
                {
                    "prediction_id": pred.id,
                    "probability": pred.prediction_probability,
                    "risk_level": pred.risk_level,
                    "recommendations": pred.recommendations,
                    "model_version": pred.model_version,
                    "prediction_date": pred.prediction_date.isoformat()
                } for pred in predictions
            ]
        }
        
        return BaseResponse(
            success=True,
            message=f"Детальная информация о записи {appointment_id}",
            data=appointment_details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения деталей записи: {str(e)}")


@router.get("/health", response_model=BaseResponse)
async def health_check(
    prediction_service: NoShowPredictionService = Depends(get_prediction_service)
) -> BaseResponse:
    """
    Проверка состояния сервиса прогнозирования
    
    Args:
        prediction_service: Сервис прогнозирования
        
    Returns:
        Статус сервиса
    """
    try:
        model_info = prediction_service.get_model_info()
        
        return BaseResponse(
            success=True,
            message="Сервис прогнозирования неявок работает",
            data={
                "service_status": "healthy",
                "model_status": model_info["status"],
                "model_type": model_info.get("model_type"),
                "feature_count": model_info.get("feature_count"),
                "training_metrics": model_info.get("training_metrics")
            }
        )
        
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f"Ошибка сервиса прогнозирования: {str(e)}",
            data={"service_status": "unhealthy"}
        ) 