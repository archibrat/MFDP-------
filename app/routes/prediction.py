from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from typing import List, Dict
import asyncio
import json
from datetime import datetime

from database.database import get_session
from services.ml_service import get_ml_service, MLService
from app.services.crud.prediction import PredictionService
from app.models.prediction import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest,
    ModelStatusResponse, PatientDataCreate
)
from auth.authenticate import authenticate

prediction_route = APIRouter(prefix="/api/predictions", tags=["predictions"])

def get_prediction_service(session: Session = Depends(get_session)) -> PredictionService:
    return PredictionService(session)

@prediction_route.post("/predict", response_model=PredictionResponse)
async def make_prediction(
    request: PredictionRequest,
    ml_service: MLService = Depends(get_ml_service),
    prediction_service: PredictionService = Depends(get_prediction_service),
    user: str = Depends(authenticate)
):
    """
    Создание предсказания для одного пациента
    
    Возвращает:
        PredictionResponse: Результат предсказания с рекомендациями
    """
    try:
        # Получение предсказания от ML-сервиса
        prediction_response = await ml_service.predict(request)
        
        # Сохранение в базу данных
        saved_prediction = await prediction_service.save_prediction(
            request.patient_data,
            prediction_response,
            user.id
        )
        
        prediction_response.prediction_id = saved_prediction.id
        
        return prediction_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@prediction_route.post("/predict/batch")
async def batch_prediction(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    ml_service: MLService = Depends(get_ml_service),
    prediction_service: PredictionService = Depends(get_prediction_service),
    user: str = Depends(authenticate)
):
    """
    Пакетное предсказание для множества пациентов
    
    Возвращает:
        dict: Информацию о запущенной задаче пакетного предсказания
    """
    try:
        # Создаем задачу для фонового выполнения
        task_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        background_tasks.add_task(
            _process_batch_predictions,
            request.patients_data,
            request.prediction_type,
            user.id,
            task_id,
            ml_service,
            prediction_service
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "total_patients": len(request.patients_data),
            "estimated_completion": "5-10 minutes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@prediction_route.get("/batch/{task_id}/status")
async def get_batch_status(
    task_id: str,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """Получение статуса пакетной задачи"""
    status = await prediction_service.get_batch_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@prediction_route.get("/batch/{task_id}/results")
async def get_batch_results(
    task_id: str,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """Получение результатов пакетной задачи"""
    results = await prediction_service.get_batch_results(task_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Возвращаем как поток для больших результатов
    def generate():
        yield json.dumps({"task_id": task_id, "results": results})
    
    return StreamingResponse(
        generate(),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=batch_results_{task_id}.json"}
    )

@prediction_route.get("/models/status", response_model=List[ModelStatusResponse])
async def get_models_status(
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Получение статуса всех загруженных моделей
    
    Возвращает:
        List[ModelStatusResponse]: Список статусов моделей
    """
    models_status = []
    
    for version, metadata in ml_service.model_manager.model_metadata.items():
        status = ModelStatusResponse(
            model_version=version,
            status="active" if version in ml_service.model_manager.models else "inactive",
            last_update=datetime.fromisoformat(metadata.get('training_date', datetime.utcnow().isoformat())),
            metrics=metadata.get('metrics', {}),
            active_predictions=await _get_active_predictions_count(version)
        )
        models_status.append(status)
    
    return models_status

@prediction_route.post("/models/{model_version}/reload")
async def reload_model(
    model_version: str,
    ml_service: MLService = Depends(get_ml_service),
    user: str = Depends(authenticate)
):
    """Перезагрузка конкретной версии модели"""
    success = await ml_service.model_manager.load_model(model_version)
    
    if success:
        return {"message": f"Model {model_version} reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to reload model {model_version}")

@prediction_route.get("/history/{patient_id}")
async def get_patient_prediction_history(
    patient_id: str,
    limit: int = 10,
    prediction_service: PredictionService = Depends(get_prediction_service),
    user: str = Depends(authenticate)
):
    """Получение истории предсказаний для пациента"""
    return await prediction_service.get_patient_history(patient_id, limit)

@prediction_route.get("/analytics/performance")
async def get_model_performance_analytics(
    days: int = 30,
    model_version: str = "latest",
    prediction_service: PredictionService = Depends(get_prediction_service),
    user: str = Depends(authenticate)
):
    """Получение аналитики производительности модели"""
    return await prediction_service.get_performance_analytics(days, model_version)

# Вспомогательные функции
async def _process_batch_predictions(
    patients_data: List[PatientDataCreate],
    prediction_type: str,
    user_id: int,
    task_id: str,
    ml_service: MLService,
    prediction_service: PredictionService
):
    """Фоновая обработка пакетных предсказаний"""
    try:
        results = []
        total = len(patients_data)
        
        # Обновляем статус
        await prediction_service.update_batch_status(task_id, "processing", 0, total)
        
        for i, patient_data in enumerate(patients_data):
            try:
                # Создаем запрос
                request = PredictionRequest(
                    patient_data=patient_data,
                    prediction_types=[prediction_type]
                )
                
                # Получаем предсказание
                prediction = await ml_service.predict(request)
                
                # Сохраняем
                saved = await prediction_service.save_prediction(
                    patient_data, prediction, user_id
                )
                
                results.append({
                    "patient_id": patient_data.client_id,
                    "prediction_id": saved.id,
                    "probability": prediction.prediction_value,
                    "risk_level": prediction.risk_level
                })
                
                # Обновляем прогресс
                await prediction_service.update_batch_status(task_id, "processing", i + 1, total)
                
            except Exception as e:
                results.append({
                    "patient_id": patient_data.client_id,
                    "error": str(e)
                })
        
        # Сохраняем результаты
        await prediction_service.save_batch_results(task_id, results)
        await prediction_service.update_batch_status(task_id, "completed", total, total)
        
    except Exception as e:
        await prediction_service.update_batch_status(task_id, "failed", 0, total, str(e))

async def _get_active_predictions_count(model_version: str) -> int:
    """Получение количества активных предсказаний для модели"""
    # Реализация подсчета активных предсказаний
    return 0
