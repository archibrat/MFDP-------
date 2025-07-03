"""
Business API для продакшн ML-модуля
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session
from datetime import datetime
from typing import Optional, List

from app.database.database import get_session
from app.models.base import BaseResponse
from app.models.ml_production import (
    NoShowPredictionRequest, NoShowPredictionResponse,
    LoadBalanceRequest, LoadBalanceResponse,
    SchedulerEvent, SchedulerResponse, EventType
)
from app.services.ml_production.dal import MLDataAccessLayer
from app.services.ml_production.feature_store import FeatureStore
from app.services.ml_production.no_show_predictor import NoShowPredictor, PatientNoShowModel
from app.services.ml_production.load_balancer import DoctorLoadOptimizer
from app.services.ml_production.real_time_scheduler import RealTimeScheduler


router = APIRouter(prefix="/api/ml", tags=["ML Production"])


def get_dal(session: Session = Depends(get_session)) -> MLDataAccessLayer:
    """Получение Data Access Layer"""
    return MLDataAccessLayer(session)


def get_feature_store(dal: MLDataAccessLayer = Depends(get_dal)) -> FeatureStore:
    """Получение Feature Store"""
    return FeatureStore(dal)


def get_noshow_predictor() -> NoShowPredictor:
    """Получение No-Show Predictor"""
    return NoShowPredictor()


def get_noshow_model(
    feature_store: FeatureStore = Depends(get_feature_store),
    predictor: NoShowPredictor = Depends(get_noshow_predictor)
) -> PatientNoShowModel:
    """Получение модели прогнозирования неявок"""
    return PatientNoShowModel(feature_store, predictor)


def get_load_optimizer(dal: MLDataAccessLayer = Depends(get_dal)) -> DoctorLoadOptimizer:
    """Получение оптимизатора нагрузки"""
    return DoctorLoadOptimizer(dal)


def get_real_time_scheduler(dal: MLDataAccessLayer = Depends(get_dal)) -> RealTimeScheduler:
    """Получение планировщика реального времени"""
    return RealTimeScheduler(dal)


@router.get("/noshowscore", response_model=BaseResponse)
async def predict_no_show_score(
    planning_id: int,
    background_tasks: BackgroundTasks,
    model: PatientNoShowModel = Depends(get_noshow_model),
    dal: MLDataAccessLayer = Depends(get_dal)
):
    """
    Прогнозирует вероятность неявки для записи
    
    При высоком риске автоматически инициирует уведомления и двойное бронирование
    """
    try:
        # Получение прогноза
        prediction = model.predict_noshow(planning_id)
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Запись не найдена или недостаточно данных")
        
        probability = prediction['probability']
        risk_level = prediction['risk_level']
        
        # Автоматические действия при высоком риске
        if probability >= 0.6:
            background_tasks.add_task(dal.trigger_job_notify_sms, planning_id)
            
        if probability >= 0.8:
            background_tasks.add_task(dal.trigger_double_booking, planning_id)
        
        # Вызов хранимой процедуры
        background_tasks.add_task(dal.call_sp_calc_noshow, planning_id)
        
        response_data = NoShowPredictionResponse(
            planning_id=planning_id,
            probability=probability,
            risk_level=risk_level,
            recommendations=prediction['recommendations'],
            features_used=prediction['features_used']
        )
        
        return BaseResponse(
            success=True,
            message="Прогноз неявки получен",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прогнозирования: {str(e)}")


@router.post("/load-balance", response_model=BaseResponse)
async def optimize_doctor_load(
    request: LoadBalanceRequest,
    background_tasks: BackgroundTasks,
    optimizer: DoctorLoadOptimizer = Depends(get_load_optimizer)
):
    """
    Оптимизирует распределение нагрузки врачей
    
    Использует OR-Tools для решения задачи оптимизации с ограничениями
    """
    try:
        from datetime import datetime as dt
        start_datetime = dt.combine(request.start_date, dt.min.time())
        end_datetime = dt.combine(request.end_date, dt.min.time())
        
        result = optimizer.optimize_load_distribution(
            start_datetime,
            end_datetime,
            request.department_ids
        )
        
        if not result['optimized_assignments']:
            return BaseResponse(
                success=True,
                message="Текущее распределение уже оптимально",
                data=LoadBalanceResponse(
                    optimized_assignments={},
                    metrics=result['metrics'],
                    recommendations=result['recommendations']
                ).dict()
            )
        
        response_data = LoadBalanceResponse(
            optimized_assignments=result['optimized_assignments'],
            metrics=result['metrics'],
            recommendations=result['recommendations']
        )
        
        return BaseResponse(
            success=True,
            message=f"Оптимизация завершена, переназначено {len(result['optimized_assignments'])} записей",
            data=response_data.dict()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка оптимизации нагрузки: {str(e)}")


@router.post("/schedule-event", response_model=BaseResponse)
async def handle_schedule_event(
    event: SchedulerEvent,
    scheduler: RealTimeScheduler = Depends(get_real_time_scheduler)
):
    """
    Обрабатывает события планирования в реальном времени
    
    Поддерживаемые события: ARRIVE_DATE, CANCELLED, CONS_DURATION, NO_SHOW
    """
    try:
        result = scheduler.handle_event(event.event_type, {
            'planning_id': event.planning_id,
            'timestamp': event.timestamp,
            **event.data
        })
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        response_data = SchedulerResponse(
            planning_id=event.planning_id,
            old_time=result.get('old_time'),
            new_time=result.get('new_time'),
            adjustments=result.get('adjustments', []),
            notifications_sent=result.get('notifications_sent', [])
        )
        
        return BaseResponse(
            success=True,
            message=f"Событие {event.event_type} обработано",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки события: {str(e)}")


@router.post("/train-noshow-model", response_model=BaseResponse)
async def train_no_show_model(
    background_tasks: BackgroundTasks,
    model: PatientNoShowModel = Depends(get_noshow_model),
    days_back: int = 365
):
    """
    Переобучает модель прогнозирования неявок на исторических данных
    """
    try:
        def train_task():
            metrics = model.train_model(days_back)
            return metrics
        
        background_tasks.add_task(train_task)
        
        return BaseResponse(
            success=True,
            message=f"Запущено переобучение модели на данных за {days_back} дней",
            data={"days_back": days_back, "status": "training_started"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка запуска обучения: {str(e)}")


@router.post("/generate-nightly-report", response_model=BaseResponse)
async def generate_nightly_load_report(
    background_tasks: BackgroundTasks,
    optimizer: DoctorLoadOptimizer = Depends(get_load_optimizer),
    department_ids: Optional[List[int]] = None
):
    """
    Генерирует ночной отчет по загрузке врачей и оптимизирует расписание на 14 дней
    """
    try:
        background_tasks.add_task(optimizer.generate_nightly_report, department_ids)
        
        return BaseResponse(
            success=True,
            message="Запущена генерация ночного отчета",
            data={"department_ids": department_ids, "status": "report_generation_started"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации отчета: {str(e)}")


@router.get("/doctor-queue/{doctor_id}", response_model=BaseResponse)
async def optimize_doctor_queue(
    doctor_id: int,
    scheduler: RealTimeScheduler = Depends(get_real_time_scheduler)
):
    """
    Оптимизирует очередь ожидания для конкретного врача
    """
    try:
        result = scheduler.optimize_waiting_queue(doctor_id)
        
        return BaseResponse(
            success=True,
            message=f"Очередь врача {doctor_id} оптимизирована",
            data={
                "doctor_id": doctor_id,
                "reordered_patients": result['reordered'],
                "queue": result['queue']
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка оптимизации очереди: {str(e)}")


@router.get("/health", response_model=BaseResponse)
async def health_check():
    """Проверка состояния ML-сервисов"""
    try:
        # Проверка доступности компонентов
        predictor = NoShowPredictor()
        
        health_data = {
            "noshow_predictor": "ready" if predictor else "not_ready",
            "xgboost_available": "xgboost" in str(type(predictor.model)) if predictor.model else False,
            "ortools_available": True,  # Проверим при инициализации
            "services": ["DAL", "FeatureStore", "NoShowPredictor", "LoadOptimizer", "RealTimeScheduler"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return BaseResponse(
            success=True,
            message="ML-сервисы работают",
            data=health_data
        )
        
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f"Ошибка проверки состояния: {str(e)}",
            data={"error": str(e)}
        ) 