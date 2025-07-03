"""
API маршруты для модуля адаптивного планирования реального времени
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from typing import Dict, Any
from sqlmodel import Session

from app.database.database import get_session
from app.repositories.medialog_repository import MedialogRepository
from app.services.real_time_scheduler_service import RealTimeScheduler
from app.models.base import BaseResponse
from app.services.real_time_scheduler_service import EventType

router = APIRouter(prefix="/api/real-time-scheduler", tags=["Real Time Scheduler"])


def get_scheduler_service(session: Session = Depends(get_session)) -> RealTimeScheduler:
    repository = MedialogRepository(session)
    return RealTimeScheduler(repository)


@router.post("/event", response_model=BaseResponse)
async def handle_event(
    event_type: EventType,
    event_data: Dict[str, Any],
    scheduler: RealTimeScheduler = Depends(get_scheduler_service)
):
    """Обработка события расписания (отмена, задержка, экстренный случай)"""
    try:
        result = scheduler.handle_event(event_type, event_data)
        return BaseResponse(success=True, message="Событие обработано", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки события: {e}")


@router.get("/health", response_model=BaseResponse)
async def health_check():
    """Проверка состояния сервиса адаптивного планирования"""
    return BaseResponse(success=True, message="Сервис адаптивного планирования работает", data={"status": "healthy"}) 