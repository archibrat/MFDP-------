"""
API endpoints для системы аудита
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from services.audit_service import (
    get_audit_service,
    AuditService,
    AuditActionType,
    AuditLevel
)

audit_router = APIRouter()


@audit_router.get("/")
async def get_audit_records(
    user_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    action_type: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    service: AuditService = Depends(get_audit_service)
):
    """Получение записей аудита с фильтрацией"""
    
    try:
        # Парсинг дат
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Парсинг типа действия
        action_types = None
        if action_type:
            try:
                action_types = [AuditActionType(action_type)]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Неверный тип действия: {action_type}")
        
        # Получение записей
        if user_id:
            records = service.get_user_actions(
                user_id=user_id,
                start_date=start_dt,
                end_date=end_dt,
                action_types=action_types
            )
        else:
            records = service._filter_records(
                start_date=start_dt,
                end_date=end_dt,
                action_types=action_types
            )
        
        # Ограничение количества
        records = records[:limit]
        
        # Сериализация
        records_data = []
        for record in records:
            records_data.append({
                "id": record.id,
                "timestamp": record.timestamp.isoformat(),
                "user_id": record.user_id,
                "user_name": record.user_name,
                "action_type": record.action_type.value,
                "resource": record.resource,
                "resource_id": record.resource_id,
                "description": record.description,
                "level": record.level.value,
                "success": record.success,
                "ip_address": record.ip_address,
                "details": record.details
            })
        
        return {
            "records": records_data,
            "total_count": len(records_data),
            "filters": {
                "user_id": user_id,
                "start_date": start_date,
                "end_date": end_date,
                "action_type": action_type
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения записей аудита: {str(e)}")


@audit_router.get("/users/{user_id}")
async def get_user_audit_log(
    user_id: str,
    days: int = Query(30, le=365),
    service: AuditService = Depends(get_audit_service)
):
    """Получение журнала аудита для конкретного пользователя"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    records = service.get_user_actions(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    # Статистика по действиям
    action_stats = {}
    for record in records:
        action = record.action_type.value
        if action not in action_stats:
            action_stats[action] = 0
        action_stats[action] += 1
    
    # Последние действия
    recent_actions = []
    for record in records[:10]:
        recent_actions.append({
            "timestamp": record.timestamp.isoformat(),
            "action": record.action_type.value,
            "resource": record.resource,
            "description": record.description,
            "success": record.success
        })
    
    return {
        "user_id": user_id,
        "period_days": days,
        "total_actions": len(records),
        "action_statistics": action_stats,
        "recent_actions": recent_actions
    }


@audit_router.get("/security-events/")
async def get_security_events(
    days: int = Query(7, le=30),
    service: AuditService = Depends(get_audit_service)
):
    """Получение событий безопасности"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    events = service.get_security_events(start_date=start_date, end_date=end_date)
    
    security_data = []
    for event in events:
        security_data.append({
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "user_name": event.user_name,
            "action": event.action_type.value,
            "description": event.description,
            "level": event.level.value,
            "success": event.success,
            "ip_address": event.ip_address
        })
    
    # Анализ неудачных попыток входа
    failed_logins = [e for e in events if e.action_type == AuditActionType.LOGIN and not e.success]
    
    return {
        "period_days": days,
        "total_events": len(events),
        "failed_login_attempts": len(failed_logins),
        "events": security_data[:100]  # Ограничиваем количество
    }


@audit_router.get("/statistics/")
async def get_audit_statistics(
    days: int = Query(30, le=365),
    service: AuditService = Depends(get_audit_service)
):
    """Получение статистики аудита"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    stats = service.get_statistics(start_date=start_date, end_date=end_date)
    
    return stats


@audit_router.post("/search/")
async def search_audit_records(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(50, le=500),
    service: AuditService = Depends(get_audit_service)
):
    """Поиск записей аудита по тексту"""
    
    try:
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        records = service.search_records(
            query=query,
            start_date=start_dt,
            end_date=end_dt
        )
        
        records = records[:limit]
        
        search_results = []
        for record in records:
            search_results.append({
                "id": record.id,
                "timestamp": record.timestamp.isoformat(),
                "user_name": record.user_name,
                "action": record.action_type.value,
                "resource": record.resource,
                "description": record.description,
                "level": record.level.value
            })
        
        return {
            "query": query,
            "results_count": len(search_results),
            "results": search_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")


@audit_router.get("/export/")
async def export_audit_log(
    format: str = Query("json", regex="^(json|csv)$"),
    days: int = Query(30, le=365),
    service: AuditService = Depends(get_audit_service)
):
    """Экспорт журнала аудита"""
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        export_data = service.export_audit_log(
            start_date=start_date,
            end_date=end_date,
            format=format
        )
        
        return {
            "export_format": format,
            "period_days": days,
            "data": export_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(e)}")


@audit_router.delete("/cleanup/")
async def cleanup_old_records(
    days_to_keep: int = Query(90, ge=30, le=365),
    service: AuditService = Depends(get_audit_service)
):
    """Очистка старых записей аудита"""
    
    try:
        cleaned_count = service.cleanup_old_records(days_to_keep=days_to_keep)
        
        return {
            "status": "success",
            "cleaned_records": cleaned_count,
            "days_kept": days_to_keep,
            "message": f"Удалено {cleaned_count} записей старше {days_to_keep} дней"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка очистки: {str(e)}")


@audit_router.get("/failed-actions/")
async def get_failed_actions(
    days: int = Query(7, le=30),
    service: AuditService = Depends(get_audit_service)
):
    """Получение неудачных действий"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    failed_actions = service.get_failed_actions(start_date=start_date, end_date=end_date)
    
    failed_data = []
    for action in failed_actions[:50]:  # Ограничиваем количество
        failed_data.append({
            "timestamp": action.timestamp.isoformat(),
            "user_name": action.user_name,
            "action": action.action_type.value,
            "resource": action.resource,
            "description": action.description,
            "level": action.level.value,
            "ip_address": action.ip_address
        })
    
    return {
        "period_days": days,
        "total_failed": len(failed_actions),
        "failed_actions": failed_data
    } 