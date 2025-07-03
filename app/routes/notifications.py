"""
API endpoints для системы уведомлений
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel

from services.notification_service import (
    get_notification_service,
    NotificationService,
    NotificationType,
    NotificationChannel,
    NotificationPreferences
)

notifications_router = APIRouter()


class SendNotificationRequest(BaseModel):
    user_id: str
    type: str
    title: str
    message: str
    channels: Optional[List[str]] = None


class SendTemplateNotificationRequest(BaseModel):
    user_id: str
    template_id: str
    variables: Dict[str, str]
    channels: Optional[List[str]] = None


class NotificationPreferencesRequest(BaseModel):
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = True
    web_enabled: bool = True
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None


@notifications_router.post("/send/")
async def send_notification(
    request: SendNotificationRequest,
    service: NotificationService = Depends(get_notification_service)
):
    """Отправка уведомления"""
    
    try:
        # Парсинг типа уведомления
        notification_type = NotificationType(request.type.lower())
        
        # Парсинг каналов
        channels = None
        if request.channels:
            channels = [NotificationChannel(ch.lower()) for ch in request.channels]
        
        # Отправка уведомления
        notification_id = await service.send_notification(
            user_id=request.user_id,
            type=notification_type,
            title=request.title,
            message=request.message,
            channels=channels
        )
        
        return {
            "status": "success",
            "notification_id": notification_id,
            "message": "Уведомление отправлено"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Некорректные параметры: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка отправки уведомления: {str(e)}")


@notifications_router.post("/send-template/")
async def send_template_notification(
    request: SendTemplateNotificationRequest,
    service: NotificationService = Depends(get_notification_service)
):
    """Отправка уведомления по шаблону"""
    
    try:
        # Парсинг каналов
        channels = None
        if request.channels:
            channels = [NotificationChannel(ch.lower()) for ch in request.channels]
        
        # Отправка по шаблону
        notification_id = await service.send_from_template(
            user_id=request.user_id,
            template_id=request.template_id,
            variables=request.variables,
            channels=channels
        )
        
        return {
            "status": "success",
            "notification_id": notification_id,
            "message": "Уведомление по шаблону отправлено"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка шаблона: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка отправки: {str(e)}")


@notifications_router.get("/users/{user_id}")
async def get_user_notifications(
    user_id: str,
    unread_only: bool = False,
    limit: int = 50,
    service: NotificationService = Depends(get_notification_service)
):
    """Получение уведомлений пользователя"""
    
    notifications = service.get_user_notifications(
        user_id=user_id,
        unread_only=unread_only,
        limit=limit
    )
    
    notifications_data = []
    for notification in notifications:
        notifications_data.append({
            "id": notification.id,
            "type": notification.type.value,
            "title": notification.title,
            "message": notification.message,
            "created_at": notification.created_at.isoformat(),
            "status": notification.status.value,
            "delivered_at": notification.delivered_at.isoformat() if notification.delivered_at else None
        })
    
    return {
        "user_id": user_id,
        "notifications": notifications_data,
        "total_count": len(notifications_data),
        "unread_count": len([n for n in notifications_data if n["status"] != "delivered"])
    }


@notifications_router.post("/mark-read/{notification_id}")
async def mark_notification_as_read(
    notification_id: str,
    service: NotificationService = Depends(get_notification_service)
):
    """Отметка уведомления как прочитанного"""
    
    success = service.mark_as_read(notification_id)
    
    if success:
        return {
            "status": "success",
            "message": "Уведомление отмечено как прочитанное"
        }
    else:
        raise HTTPException(status_code=404, detail="Уведомление не найдено")


@notifications_router.get("/templates/")
async def get_notification_templates(
    service: NotificationService = Depends(get_notification_service)
):
    """Получение списка шаблонов уведомлений"""
    
    templates_data = []
    for template_id, template in service.templates.items():
        templates_data.append({
            "id": template.id,
            "name": template.name,
            "type": template.type.value,
            "channels": [ch.value for ch in template.channels],
            "variables": template.variables,
            "subject_template": template.subject_template,
            "body_template": template.body_template
        })
    
    return {
        "templates": templates_data,
        "total_count": len(templates_data)
    }


@notifications_router.put("/preferences/{user_id}")
async def update_user_preferences(
    user_id: str,
    preferences: NotificationPreferencesRequest,
    service: NotificationService = Depends(get_notification_service)
):
    """Обновление предпочтений пользователя"""
    
    user_prefs = NotificationPreferences(
        user_id=user_id,
        email_enabled=preferences.email_enabled,
        sms_enabled=preferences.sms_enabled,
        push_enabled=preferences.push_enabled,
        web_enabled=preferences.web_enabled,
        quiet_hours_start=preferences.quiet_hours_start,
        quiet_hours_end=preferences.quiet_hours_end
    )
    
    service.set_user_preferences(user_id, user_prefs)
    
    return {
        "status": "success",
        "message": "Предпочтения пользователя обновлены"
    }


@notifications_router.get("/preferences/{user_id}")
async def get_user_preferences(
    user_id: str,
    service: NotificationService = Depends(get_notification_service)
):
    """Получение предпочтений пользователя"""
    
    prefs = service.user_preferences.get(user_id)
    
    if not prefs:
        # Возвращаем настройки по умолчанию
        prefs = NotificationPreferences(user_id=user_id)
    
    return {
        "user_id": user_id,
        "email_enabled": prefs.email_enabled,
        "sms_enabled": prefs.sms_enabled,
        "push_enabled": prefs.push_enabled,
        "web_enabled": prefs.web_enabled,
        "quiet_hours_start": prefs.quiet_hours_start,
        "quiet_hours_end": prefs.quiet_hours_end
    }


@notifications_router.get("/statistics/")
async def get_notification_statistics(
    days: int = 7,
    service: NotificationService = Depends(get_notification_service)
):
    """Получение статистики уведомлений"""
    
    stats = service.get_notification_statistics(days=days)
    
    return stats


@notifications_router.post("/test-send/")
async def test_notification_sending(
    user_id: str,
    service: NotificationService = Depends(get_notification_service)
):
    """Тестовая отправка уведомления"""
    
    try:
        notification_id = await service.send_notification(
            user_id=user_id,
            type=NotificationType.INFO,
            title="Тестовое уведомление",
            message="Это тестовое уведомление для проверки системы.",
            channels=[NotificationChannel.WEB]
        )
        
        return {
            "status": "success",
            "notification_id": notification_id,
            "message": "Тестовое уведомление отправлено"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка тестовой отправки: {str(e)}")


@notifications_router.get("/channels/")
async def get_notification_channels():
    """Получение списка доступных каналов уведомлений"""
    
    channels = [
        {
            "code": channel.value,
            "name": channel.value.capitalize(),
            "description": f"Уведомления через {channel.value}"
        }
        for channel in NotificationChannel
    ]
    
    return {
        "channels": channels
    }


@notifications_router.get("/types/")
async def get_notification_types():
    """Получение списка типов уведомлений"""
    
    types = [
        {
            "code": type.value,
            "name": type.value.capitalize(),
            "description": f"Уведомления типа {type.value}"
        }
        for type in NotificationType
    ]
    
    return {
        "types": types
    } 