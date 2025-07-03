"""
Сервис уведомлений для отправки alerts, email и push-уведомлений
"""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    WEB = "web"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"


class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"


@dataclass
class NotificationTemplate:
    id: str
    name: str
    type: NotificationType
    subject_template: str
    body_template: str
    channels: List[NotificationChannel]
    variables: List[str]


@dataclass
class Notification:
    id: str
    user_id: str
    type: NotificationType
    channel: NotificationChannel
    title: str
    message: str
    created_at: datetime
    status: NotificationStatus = NotificationStatus.PENDING
    delivered_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    template_id: Optional[str] = None


@dataclass
class NotificationPreferences:
    user_id: str
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = True
    web_enabled: bool = True
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    channels_by_type: Optional[Dict[str, List[str]]] = None


class NotificationService:
    """Основной сервис уведомлений"""
    
    def __init__(self):
        self.notifications: List[Notification] = []
        self.templates: Dict[str, NotificationTemplate] = {}
        self.user_preferences: Dict[str, NotificationPreferences] = {}
        self.storage_path = Path("./data/notifications")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Настройки SMTP
        self.smtp_settings = {
            "host": "localhost",
            "port": 587,
            "username": "",
            "password": "",
            "use_tls": True
        }
        
        # Настройки SMS
        self.sms_settings = {
            "api_url": "",
            "api_key": "",
            "sender": "MFDP"
        }
        
        # Инициализация базовых шаблонов
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Инициализация стандартных шаблонов уведомлений"""
        
        templates = [
            NotificationTemplate(
                id="high_risk_patient",
                name="Пациент высокого риска",
                type=NotificationType.WARNING,
                subject_template="Пациент высокого риска неявки: {patient_name}",
                body_template="Пациент {patient_name} имеет высокий риск неявки ({risk_probability}%) на прием {appointment_date}. Рекомендуется связаться с пациентом.",
                channels=[NotificationChannel.WEB, NotificationChannel.EMAIL],
                variables=["patient_name", "risk_probability", "appointment_date"]
            ),
            NotificationTemplate(
                id="ml_model_alert",
                name="Предупреждение ML модели",
                type=NotificationType.ERROR,
                subject_template="Проблема с ML моделью: {model_name}",
                body_template="ML модель {model_name} показывает аномальные результаты. Точность снизилась до {accuracy}%. Требуется проверка.",
                channels=[NotificationChannel.WEB, NotificationChannel.EMAIL],
                variables=["model_name", "accuracy"]
            ),
            NotificationTemplate(
                id="system_alert",
                name="Системное предупреждение",
                type=NotificationType.CRITICAL,
                subject_template="Критическая ошибка системы",
                body_template="Обнаружена критическая ошибка: {error_message}. Время: {timestamp}. Требуется немедленное внимание.",
                channels=[NotificationChannel.WEB, NotificationChannel.EMAIL, NotificationChannel.SMS],
                variables=["error_message", "timestamp"]
            ),
            NotificationTemplate(
                id="daily_report",
                name="Ежедневный отчет",
                type=NotificationType.INFO,
                subject_template="Ежедневный отчет MFDP - {date}",
                body_template="Сводка за {date}: Обработано пациентов: {patients_count}, Точность прогнозов: {accuracy}%, Неявки: {no_shows}",
                channels=[NotificationChannel.EMAIL],
                variables=["date", "patients_count", "accuracy", "no_shows"]
            )
        ]
        
        for template in templates:
            self.templates[template.id] = template
    
    async def send_notification(self,
                              user_id: str,
                              type: NotificationType,
                              title: str,
                              message: str,
                              channels: Optional[List[NotificationChannel]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Отправка уведомления"""
        
        notification_id = f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Получение предпочтений пользователя
        user_prefs = self.user_preferences.get(user_id, NotificationPreferences(user_id=user_id))
        
        # Определение каналов для отправки
        if channels is None:
            channels = self._get_default_channels_for_type(type, user_prefs)
        else:
            channels = self._filter_by_user_preferences(channels, user_prefs)
        
        # Проверка тихих часов
        if self._is_quiet_hours(user_prefs):
            if type not in [NotificationType.CRITICAL, NotificationType.ERROR]:
                # Отложить неважные уведомления
                await self._schedule_notification(user_id, type, title, message, channels, metadata)
                return notification_id
        
        # Отправка по всем каналам
        results = []
        for channel in channels:
            notification = Notification(
                id=f"{notification_id}_{channel.value}",
                user_id=user_id,
                type=type,
                channel=channel,
                title=title,
                message=message,
                created_at=datetime.now(),
                metadata=metadata
            )
            
            try:
                success = await self._send_via_channel(notification, channel)
                notification.status = NotificationStatus.SENT if success else NotificationStatus.FAILED
                if success:
                    notification.delivered_at = datetime.now()
                
                self.notifications.append(notification)
                results.append(success)
                
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {str(e)}")
                notification.status = NotificationStatus.FAILED
                self.notifications.append(notification)
                results.append(False)
        
        # Сохранение в файл
        await self._save_notifications_to_file()
        
        return notification_id
    
    async def send_from_template(self,
                               user_id: str,
                               template_id: str,
                               variables: Dict[str, Any],
                               channels: Optional[List[NotificationChannel]] = None) -> str:
        """Отправка уведомления по шаблону"""
        
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Заполнение шаблона
        try:
            title = template.subject_template.format(**variables)
            message = template.body_template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for template {template_id}")
        
        # Использование каналов из шаблона, если не указаны
        if channels is None:
            channels = template.channels
        
        return await self.send_notification(
            user_id=user_id,
            type=template.type,
            title=title,
            message=message,
            channels=channels,
            metadata={"template_id": template_id, "variables": variables}
        )
    
    async def _send_via_channel(self, notification: Notification, channel: NotificationChannel) -> bool:
        """Отправка через конкретный канал"""
        
        try:
            if channel == NotificationChannel.WEB:
                return await self._send_web_notification(notification)
            elif channel == NotificationChannel.EMAIL:
                return await self._send_email_notification(notification)
            elif channel == NotificationChannel.SMS:
                return await self._send_sms_notification(notification)
            elif channel == NotificationChannel.PUSH:
                return await self._send_push_notification(notification)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook_notification(notification)
            else:
                logger.warning(f"Unsupported notification channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send via {channel.value}: {str(e)}")
            return False
    
    async def _send_web_notification(self, notification: Notification) -> bool:
        """Отправка веб-уведомления (сохранение в базе)"""
        # Веб-уведомления просто сохраняются для отображения в интерфейсе
        return True
    
    async def _send_email_notification(self, notification: Notification) -> bool:
        """Отправка email уведомления"""
        try:
            # Получение email пользователя (заглушка)
            user_email = f"user_{notification.user_id}@example.com"
            
            # Создание сообщения
            msg = MIMEMultipart()
            msg['From'] = self.smtp_settings['username']
            msg['To'] = user_email
            msg['Subject'] = notification.title
            
            body = notification.message
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # Отправка (симуляция)
            logger.info(f"Email sent to {user_email}: {notification.title}")
            return True
            
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            return False
    
    async def _send_sms_notification(self, notification: Notification) -> bool:
        """Отправка SMS уведомления"""
        try:
            # Получение телефона пользователя (заглушка)
            user_phone = f"+7900000{notification.user_id.zfill(4)}"
            
            # Отправка через API (симуляция)
            logger.info(f"SMS sent to {user_phone}: {notification.message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"SMS sending failed: {str(e)}")
            return False
    
    async def _send_push_notification(self, notification: Notification) -> bool:
        """Отправка push-уведомления"""
        try:
            # Push-уведомления через FCM или другой сервис (симуляция)
            logger.info(f"Push notification sent to user {notification.user_id}: {notification.title}")
            return True
            
        except Exception as e:
            logger.error(f"Push notification failed: {str(e)}")
            return False
    
    async def _send_webhook_notification(self, notification: Notification) -> bool:
        """Отправка webhook уведомления"""
        try:
            webhook_url = "https://example.com/webhook"  # Заглушка
            
            payload = {
                "id": notification.id,
                "user_id": notification.user_id,
                "type": notification.type.value,
                "title": notification.title,
                "message": notification.message,
                "timestamp": notification.created_at.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Webhook notification failed: {str(e)}")
            return False
    
    def _get_default_channels_for_type(self, 
                                     type: NotificationType, 
                                     prefs: NotificationPreferences) -> List[NotificationChannel]:
        """Получение каналов по умолчанию для типа уведомления"""
        
        channels = []
        
        if prefs.web_enabled:
            channels.append(NotificationChannel.WEB)
        
        if type in [NotificationType.CRITICAL, NotificationType.ERROR]:
            if prefs.email_enabled:
                channels.append(NotificationChannel.EMAIL)
            if prefs.sms_enabled:
                channels.append(NotificationChannel.SMS)
        elif type == NotificationType.WARNING:
            if prefs.email_enabled:
                channels.append(NotificationChannel.EMAIL)
            if prefs.push_enabled:
                channels.append(NotificationChannel.PUSH)
        else:  # INFO, SUCCESS
            if prefs.push_enabled:
                channels.append(NotificationChannel.PUSH)
        
        return channels
    
    def _filter_by_user_preferences(self, 
                                   channels: List[NotificationChannel],
                                   prefs: NotificationPreferences) -> List[NotificationChannel]:
        """Фильтрация каналов по предпочтениям пользователя"""
        
        filtered = []
        
        for channel in channels:
            if channel == NotificationChannel.WEB and prefs.web_enabled:
                filtered.append(channel)
            elif channel == NotificationChannel.EMAIL and prefs.email_enabled:
                filtered.append(channel)
            elif channel == NotificationChannel.SMS and prefs.sms_enabled:
                filtered.append(channel)
            elif channel == NotificationChannel.PUSH and prefs.push_enabled:
                filtered.append(channel)
            else:
                # Webhook всегда разрешен
                if channel == NotificationChannel.WEBHOOK:
                    filtered.append(channel)
        
        return filtered
    
    def _is_quiet_hours(self, prefs: NotificationPreferences) -> bool:
        """Проверка тихих часов"""
        
        if not prefs.quiet_hours_start or not prefs.quiet_hours_end:
            return False
        
        current_time = datetime.now().time()
        
        try:
            start_time = datetime.strptime(prefs.quiet_hours_start, "%H:%M").time()
            end_time = datetime.strptime(prefs.quiet_hours_end, "%H:%M").time()
            
            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:  # Период через полночь
                return current_time >= start_time or current_time <= end_time
                
        except ValueError:
            return False
    
    async def _schedule_notification(self, *args, **kwargs):
        """Планирование отложенного уведомления"""
        # Простая реализация - сохранение в список отложенных
        logger.info("Notification scheduled for later delivery")
    
    async def _save_notifications_to_file(self):
        """Сохранение уведомлений в файл"""
        try:
            filename = self.storage_path / "notifications.json"
            
            # Сохраняем только последние 1000 уведомлений
            recent_notifications = self.notifications[-1000:]
            
            data = []
            for notif in recent_notifications:
                notif_dict = asdict(notif)
                notif_dict['created_at'] = notif.created_at.isoformat()
                if notif.delivered_at:
                    notif_dict['delivered_at'] = notif.delivered_at.isoformat()
                notif_dict['type'] = notif.type.value
                notif_dict['channel'] = notif.channel.value
                notif_dict['status'] = notif.status.value
                data.append(notif_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save notifications: {str(e)}")
    
    def get_user_notifications(self, 
                             user_id: str,
                             unread_only: bool = False,
                             limit: int = 50) -> List[Notification]:
        """Получение уведомлений пользователя"""
        
        user_notifications = [
            notif for notif in self.notifications 
            if notif.user_id == user_id and notif.channel == NotificationChannel.WEB
        ]
        
        if unread_only:
            user_notifications = [
                notif for notif in user_notifications 
                if notif.status != NotificationStatus.DELIVERED
            ]
        
        # Сортировка по дате (новые первыми)
        user_notifications.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_notifications[:limit]
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Отметка уведомления как прочитанного"""
        
        for notification in self.notifications:
            if notification.id == notification_id:
                notification.status = NotificationStatus.DELIVERED
                notification.delivered_at = datetime.now()
                return True
        
        return False
    
    def set_user_preferences(self, user_id: str, preferences: NotificationPreferences):
        """Установка предпочтений пользователя"""
        self.user_preferences[user_id] = preferences
    
    def get_notification_statistics(self, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Статистика уведомлений"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        
        if end_date is None:
            end_date = datetime.now()
        
        # Фильтрация по дате
        filtered_notifications = [
            notif for notif in self.notifications
            if start_date <= notif.created_at <= end_date
        ]
        
        # Подсчет статистики
        total = len(filtered_notifications)
        sent = len([n for n in filtered_notifications if n.status == NotificationStatus.SENT])
        failed = len([n for n in filtered_notifications if n.status == NotificationStatus.FAILED])
        
        # Статистика по типам
        type_stats = {}
        for notif in filtered_notifications:
            type_name = notif.type.value
            if type_name not in type_stats:
                type_stats[type_name] = 0
            type_stats[type_name] += 1
        
        # Статистика по каналам
        channel_stats = {}
        for notif in filtered_notifications:
            channel_name = notif.channel.value
            if channel_name not in channel_stats:
                channel_stats[channel_name] = 0
            channel_stats[channel_name] += 1
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_notifications": total,
                "sent_successfully": sent,
                "failed": failed,
                "success_rate": round(sent / total * 100, 2) if total > 0 else 100
            },
            "by_type": type_stats,
            "by_channel": channel_stats
        }


# Глобальный экземпляр сервиса уведомлений
notification_service = NotificationService()


def get_notification_service() -> NotificationService:
    """Получение экземпляра сервиса уведомлений"""
    return notification_service 