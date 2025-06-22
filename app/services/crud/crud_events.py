from typing import Optional, List, Dict, Any
from sqlmodel import Session, select, and_, or_
from datetime import datetime, timedelta
from services.http.base_service import BaseService
from repositories.event_repository import EventRepository
from models.event import Event
from models.event import EventCreate, EventUpdate, EventFilterParams, EventType, EventSeverity
from exceptions import ValidationException
from services.logging.logging import get_logger

logger = get_logger(__name__)

class EventService(BaseService[Event, EventCreate, EventUpdate, EventRepository]):
    """Сервис для работы с событиями"""
    
    def __init__(self, session: Session):
        repository = EventRepository()
        super().__init__(repository)
        self.session = session
        
        # Регистрируем хуки
        self.add_after_create_hook(self._log_event_created)
        self.add_before_delete_hook(self._validate_event_deletion)
    
    # Переопределяем методы валидации
    def validate_create(self, session: Session, obj_in: EventCreate) -> None:
        """
        Валидация при создании события
        
        Аргументы:
            session: Сессия базы данных
            obj_in: Данные для создания
            
        Исключения:
            ValidationException: При ошибке валидации
        """
        # Проверяем, что сообщение не пустое
        if not obj_in.message or not obj_in.message.strip():
            raise ValidationException("Сообщение события не может быть пустым")
        
        # Проверяем размер метаданных
        if obj_in.metadata and len(str(obj_in.metadata)) > 5000:
            raise ValidationException("Метаданные события слишком большие")
    
    def validate_delete(self, session: Session, obj: Event) -> None:
        """
        Валидация при удалении события
        
        Аргументы:
            session: Сессия базы данных
            obj: Объект для удаления
            
        Исключения:
            ValidationException: При ошибке валидации
        """
        # Проверяем, можно ли удалить критические события
        if obj.severity == EventSeverity.CRITICAL:
            # Разрешаем удаление только старых критических событий (старше 30 дней)
            if obj.created_at > datetime.utcnow() - timedelta(days=30):
                raise ValidationException("Нельзя удалять критические события младше 30 дней")
    
    # Хуки
    async def _log_event_created(self, session: Session, obj: Event) -> Event:
        """Хук логирования после создания события"""
        logger.info(f"Event created: {obj.event_type} - {obj.message} (ID: {obj.id})")
        return obj
    
    async def _validate_event_deletion(self, session: Session, obj: Event) -> None:
        """Хук валидации перед удалением события"""
        logger.warning(f"Attempting to delete event: {obj.event_type} (ID: {obj.id})")
    
    # Специфичные методы для событий
    def get_by_type(self, event_type: EventType, limit: int = 100) -> List[Event]:
        """
        Получение событий по типу
        
        Аргументы:
            event_type: Тип события
            limit: Максимальное количество событий
            
        Возвращает:
            List[Event]: Список событий
        """
        return self.repository.get_by_type(self.session, event_type, limit)
    
    def get_by_severity(self, severity: EventSeverity, limit: int = 100) -> List[Event]:
        """
        Получение событий по уровню важности
        
        Аргументы:
            severity: Уровень важности
            limit: Максимальное количество событий
            
        Возвращает:
            List[Event]: Список событий
        """
        return self.repository.get_by_severity(self.session, severity, limit)
    
    def get_by_user(self, user_id: int, limit: int = 100) -> List[Event]:
        """
        Получение событий пользователя
        
        Аргументы:
            user_id: ID пользователя
            limit: Максимальное количество событий
            
        Возвращает:
            List[Event]: Список событий
        """
        return self.repository.get_by_user(self.session, user_id, limit)
    
    def get_recent_events(self, hours: int = 24, limit: int = 100) -> List[Event]:
        """
        Получение недавних событий
        
        Аргументы:
            hours: Количество часов назад
            limit: Максимальное количество событий
            
        Возвращает:
            List[Event]: Список недавних событий
        """
        return self.repository.get_recent_events(self.session, hours, limit)
    
    def get_critical_events(self, hours: int = 24) -> List[Event]:
        """
        Получение критических событий
        
        Аргументы:
            hours: Количество часов назад
            
        Возвращает:
            List[Event]: Список критических событий
        """
        return self.repository.get_critical_events(self.session, hours)
    
    def search_events(self, query: str, limit: int = 100) -> List[Event]:
        """
        Поиск событий по сообщению
        
        Аргументы:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Возвращает:
            List[Event]: Список найденных событий
        """
        return self.repository.search_events(self.session, query, limit)
    
    def get_filtered_events(self, filters: EventFilterParams) -> List[Event]:
        """
        Получение событий с фильтрацией
        
        Аргументы:
            filters: Параметры фильтрации
            
        Возвращает:
            List[Event]: Список отфильтрованных событий
        """
        query = select(Event)
        
        # Применяем фильтры
        conditions = []
        
        if filters.event_type:
            conditions.append(Event.event_type == filters.event_type)
        
        if filters.severity:
            conditions.append(Event.severity == filters.severity)
        
        if filters.user_id:
            conditions.append(Event.user_id == filters.user_id)
        
        if filters.date_from:
            conditions.append(Event.created_at >= filters.date_from)
        
        if filters.date_to:
            conditions.append(Event.created_at <= filters.date_to)
        
        if filters.search:
            conditions.append(Event.message.contains(filters.search))
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(Event.created_at.desc()).limit(100)
        
        return self.session.exec(query).all()
    
    def create_system_event(
        self, 
        event_type: EventType, 
        message: str, 
        severity: EventSeverity = EventSeverity.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None
    ) -> Event:
        """
        Создание системного события
        
        Аргументы:
            event_type: Тип события
            message: Сообщение
            severity: Уровень важности
            metadata: Дополнительные данные
            user_id: ID пользователя (опционально)
            
        Возвращает:
            Event: Созданное событие
        """
        event_data = EventCreate(
            event_type=event_type,
            message=message,
            severity=severity,
            metadata=metadata,
            user_id=user_id
        )
        
        return self.repository.create(self.session, obj_in=event_data)
    
    def create_user_event(
        self, 
        user_id: int,
        event_type: EventType, 
        message: str, 
        severity: EventSeverity = EventSeverity.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Создание пользовательского события
        
        Аргументы:
            user_id: ID пользователя
            event_type: Тип события
            message: Сообщение
            severity: Уровень важности
            metadata: Дополнительные данные
            
        Возвращает:
            Event: Созданное событие
        """
        return self.create_system_event(
            event_type=event_type,
            message=message,
            severity=severity,
            metadata=metadata,
            user_id=user_id
        )
    
    def cleanup_old_events(self, days: int = 90) -> int:
        """
        Очистка старых событий
        
        Аргументы:
            days: Количество дней для хранения событий
            
        Возвращает:
            int: Количество удаленных событий
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Удаляем только события с низким и средним уровнем важности
        query = select(Event).where(
            and_(
                Event.created_at < cutoff_date,
                Event.severity.in_([EventSeverity.LOW, EventSeverity.MEDIUM])
            )
        )
        
        events_to_delete = self.session.exec(query).all()
        count = len(events_to_delete)
        
        for event in events_to_delete:
            self.session.delete(event)
        
        self.session.commit()
        
        logger.info(f"Cleaned up {count} old events older than {days} days")
        return count
    
    def get_event_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Получение статистики событий
        
        Аргументы:
            days: Количество дней для анализа
            
        Возвращает:
            Dict[str, Any]: Статистика событий
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Общее количество событий
        total_query = select(Event).where(Event.created_at >= cutoff_date)
        total_events = len(self.session.exec(total_query).all())
        
        # Статистика по типам
        type_stats = {}
        for event_type in EventType:
            type_query = select(Event).where(
                and_(
                    Event.created_at >= cutoff_date,
                    Event.event_type == event_type
                )
            )
            type_stats[event_type.value] = len(self.session.exec(type_query).all())
        
        # Статистика по уровням важности
        severity_stats = {}
        for severity in EventSeverity:
            severity_query = select(Event).where(
                and_(
                    Event.created_at >= cutoff_date,
                    Event.severity == severity
                )
            )
            severity_stats[severity.value] = len(self.session.exec(severity_query).all())
        
        return {
            "period_days": days,
            "total_events": total_events,
            "by_type": type_stats,
            "by_severity": severity_stats,
            "generated_at": datetime.utcnow().isoformat()
        }

# Функция для создания сервиса (для удобства)
def get_event_service(session: Session) -> EventService:
    """Создание экземпляра EventService"""
    return EventService(session)