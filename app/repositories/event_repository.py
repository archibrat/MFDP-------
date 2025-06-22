from typing import Optional, List
from sqlmodel import Session, select, and_, or_
from datetime import datetime, timedelta
from repositories.base_repository import BaseRepository
from models.event import Event, EventCreate, EventUpdate, EventType, EventSeverity

class EventRepository(BaseRepository[Event, EventCreate, EventUpdate]):
    """Репозиторий для работы с событиями"""
    
    def __init__(self):
        super().__init__(Event)
    
    def get_by_type(self, session: Session, event_type: EventType, limit: int = 100) -> List[Event]:
        """Получение событий по типу"""
        statement = select(Event).where(Event.event_type == event_type).limit(limit)
        return session.exec(statement).all()
    
    def get_by_severity(self, session: Session, severity: EventSeverity, limit: int = 100) -> List[Event]:
        """Получение событий по уровню важности"""
        statement = select(Event).where(Event.severity == severity).limit(limit)
        return session.exec(statement).all()
    
    def get_by_user(self, session: Session, user_id: int, limit: int = 100) -> List[Event]:
        """Получение событий пользователя"""
        statement = select(Event).where(Event.user_id == user_id).limit(limit)
        return session.exec(statement).all()
    
    def get_recent_events(self, session: Session, hours: int = 24, limit: int = 100) -> List[Event]:
        """Получение недавних событий"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        statement = select(Event).where(
            Event.created_at >= cutoff_time
        ).order_by(Event.created_at.desc()).limit(limit)
        return session.exec(statement).all()
    
    def get_critical_events(self, session: Session, hours: int = 24) -> List[Event]:
        """Получение критических событий"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        statement = select(Event).where(
            and_(
                Event.severity == EventSeverity.CRITICAL,
                Event.created_at >= cutoff_time
            )
        ).order_by(Event.created_at.desc())
        return session.exec(statement).all()
    
    def search_events(self, session: Session, query: str, limit: int = 100) -> List[Event]:
        """Поиск событий по сообщению"""
        statement = select(Event).where(
            Event.message.contains(query)
        ).order_by(Event.created_at.desc()).limit(limit)
        return session.exec(statement).all()
    
    def get_events_by_date_range(
        self, 
        session: Session, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 1000
    ) -> List[Event]:
        """Получение событий за период"""
        statement = select(Event).where(
            and_(
                Event.created_at >= start_date,
                Event.created_at <= end_date
            )
        ).order_by(Event.created_at.desc()).limit(limit)
        return session.exec(statement).all()
    
    def count_by_type(self, session: Session, event_type: EventType) -> int:
        """Подсчет событий по типу"""
        return self.count(session, {"event_type": event_type})
    
    def count_by_severity(self, session: Session, severity: EventSeverity) -> int:
        """Подсчет событий по уровню важности"""
        return self.count(session, {"severity": severity})
    
    def get_user_activity(self, session: Session, user_id: int, days: int = 7) -> List[Event]:
        """Получение активности пользователя"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        statement = select(Event).where(
            and_(
                Event.user_id == user_id,
                Event.created_at >= cutoff_time
            )
        ).order_by(Event.created_at.desc())
        return session.exec(statement).all()

# Глобальный экземпляр репозитория
event_repository = EventRepository()