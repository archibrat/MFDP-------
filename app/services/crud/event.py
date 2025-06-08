from models.event import Event, EventUpdate
from sqlmodel import Session, select
from typing import List, Optional
from datetime import datetime

def get_all_events(session: Session) -> List[Event]:
    """
    Получение всех событий.
    
    Аргументы:
        session: Сессия базы данных
    
    Возвращает:
        List[Event]: Список всех событий
    """
    try:
        statement = select(Event)
        events = session.exec(statement).all()
        return events
    except Exception as e:
        raise

def get_event_by_id(event_id: int, session: Session) -> Optional[Event]:
    """
    Получение события по ID.
    
    Аргументы:
        event_id: ID события для поиска
        session: Сессия базы данных
    
    Возвращает:
        Optional[Event]: Найденное событие или None
    """
    try:
        statement = select(Event).where(Event.id == event_id)
        event = session.exec(statement).first()
        return event
    except Exception as e:
        raise

def update_event(event_id: int, event_update: EventUpdate, session: Session) -> Optional[Event]:
    """
    Обновление существующего события.
    
    Аргументы:
        event_id: ID события для обновления
        event_update: Новые данные события
        session: Сессия базы данных
    
    Возвращает:
        Optional[Event]: Обновленное событие или None, если не найдено
    """
    try:
        event = get_event_by_id(event_id, session)
        if not event:
            return None

        event_data = event_update.dict(exclude_unset=True)
        for key, value in event_data.items():
            setattr(event, key, value)

        session.add(event)
        session.commit()
        session.refresh(event)
        return event
    except Exception as e:
        session.rollback()
        raise

def create_event(event: Event, session: Session) -> Event:
    """
    Создание нового события.
    
    Аргументы:
        event: Событие для создания
        session: Сессия базы данных
    
    Возвращает:
        Event: Созданное событие с присвоенным ID
    """
    try:
        session.add(event)
        session.commit()
        session.refresh(event)
        return event
    except Exception as e:
        session.rollback()
        raise
    
def delete_all_events(session: Session) -> int:
    """
    Удаление всех событий.
    
    Аргументы:
        session: Сессия базы данных
    
    Возвращает:
        int: Количество удаленных событий
    """
    try:
        statement = select(Event)
        events = session.exec(statement).all()
        count = len(events)
        
        for event in events:
            session.delete(event)
        
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        raise
    
def delete_event(event_id: int, session: Session) -> bool:
    """
    Удаление события по ID.
    
    Аргументы:
        event_id: ID события для удаления
        session: Сессия базы данных
    
    Возвращает:
        bool: True если удалено, False если не найдено
    """
    try:
        event = get_event_by_id(event_id, session)
        if not event:
            return False
            
        session.delete(event)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        raise

