from fastapi import APIRouter, HTTPException, status, Depends
from database.database import get_session
from models.event import Event 
from typing import List

event_route = APIRouter()
events = []

@event_route.get("/", response_model=List[Event]) 
async def retrieve_all_events() -> List[Event]:
    """
    Получить список всех событий.
    
    Возвращает:
        List[Event]: Список всех событий
    """
    return events

@event_route.get("/{id}") 
async def retrieve_event(id: int):
    """
    Получить событие по ID.
    
    Аргументы:
        id (int): Идентификатор события
        
    Возвращает:
        dict: Данные события
        
    Вызывает:
        HTTPException: Если событие не найдено
    """
    for event in events: 
        if event['id'] == id:
            return event 
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, 
        detail="Событие с указанным ID не существует"
    )

@event_route.post("/new")
async def create_event(body: dict) -> dict:
    """
    Создать новое событие.
    
    Аргументы:
        body (dict): Данные события
        
    Возвращает:
        dict: Сообщение об успешном создании
    """
    events.append(body)
    return {"message": "Событие успешно создано"}

@event_route.delete("/{id}")
async def delete_event(id: int) -> dict:
    """
    Удалить событие по ID.
    
    Аргументы:
        id (int): Идентификатор события для удаления
        
    Возвращает:
        dict: Сообщение об успешном удалении
        
    Вызывает:
        HTTPException: Если событие не найдено
    """ 
    for event in events:
        if event.id == id: 
            events.remove(event)
            return {"message": "Событие успешно удалено"}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, 
        detail="Событие с указанным ID не существует"
    )

@event_route.delete("/")
async def delete_all_events() -> dict:
    """
    Удалить все события.
    
    Возвращает:
        dict: Сообщение об успешном удалении всех событий
    """
    events.clear()
    return {"message": "Все события успешно удалены"}