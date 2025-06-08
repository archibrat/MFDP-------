from models.user import User, UserCreate
from models.event import Event
from sqlmodel import Session, select
from sqlalchemy.orm import selectinload
from typing import List, Optional

def get_all_users(session: Session) -> List[User]:
    """
    Получить всех пользователей с их событиями.
    
    Аргументы:
        session: Сессия базы данных
    
    Возвращает:
        List[User]: Список всех пользователей
    """
    try:
        statement = select(User).options(
            selectinload(User.events).selectinload(Event.creator)
        )
        users = session.exec(statement).all()
        return users
    except Exception as e:
        raise

def get_user_by_id(user_id: int, session: Session) -> Optional[User]:
    """
    Получить пользователя по ID.
    
    Аргументы:
        user_id: ID пользователя для поиска
        session: Сессия базы данных
    
    Возвращает:
        Optional[User]: Найденный пользователь или None
    """
    try:
        statement = select(User).where(User.id == user_id).options(
            selectinload(User.events)
        )
        user = session.exec(statement).first()
        return user
    except Exception as e:
        raise

def get_user_by_email(email: str, session: Session) -> Optional[User]:
    """
    Получить пользователя по email.
    
    Аргументы:
        email: Email для поиска
        session: Сессия базы данных
    
    Возвращает:
        Optional[User]: Найденный пользователь или None
    """
    try:
        statement = select(User).where(User.email == email).options(
            selectinload(User.events)
        )
        user = session.exec(statement).first()
        return user
    except Exception as e:
        raise

def create_user(user: UserCreate, session: Session) -> User:
    """
    Создать нового пользователя.
    
    Аргументы:
        user: DTO для создания пользователя
        session: Сессия базы данных
    
    Возвращает:
        User: Созданный пользователь с ID
    """
    db_user = User.from_orm(user)
    try:
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        return db_user
    except Exception as e:
        session.rollback()
        raise

def delete_user(user_id: int, session: Session) -> bool:
    """
    Удалить пользователя по ID.
    
    Аргументы:
        user_id: ID пользователя для удаления
        session: Сессия базы данных
    
    Возвращает:
        bool: True если удален, False если не найден
    """
    try:
        user = get_user_by_id(user_id, session)
        if user:
            session.delete(user)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        raise