from typing import Optional, List
from sqlmodel import Session, select
from repositories.base_repository import BaseRepository
from app.models.user import User, UserCreate, UserUpdate

class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """Репозиторий для работы с пользователями"""
    
    def __init__(self):
        super().__init__(User)
    
    def get_by_email(self, session: Session, email: str) -> Optional[User]:
        """Получение пользователя по email"""
        statement = select(User).where(User.email == email)
        return session.exec(statement).first()
    
    def get_by_username(self, session: Session, username: str) -> Optional[User]:
        """Получение пользователя по имени пользователя"""
        statement = select(User).where(User.username == username)
        return session.exec(statement).first()
    
    def get_active_users(self, session: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Получение активных пользователей"""
        statement = select(User).where(User.is_active == True).offset(skip).limit(limit)
        return list(session.exec(statement).all())
    
    def search_users(self, session: Session, query: str, limit: int = 10) -> List[User]:
        """Поиск пользователей по имени или email"""
        # Простой поиск по точному совпадению
        statement = select(User).where(
            (User.username == query) | 
            (User.email == query) |
            (User.full_name == query)
        ).limit(limit)
        return list(session.exec(statement).all())
    
    def email_exists(self, session: Session, email: str, exclude_id: Optional[int] = None) -> bool:
        """Проверка существования email"""
        statement = select(User).where(User.email == email)
        if exclude_id:
            statement = statement.where(User.id != exclude_id)
        return session.exec(statement).first() is not None
    
    def username_exists(self, session: Session, username: str, exclude_id: Optional[int] = None) -> bool:
        """Проверка существования имени пользователя"""
        statement = select(User).where(User.username == username)
        if exclude_id:
            statement = statement.where(User.id != exclude_id)
        return session.exec(statement).first() is not None
    
    def get_admins(self, session: Session) -> List[User]:
        """Получение списка администраторов"""
        statement = select(User).where(User.role == "admin")
        return list(session.exec(statement).all())
    
    def count_active_users(self, session: Session) -> int:
        """Подсчет активных пользователей"""
        return self.count(session, {"is_active": True})

# Глобальный экземпляр репозитория
user_repository = UserRepository()