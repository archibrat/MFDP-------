from typing import Optional, List, Dict, Any
from sqlmodel import Session, select
from services.http.base_service import BaseService
from repositories.user_repository import UserRepository
from app.models.user import User, UserCreate, UserUpdate, UserResponse
from auth.password_utils import password_manager
from app.exceptions.base_exception import ValidationException, ConflictException
from services.logging.logging import get_logger

logger = get_logger(logger_name=__name__)

class UserService(BaseService[User, UserCreate, UserUpdate, UserRepository]):
    """Сервис для работы с пользователями"""
    
    def __init__(self, session: Session):
        repository = UserRepository()
        super().__init__(repository)
        self.session = session
        
        # Регистрируем хуки
        self.add_after_create_hook(self._log_user_created)
        self.add_after_update_hook(self._log_user_updated)
        self.add_before_delete_hook(self._validate_user_deletion)
    
    # Переопределяем методы валидации
    def validate_create(self, session: Session, obj_in: UserCreate) -> None:
        """
        Валидация при создании пользователя
        
        Аргументы:
            session: Сессия базы данных
            obj_in: Данные для создания
            
        Исключения:
            ValidationException: При ошибке валидации
        """
        # Проверяем уникальность email
        if self.repository.email_exists(session, obj_in.email):
            raise ConflictException(f"Пользователь с email {obj_in.email} уже существует")
        
        # Проверяем уникальность username
        if self.repository.username_exists(session, obj_in.username):
            raise ConflictException(f"Пользователь с именем {obj_in.username} уже существует")
    
    def validate_update(self, session: Session, obj: User, obj_in: UserUpdate) -> None:
        """
        Валидация при обновлении пользователя
        
        Аргументы:
            session: Сессия базы данных
            obj: Существующий объект
            obj_in: Данные для обновления
            
        Исключения:
            ValidationException: При ошибке валидации
        """
        # Проверяем уникальность email (если изменяется)
        if obj_in.email and obj_in.email != obj.email and self.repository.email_exists(session, obj_in.email, exclude_id=obj.id):
            raise ConflictException(f"Пользователь с email {obj_in.email} уже существует")
        if obj_in.username and obj_in.username != obj.username:
            if self.repository.username_exists(session, obj_in.username, exclude_id=obj.id):
                raise ConflictException(f"Пользователь с именем {obj_in.username} уже существует")
    
    def validate_delete(self, session: Session, obj: User) -> None:
        """
        Валидация при удалении пользователя
        
        Аргументы:
            session: Сессия базы данных
            obj: Объект для удаления
            
        Исключения:
            ValidationException: При ошибке валидации
        """
        # Проверяем, есть ли связанные записи
        # Здесь можно добавить проверки на связанные ML задачи, события и т.д.
        pass
    
    # Хуки
    async def before_create(self, session: Session, obj_in: UserCreate) -> UserCreate:
        """Хук перед созданием пользователя"""
        # Хешируем пароль
        obj_in.password = password_manager.validate_and_hash_password(obj_in.password)
        return await super().before_create(session, obj_in)
    
    async def _log_user_created(self, session: Session, obj: User) -> User:
        """Хук логирования после создания пользователя"""
        logger.info(f"User created: {obj.username} (ID: {obj.id})")
        return obj
    
    async def _log_user_updated(self, session: Session, obj: User) -> User:
        """Хук логирования после обновления пользователя"""
        logger.info(f"User updated: {obj.username} (ID: {obj.id})")
        return obj
    
    async def _validate_user_deletion(self, session: Session, obj: User) -> None:
        """Хук валидации перед удалением пользователя"""
        logger.warning(f"Attempting to delete user: {obj.username} (ID: {obj.id})")
    
    # Специфичные методы для пользователей
    def get_by_email(self, email: str) -> Optional[User]:
        """
        Получение пользователя по email
        
        Аргументы:
            email: Email пользователя
            
        Возвращает:
            Optional[User]: Пользователь или None
        """
        return self.repository.get_by_email(self.session, email)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """
        Получение пользователя по имени пользователя
        
        Аргументы:
            username: Имя пользователя
            
        Возвращает:
            Optional[User]: Пользователь или None
        """
        return self.repository.get_by_username(self.session, username)
    
    def get_active_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """
        Получение активных пользователей
        
        Аргументы:
            skip: Количество пользователей для пропуска
            limit: Максимальное количество пользователей
            
        Возвращает:
            List[User]: Список активных пользователей
        """
        return self.repository.get_active_users(self.session, skip, limit)
    
    def search_users(self, query: str, limit: int = 10) -> List[User]:
        """
        Поиск пользователей
        
        Аргументы:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Возвращает:
            List[User]: Список найденных пользователей
        """
        return self.repository.search_users(self.session, query, limit)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Аутентификация пользователя
        
        Аргументы:
            username: Имя пользователя или email
            password: Пароль
            
        Возвращает:
            Optional[User]: Аутентифицированный пользователь или None
        """
        # Попробуем найти по username
        user = self.get_by_username(username)
        
        # Если не найден, попробуем по email
        if not user:
            user = self.get_by_email(username)
        
        # Проверяем пароль
        if user and password_manager.verify_password(password, user.password):
            if user.is_active:
                return user
        
        return None
    
    def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """
        Изменение пароля пользователя
        
        Аргументы:
            user_id: ID пользователя
            current_password: Текущий пароль
            new_password: Новый пароль
            
        Возвращает:
            bool: True если пароль изменен успешно
            
        Исключения:
            ValidationException: При неверном текущем пароле
        """
        user = self.get_or_404(self.session, user_id)
        
        # Проверяем текущий пароль
        if not password_manager.verify_password(current_password, user.password):
            raise ValidationException("Неверный текущий пароль")
        
        # Хешируем новый пароль
        new_hashed_password = password_manager.validate_and_hash_password(new_password)
        
        # Обновляем пароль
        user.password = new_hashed_password
        self.session.add(user)
        self.session.commit()
        
        logger.info(f"Password changed for user: {user.username} (ID: {user.id})")
        return True
    
    def deactivate_user(self, user_id: int) -> User:
        """
        Деактивация пользователя
        
        Аргументы:
            user_id: ID пользователя
            
        Возвращает:
            User: Деактивированный пользователь
        """
        user = self.get_or_404(self.session, user_id)
        user.is_active = False
        self.session.add(user)
        self.session.commit()
        
        logger.info(f"User deactivated: {user.username} (ID: {user.id})")
        return user
    
    def activate_user(self, user_id: int) -> User:
        """
        Активация пользователя
        
        Аргументы:
            user_id: ID пользователя
            
        Возвращает:
            User: Активированный пользователь
        """
        user = self.get_or_404(self.session, user_id)
        user.is_active = True
        self.session.add(user)
        self.session.commit()
        
        logger.info(f"User activated: {user.username} (ID: {user.id})")
        return user

# Функция для создания сервиса (для удобства)
def get_user_service(session: Session) -> UserService:
    """Создание экземпляра UserService"""
    return UserService(session)