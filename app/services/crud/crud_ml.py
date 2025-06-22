from typing import Optional, List, Dict, Any
from sqlmodel import Session, select, and_, or_
from datetime import datetime, timedelta
from services.http.base_service import BaseService
from repositories.mltask_repository import MLTaskRepository
from models.mltask import MLTask
from models.mltask import MLTaskCreate, MLTaskUpdate, MLTaskFilterParams, TaskStatus, TaskType
from exceptions import ValidationException, ConflictException
from services.logging.logging import get_logger

logger = get_logger(__name__)

class MLTaskService(BaseService[MLTask, MLTaskCreate, MLTaskUpdate, MLTaskRepository]):
    """Сервис для работы с ML задачами"""
    
    def __init__(self, session: Session):
        repository = MLTaskRepository()
        super().__init__(repository)
        self.session = session
        
        # Регистрируем хуки
        self.add_after_create_hook(self._log_task_created)
        self.add_after_update_hook(self._log_task_updated)
        self.add_before_delete_hook(self._validate_task_deletion)
    
    # Переопределяем методы валидации
    def validate_create(self, session: Session, obj_in: MLTaskCreate) -> None:
        """
        Валидация при создании ML задачи
        
        Аргументы:
            session: Сессия базы данных
            obj_in: Данные для создания
            
        Исключения:
            ValidationException: При ошибке валидации
        """
        # Проверяем входные данные
        if not obj_in.input_data:
            raise ValidationException("Входные данные не могут быть пустыми")
        
        # Проверяем размер входных данных
        if len(str(obj_in.input_data)) > 50000:
            raise ValidationException("Входные данные слишком большие")
        
        # Проверяем параметры
        if obj_in.parameters and len(str(obj_in.parameters)) > 10000:
            raise ValidationException("Параметры задачи слишком большие")
    
    def validate_update(self, session: Session, obj: MLTask, obj_in: MLTaskUpdate) -> None:
        """
        Валидация при обновлении ML задачи
        
        Аргументы:
            session: Сессия базы данных
            obj: Существующий объект
            obj_in: Данные для обновления
            
        Исключения:
            ValidationException: При ошибке валидации
        """
        # Проверяем переходы статусов
        if obj_in.status and obj_in.status != obj.status:
            self._validate_status_transition(obj.status, obj_in.status)
        
        # Проверяем размер результата
        if obj_in.result and len(str(obj_in.result)) > 100000:
            raise ValidationException("Результат задачи слишком большой")
    
    def validate_delete(self, session: Session, obj: MLTask) -> None:
        """
        Валидация при удалении ML задачи
        
        Аргументы:
            session: Сессия базы данных
            obj: Объект для удаления
            
        Исключения:
            ValidationException: При ошибке валидации
        """
        # Нельзя удалять выполняющиеся задачи
        if obj.status == TaskStatus.PROCESSING:
            raise ValidationException("Нельзя удалять выполняющиеся задачи")
    
    def _validate_status_transition(self, current_status: TaskStatus, new_status: TaskStatus) -> None:
        """
        Валидация перехода статусов
        
        Аргументы:
            current_status: Текущий статус
            new_status: Новый статус
            
        Исключения:
            ValidationException: При недопустимом переходе
        """
        # Определяем допустимые переходы
        allowed_transitions = {
            TaskStatus.PENDING: [TaskStatus.PROCESSING, TaskStatus.CANCELLED],
            TaskStatus.PROCESSING: [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED],
            TaskStatus.COMPLETED: [],  # Завершенные задачи нельзя изменять
            TaskStatus.FAILED: [TaskStatus.PENDING],  # Можно перезапустить
            TaskStatus.CANCELLED: [TaskStatus.PENDING]  # Можно перезапустить
        }
        
        if new_status not in allowed_transitions.get(current_status, []):
            raise ValidationException(
                f"Недопустимый переход статуса с {current_status.value} на {new_status.value}"
            )
    
    # Хуки
    async def before_create(self, session: Session, obj_in: MLTaskCreate) -> MLTaskCreate:
        """Хук перед созданием ML задачи"""
        # Устанавливаем начальный статус
        obj_in.status = TaskStatus.PENDING
        return await super().before_create(session, obj_in)
    
    async def _log_task_created(self, session: Session, obj: MLTask) -> MLTask:
        """Хук логирования после создания задачи"""
        logger.info(f"ML Task created: {obj.task_type} (ID: {obj.id}, Priority: {obj.priority})")
        return obj
    
    async def _log_task_updated(self, session: Session, obj: MLTask) -> MLTask:
        """Хук логирования после обновления задачи"""
        logger.info(f"ML Task updated: {obj.task_type} (ID: {obj.id}, Status: {obj.status})")
        return obj
    
    async def _validate_task_deletion(self, session: Session, obj: MLTask) -> None:
        """Хук валидации перед удалением задачи"""
        logger.warning(f"Attempting to delete ML task: {obj.task_type} (ID: {obj.id})")
    
