from typing import Optional, List
from sqlmodel import Session, select, and_
from datetime import datetime, timedelta
from repositories.base_repository import BaseRepository
from models.mltask import MLTask, MLTaskCreate, MLTaskUpdate, TaskStatus, TaskType

class MLTaskRepository(BaseRepository[MLTask, MLTaskCreate, MLTaskUpdate]):
    """Репозиторий для работы с ML задачами"""
    
    def __init__(self):
        super().__init__(MLTask)
    
    def get_by_status(self, session: Session, status: TaskStatus, limit: int = 100) -> List[MLTask]:
        """Получение задач по статусу"""
        statement = select(MLTask).where(MLTask.status == status).limit(limit)
        return session.exec(statement).all()
    
    def get_by_type(self, session: Session, task_type: TaskType, limit: int = 100) -> List[MLTask]:
        """Получение задач по типу"""
        statement = select(MLTask).where(MLTask.task_type == task_type).limit(limit)
        return session.exec(statement).all()
    
    def get_by_user(self, session: Session, user_id: int, limit: int = 100) -> List[MLTask]:
        """Получение задач пользователя"""
        statement = select(MLTask).where(MLTask.user_id == user_id).limit(limit)
        return session.exec(statement).all()
    
    def get_pending_tasks(self, session: Session, limit: int = 100) -> List[MLTask]:
        """Получение задач в очереди, отсортированных по приоритету"""
        statement = select(MLTask).where(
            MLTask.status == TaskStatus.PENDING
        ).order_by(MLTask.priority.desc(), MLTask.created_at.asc()).limit(limit)
        return session.exec(statement).all()
    
    def get_active_tasks(self, session: Session) -> List[MLTask]:
        """Получение активных задач (в процессе выполнения)"""
        statement = select(MLTask).where(MLTask.status == TaskStatus.PROCESSING)
        return session.exec(statement).all()
    
    def get_completed_tasks(self, session: Session, days: int = 7, limit: int = 100) -> List[MLTask]:
        """Получение завершенных задач за период"""
        cutoff_time = datetime.now() - timedelta(days=days)
        statement = select(MLTask).where(
            and_(
                MLTask.status == TaskStatus.COMPLETED,
                MLTask.completed_at >= cutoff_time
            )
        ).order_by(MLTask.completed_at.desc()).limit(limit)
        return session.exec(statement).all()
    
    def get_failed_tasks(self, session: Session, days: int = 7, limit: int = 100) -> List[MLTask]:
        """Получение неудачных задач за период"""
        cutoff_time = datetime.now() - timedelta(days=days)
        statement = select(MLTask).where(
            and_(
                MLTask.status == TaskStatus.FAILED,
                MLTask.updated_at >= cutoff_time
            )
        ).order_by(MLTask.updated_at.desc()).limit(limit)
        return session.exec(statement).all()
    
    def get_tasks_by_priority(self, session: Session, min_priority: int = 1, limit: int = 100) -> List[MLTask]:
        """Получение задач с приоритетом выше указанного"""
        statement = select(MLTask).where(
            MLTask.priority >= min_priority
        ).order_by(MLTask.priority.desc(), MLTask.created_at.asc()).limit(limit)
        return session.exec(statement).all()
    
    def count_by_status(self, session: Session, status: TaskStatus) -> int:
        """Подсчет задач по статусу"""
        return self.count(session, {"status": status})
    
    def count_by_type(self, session: Session, task_type: TaskType) -> int:
        """Подсчет задач по типу"""
        return self.count(session, {"task_type": task_type})
    
    def get_queue_stats(self, session: Session) -> dict:
        """Получение статистики очереди задач"""
        return {
            "pending": self.count_by_status(session, TaskStatus.PENDING),
            "processing": self.count_by_status(session, TaskStatus.PROCESSING),
            "completed": self.count_by_status(session, TaskStatus.COMPLETED),
            "failed": self.count_by_status(session, TaskStatus.FAILED),
            "cancelled": self.count_by_status(session, TaskStatus.CANCELLED)
        }
    
    def cleanup_old_tasks(self, session: Session, days: int = 30) -> int:
        """Очистка старых завершенных задач"""
        cutoff_time = datetime.now() - timedelta(days=days)
        statement = select(MLTask).where(
            and_(
                MLTask.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED]),
                MLTask.updated_at < cutoff_time
            )
        )
        
        tasks_to_delete = session.exec(statement).all()
        count = len(tasks_to_delete)
        
        for task in tasks_to_delete:
            session.delete(task)
        
        session.commit()
        return count

# Глобальный экземпляр репозитория
mltask_repository = MLTaskRepository()