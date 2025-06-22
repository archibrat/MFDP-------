from typing import Generic, TypeVar, Optional, List, Dict, Any, Callable
from sqlmodel import Session
from app.repositories.base_repository import BaseRepository
from app.models.base import PaginationParams, PaginatedResponse
from app.exceptions import ValidationException, NotFoundException
from app.services.logging.logging import get_logger
import asyncio

logger = get_logger(__name__)

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")
RepositoryType = TypeVar("RepositoryType", bound=BaseRepository)

class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType, RepositoryType]):
    """
    Унифицированный базовый сервисный слой с хуками для валидации и обработки.
    """

    def __init__(self, repository: RepositoryType):
        self.repository = repository
        self._before_create_hooks: List[Callable] = []
        self._after_create_hooks: List[Callable] = []
        self._before_update_hooks: List[Callable] = []
        self._after_update_hooks: List[Callable] = []
        self._before_delete_hooks: List[Callable] = []
        self._after_delete_hooks: List[Callable] = []

    # Методы для регистрации хуков
    def add_before_create_hook(self, hook: Callable):
        self._before_create_hooks.append(hook)

    def add_after_create_hook(self, hook: Callable):
        self._after_create_hooks.append(hook)

    def add_before_update_hook(self, hook: Callable):
        self._before_update_hooks.append(hook)

    def add_after_update_hook(self, hook: Callable):
        self._after_update_hooks.append(hook)

    def add_before_delete_hook(self, hook: Callable):
        self._before_delete_hooks.append(hook)

    def add_after_delete_hook(self, hook: Callable):
        self._after_delete_hooks.append(hook)

    # Базовые хуки (можно переопределять в наследниках)
    async def before_create(self, session: Session, obj_in: CreateSchemaType) -> CreateSchemaType:
        for hook in self._before_create_hooks:
            if asyncio.iscoroutinefunction(hook):
                obj_in = await hook(session, obj_in)
            else:
                obj_in = hook(session, obj_in)
        return obj_in

    async def after_create(self, session: Session, obj: ModelType) -> ModelType:
        for hook in self._after_create_hooks:
            if asyncio.iscoroutinefunction(hook):
                obj = await hook(session, obj)
            else:
                obj = hook(session, obj)
        return obj

    async def before_update(self, session: Session, obj: ModelType, obj_in: UpdateSchemaType) -> UpdateSchemaType:
        for hook in self._before_update_hooks:
            if asyncio.iscoroutinefunction(hook):
                obj_in = await hook(session, obj, obj_in)
            else:
                obj_in = hook(session, obj, obj_in)
        return obj_in

    async def after_update(self, session: Session, obj: ModelType) -> ModelType:
        for hook in self._after_update_hooks:
            if asyncio.iscoroutinefunction(hook):
                obj = await hook(session, obj)
            else:
                obj = hook(session, obj)
        return obj

    async def before_delete(self, session: Session, obj: ModelType) -> None:
        for hook in self._before_delete_hooks:
            if asyncio.iscoroutinefunction(hook):
                await hook(session, obj)
            else:
                hook(session, obj)

    async def after_delete(self, session: Session, obj: ModelType) -> None:
        for hook in self._after_delete_hooks:
            if asyncio.iscoroutinefunction(hook):
                await hook(session, obj)
            else:
                hook(session, obj)

    # Методы валидации (можно переопределять в наследниках)
    def validate_create(self, session: Session, obj_in: CreateSchemaType) -> None:
        pass

    def validate_update(self, session: Session, obj: ModelType, obj_in: UpdateSchemaType) -> None:
        pass

    def validate_delete(self, session: Session, obj: ModelType) -> None:
        pass

    # CRUD-операции
    async def create(self, session: Session, *, obj_in: CreateSchemaType) -> ModelType:
        logger.info(f"Creating {self.repository.model.__name__}")
        self.validate_create(session, obj_in)
        obj_in = await self.before_create(session, obj_in)
        obj = self.repository.create(session, obj_in=obj_in)
        obj = await self.after_create(session, obj)
        logger.info(f"Created {self.repository.model.__name__} with ID: {getattr(obj, 'id', None)}")
        return obj

    def get(self, session: Session, id: int) -> Optional[ModelType]:
        return self.repository.get(session, id)

    def get_or_404(self, session: Session, id: int) -> ModelType:
        obj = self.get(session, id)
        if not obj:
            raise NotFoundException(f"{self.repository.model.__name__} with id={id} not found")
        return obj

    def get_multi(self, session: Session, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[ModelType]:
        return self.repository.get_multi(session, skip=skip, limit=limit, filters=filters)

    def get_paginated(self, session: Session, pagination: PaginationParams, filters: Optional[Dict[str, Any]] = None) -> PaginatedResponse:
        return self.repository.get_paginated(session, pagination, filters)

    async def update(self, session: Session, *, db_obj: ModelType, obj_in: UpdateSchemaType) -> ModelType:
        self.validate_update(session, db_obj, obj_in)
        obj_in = await self.before_update(session, db_obj, obj_in)
        obj = self.repository.update(session, db_obj=db_obj, obj_in=obj_in)
        obj = await self.after_update(session, obj)
        return obj

    async def delete(self, session: Session, *, id: int) -> ModelType:
        obj = self.get_or_404(session, id)
        self.validate_delete(session, obj)
        await self.before_delete(session, obj)
        deleted_obj = self.repository.delete(session, id=id)
        await self.after_delete(session, deleted_obj)
        return deleted_obj

    def exists(self, session: Session, id: int) -> bool:
        return self.repository.exists(session, id)

    def count(self, session: Session, filters: Optional[Dict[str, Any]] = None) -> int:
        return self.repository.count(session, filters=filters)