from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlmodel import SQLModel, Session, select, func
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from models.base import PaginationParams, PaginatedResponse

# Типы для generic репозитория
ModelType = TypeVar("ModelType", bound=SQLModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=SQLModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=SQLModel)

class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Базовый репозиторий с типобезопасными CRUD операциями
    
    Типы:
        ModelType: Тип модели SQLModel
        CreateSchemaType: Схема для создания объекта
        UpdateSchemaType: Схема для обновления объекта
    """
    
    def __init__(self, model: Type[ModelType]):
        """
        Инициализация репозитория
        
        Аргументы:
            model: Класс модели SQLModel
        """
        self.model = model
    
    def create(self, session: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """
        Создание нового объекта
        
        Аргументы:
            session: Сессия базы данных
            obj_in: Данные для создания объекта
            
        Возвращает:
            ModelType: Созданный объект
            
        Исключения:
            HTTPException: При ошибке создания
        """
        try:
            # Создаем объект из входных данных
            if isinstance(obj_in, dict):
                db_obj = self.model(**obj_in)
            else:
                obj_data = obj_in.model_dump(exclude_unset=True)
                db_obj = self.model(**obj_data)

            return self._extracted_from_update_23(session, db_obj)
        except IntegrityError as e:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ошибка целостности данных при создании объекта"
            )
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Внутренняя ошибка сервера при создании объекта"
            )
    
    def get(self, session: Session, id: int) -> Optional[ModelType]:
        """
        Получение объекта по ID
        
        Аргументы:
            session: Сессия базы данных
            id: Идентификатор объекта
            
        Возвращает:
            Optional[ModelType]: Объект или None
        """
        return session.get(self.model, id)
    
    def get_or_404(self, session: Session, id: int) -> ModelType:
        """
        Получение объекта по ID с исключением при отсутствии
        
        Аргументы:
            session: Сессия базы данных
            id: Идентификатор объекта
            
        Возвращает:
            ModelType: Найденный объект
            
        Исключения:
            HTTPException: При отсутствии объекта
        """
        if obj := self.get(session, id):
            return obj
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Объект с ID {id} не найден"
            )
    
    def get_multi(
        self, 
        session: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """
        Получение списка объектов
        
        Аргументы:
            session: Сессия базы данных
            skip: Количество объектов для пропуска
            limit: Максимальное количество объектов
            filters: Фильтры для запроса
            
        Возвращает:
            List[ModelType]: Список объектов
        """
        query = select(self.model)
        
        # Применяем фильтры
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    query = query.where(getattr(self.model, field) == value)
        
        query = query.offset(skip).limit(limit)
        return session.exec(query).all()
    
    def get_paginated(
        self,
        session: Session,
        pagination: PaginationParams,
        filters: Optional[Dict[str, Any]] = None
    ) -> PaginatedResponse:
        """
        Получение пагинированного списка объектов
        
        Аргументы:
            session: Сессия базы данных
            pagination: Параметры пагинации
            filters: Фильтры для запроса
            
        Возвращает:
            PaginatedResponse: Пагинированный ответ
        """
        # Подсчет общего количества
        count_query = select(func.count(self.model.id))
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    count_query = count_query.where(getattr(self.model, field) == value)
        
        total = session.exec(count_query).one()
        
        # Получение данных
        skip = (pagination.page - 1) * pagination.size
        items = self.get_multi(
            session, 
            skip=skip, 
            limit=pagination.size, 
            filters=filters
        )
        
        return PaginatedResponse.create(
            data=items,
            total=total,
            page=pagination.page,
            size=pagination.size
        )
    
    def update(
        self, 
        session: Session, 
        *, 
        db_obj: ModelType, 
        obj_in: UpdateSchemaType | Dict[str, Any]
    ) -> ModelType:
        """
        Обновление объекта
        
        Аргументы:
            session: Сессия базы данных
            db_obj: Объект для обновления
            obj_in: Данные для обновления
            
        Возвращает:
            ModelType: Обновленный объект
            
        Исключения:
            HTTPException: При ошибке обновления
        """
        try:
            # Получаем данные для обновления
            if isinstance(obj_in, dict):
                update_data = obj_in
            else:
                update_data = obj_in.model_dump(exclude_unset=True)

            # Обновляем поля объекта
            for field, value in update_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)

            # Обновляем updated_at если поле существует
            if hasattr(db_obj, 'updated_at'):
                from datetime import datetime
                db_obj.updated_at = datetime.utcnow()

            return self._extracted_from_update_23(session, db_obj)
        except IntegrityError:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ошибка целостности данных при обновлении объекта"
            )
        except Exception:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Внутренняя ошибка сервера при обновлении объекта"
            )

    # TODO Rename this here and in `create` and `update`
    def _extracted_from_update_23(self, session, db_obj):
        session.add(db_obj)
        session.commit()
        session.refresh(db_obj)
        return db_obj
    
    def delete(self, session: Session, *, id: int) -> ModelType:
        """
        Удаление объекта по ID
        
        Аргументы:
            session: Сессия базы данных
            id: Идентификатор объекта
            
        Возвращает:
            ModelType: Удаленный объект
            
        Исключения:
            HTTPException: При отсутствии объекта или ошибке удаления
        """
        try:
            obj = self.get_or_404(session, id)
            session.delete(obj)
            session.commit()
            return obj
            
        except HTTPException:
            raise
        except Exception:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Внутренняя ошибка сервера при удалении объекта"
            )
    
    def exists(self, session: Session, id: int) -> bool:
        """
        Проверка существования объекта
        
        Аргументы:
            session: Сессия базы данных
            id: Идентификатор объекта
            
        Возвращает:
            bool: True если объект существует
        """
        return session.get(self.model, id) is not None
    
    def count(self, session: Session, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Подсчет количества объектов
        
        Аргументы:
            session: Сессия базы данных
            filters: Фильтры для подсчета
            
        Возвращает:
            int: Количество объектов
        """
        query = select(func.count(self.model.id))
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    query = query.where(getattr(self.model, field) == value)
        
        return session.exec(query).one()