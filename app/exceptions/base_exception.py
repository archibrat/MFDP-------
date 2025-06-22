from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from pydantic import ValidationError
from typing import Union
import logging

logger = logging.getLogger(__name__)

# Кастомные исключения
class BaseAppException(Exception):
    """Базовое исключение приложения"""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationException(BaseAppException):
    """Исключение валидации"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, details)

class AuthenticationException(BaseAppException):
    """Исключение аутентификации"""
    def __init__(self, message: str = "Не авторизован"):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)

class AuthorizationException(BaseAppException):
    """Исключение авторизации"""
    def __init__(self, message: str = "Недостаточно прав"):
        super().__init__(message, status.HTTP_403_FORBIDDEN)

class NotFoundException(BaseAppException):
    """Исключение "не найдено" """
    def __init__(self, message: str = "Ресурс не найден"):
        super().__init__(message, status.HTTP_404_NOT_FOUND)

class ConflictException(BaseAppException):
    """Исключение конфликта"""
    def __init__(self, message: str = "Конфликт данных"):
        super().__init__(message, status.HTTP_409_CONFLICT)

class MLServiceException(BaseAppException):
    """Исключение ML сервиса"""
    def __init__(self, message: str = "Ошибка ML сервиса"):
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)

# Обработчики исключений
async def base_app_exception_handler(request: Request, exc: BaseAppException):
    """Обработчик базовых исключений приложения"""
    logger.error(f"Application error: {exc.message}", extra={
        "status_code": exc.status_code,
        "details": exc.details,
        "path": request.url.path
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "details": exc.details,
            "path": request.url.path
        }
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработчик HTTP исключений"""
    logger.warning(f"HTTP error: {exc.detail}", extra={
        "status_code": exc.status_code,
        "path": request.url.path
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "path": request.url.path
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Обработчик ошибок валидации"""
    logger.warning(f"Validation error: {exc.errors()}", extra={
        "path": request.url.path
    })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "Ошибка валидации данных",
            "details": exc.errors(),
            "path": request.url.path
        }
    )

async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Обработчик ошибок SQLAlchemy"""
    logger.error(f"Database error: {str(exc)}", extra={
        "path": request.url.path
    })
    
    if isinstance(exc, IntegrityError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "success": False,
                "message": "Нарушение целостности данных",
                "path": request.url.path
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Ошибка базы данных",
            "path": request.url.path
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Общий обработчик исключений"""
    logger.error(f"Unexpected error: {str(exc)}", extra={
        "path": request.url.path,
        "exception_type": type(exc).__name__
    }, exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Внутренняя ошибка сервера",
            "path": request.url.path
        }
    )

def register_exception_handlers(app):
    """Регистрация обработчиков исключений"""
    app.add_exception_handler(BaseAppException, base_app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)