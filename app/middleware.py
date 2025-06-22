from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api_analytics.fastapi import Analytics
from database.config import get_settings
import time
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

def setup_cors_middleware(app: FastAPI):
    """Настройка CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS if hasattr(settings, 'ALLOWED_ORIGINS') else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class LoggingMiddleware:
    """Middleware для логирования запросов"""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        # Логируем входящий запрос
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Логируем ответ
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {process_time:.4f}s"
            )
            
            # Добавляем заголовок с временем обработки
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Time: {process_time:.4f}s"
            )
            raise

def setup_logging_middleware(app: FastAPI):
    """Настройка Logging middleware"""
    app.add_middleware(LoggingMiddleware)

def setup_all_middleware(app: FastAPI):
    """Настройка всех middleware"""
    setup_logging_middleware(app)
    setup_cors_middleware(app)
