"""
Основной файл запуска ML API
Упрощенная версия API только для ML-компонентов
"""

import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Настройка режима разработки для ML API
os.environ["DEVELOPMENT_MODE"] = "true"

from app.routes.ml_production import router as ml_production_router
from app.routes.health import router as health_router
from app.services.logging.logging import get_logger

logger = get_logger(logger_name=__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    logger.info("Запуск ML API...")
    logger.info("ML Production модуль готов к работе")
    yield
    logger.info("Завершение работы ML API")


def create_ml_app() -> FastAPI:
    """
    Создание ML API приложения
    
    Возвращает:
        FastAPI: Экземпляр приложения для ML
    """
    
    app = FastAPI(
        title="MFDP ML Production API",
        description="Medical Flow Data Platform ",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Настройка CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Регистрация маршрутов
    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(ml_production_router, tags=["ML Production"])
    
    @app.get("/")
    async def root():
        """Корень"""
        return {
            "service": "MFDP ML Production API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "ml": "/api/ml"
            }
        }
    
    @app.get("/status")
    async def status():
        """Статус ML сервисов"""
        try:
            return {
                "status": "healthy",
                "services": {
                    "no_show_predictor": "ready",
                    "load_balancer": "ready", 
                    "real_time_scheduler": "ready",
                    "feature_store": "ready",
                    "data_access_layer": "ready"
                },
                "message": "Все ML сервисы готовы к работе"
            }
        except Exception as e:
            logger.error(f"Ошибка проверки статуса: {str(e)}")
            raise HTTPException(status_code=500, detail="Ошибка ML сервисов")
    
    return app


# Создание приложения
ml_app = create_ml_app()

if __name__ == "__main__":
    uvicorn.run(
        "api_ml:ml_app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    ) 