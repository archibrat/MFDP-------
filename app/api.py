from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from app.routes.home import home_route
from app.routes.user import user_route
from app.routes.event import event_route
from app.routes.auth import auth_route
from app.routes.ml import ml_route
from app.routes.analytics import analytics_router
from app.routes.prediction import prediction_route
from app.routes.optimization import optimization_router
from app.routes.import_export import import_export_router
from app.routes.audit import audit_router
from app.routes.notifications import notifications_router
from app.routes.reports import reports_router
from app.routes.no_show_prediction import router as no_show_prediction_router
from app.routes.doctor_load_optimization import router as doctor_load_optimization_router
from app.routes.real_time_scheduler import router as real_time_scheduler_router
from app.routes.ml_production import router as ml_production_router
from app.database.initdb import init_db, run_init
from app.database.config import get_settings
from app.services.logging.logging import get_logger
import uvicorn
import os

logger = get_logger(logger_name=__name__)
settings = get_settings()

# Режим разработки без базы данных
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"

def create_application() -> FastAPI:
    """
    Создание и конфигурация FastAPI приложения.
    
    Возвращает:
        FastAPI: Настроенный экземпляр приложения
    """
    
    app = FastAPI(
        title="MFDP API",
        description="Medical Flow Data Platform API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
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
    from app.routes.health import router as health_router
    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(home_route, tags=['Home'])
    app.include_router(auth_route, prefix='/auth', tags=['Auth'])
    app.include_router(ml_route, prefix='/api/ml', tags=['ML'])
    app.include_router(user_route, prefix='/api/users', tags=['Users'])
    app.include_router(event_route, prefix='/api/events', tags=['Events'])
    app.include_router(analytics_router, prefix='/api/analytics', tags=['Analytics'])
    app.include_router(prediction_route, prefix="/predict", tags=["prediction"])
    app.include_router(optimization_router, prefix="/optimization", tags=["optimization"])
    app.include_router(import_export_router, prefix="/integration", tags=["integration"])
    app.include_router(audit_router, prefix="/audit", tags=["audit"])
    app.include_router(notifications_router, prefix="/notifications", tags=["notifications"])
    app.include_router(reports_router, prefix="/reports", tags=["reports"])
    app.include_router(no_show_prediction_router)
    app.include_router(doctor_load_optimization_router)
    app.include_router(real_time_scheduler_router)
    app.include_router(ml_production_router)

    return app

app = create_application()

@app.on_event("startup") 
async def on_startup():
    try:
        logger.info("Запуск приложения...")
        
        if not DEVELOPMENT_MODE:
            logger.info("Инициализация базы данных...")
            await init_db(drop_all=False)
            logger.info("База данных инициализирована")
        else:
            logger.info("Режим разработки - база данных отключена")
            
        logger.info("Запуск приложения успешно завершен")
    except Exception as e:
        logger.error(f"Ошибка при запуске: {str(e)}")
        if not DEVELOPMENT_MODE:
            logger.error("Для работы без БД установите переменную окружения DEVELOPMENT_MODE=true")
            raise
        else:
            logger.warning("Продолжаем работу в режиме разработки без БД")
    
@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении работы приложения."""
    logger.info("Завершение работы приложения...")

if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host='0.0.0.0',
        port=8080,
        reload=True,
        log_level="info"
    )
