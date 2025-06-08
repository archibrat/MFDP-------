from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.home import home_route
from routes.user import user_route
from routes.event import event_route
from routes.auth import auth_route
from routes.ml import ml_route
from database.initdb import init_db
from database.config import get_settings
from api_analytics.fastapi import Analytics
from services.logging.logging import get_logger
import uvicorn

logger = get_logger(logger_name=__name__)
settings = get_settings()

def create_application() -> FastAPI:
    """
    Создание и конфигурация FastAPI приложения.
    
    Возвращает:
        FastAPI: Настроенный экземпляр приложения
    """
    
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    app.add_middleware(Analytics, api_key="39d40b20-6328-4a67-ae74-940f0cab5737")  # Добавление промежуточного слоя

    # Настройка CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Регистрация маршрутов
    app.include_router(home_route, tags=['Home'])
    app.include_router(auth_route, prefix='/auth', tags=['Auth'])
    app.include_router(ml_route, prefix='/api/ml', tags=['ML'])
    app.include_router(user_route, prefix='/api/users', tags=['Users'])
    app.include_router(event_route, prefix='/api/events', tags=['Events'])

    return app

app = create_application()

@app.on_event("startup") 
def on_startup():
    try:
        logger.info("Инициализация базы данных...")
        init_db(drop_all=True)
        logger.info("Запуск приложения успешно завершен")
    except Exception as e:
        logger.error(f"Ошибка при запуске: {str(e)}")
        raise
    
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