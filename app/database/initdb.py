"""
Инициализация базы данных PostgreSQL
"""

import asyncio
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database.database import engine, init_database, get_session
from app.database.config import get_settings

logger = logging.getLogger(__name__)


async def create_initial_data():
    """Создание начальных данных в базе данных"""
    try:
        session = next(get_session())
        
        # Проверяем, существуют ли уже данные
        result = session.execute(text("SELECT COUNT(*) FROM users"))
        user_count = result.scalar()
        
        if user_count == 0:
            logger.info("Создание начальных данных...")
            
            # Создание администратора
            session.execute(text("""
                INSERT INTO users (username, email, full_name, hashed_password, role, is_active, created_at)
                VALUES (
                    'admin',
                    'admin@mfdp.local',
                    'Системный администратор',
                    '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',
                    'admin',
                    true,
                    NOW()
                )
            """))
            
            # Создание демо-врача
            session.execute(text("""
                INSERT INTO users (username, email, full_name, hashed_password, role, is_active, created_at)
                VALUES (
                    'doctor1',
                    'doctor@mfdp.local',
                    'Доктор Иванов',
                    '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',
                    'doctor',
                    true,
                    NOW()
                )
            """))
            
            # Создание демо-пациентов
            session.execute(text("""
                INSERT INTO patients (patient_id, gender, age, neighbourhood, scholarship, hypertension, diabetes, alcoholism, handcap)
                VALUES 
                    (12345, 'F', 45, 'Central', false, true, false, false, 0),
                    (12346, 'M', 62, 'North', false, false, true, false, 0),
                    (12347, 'F', 38, 'South', true, false, false, false, 0),
                    (12348, 'M', 55, 'East', false, true, true, false, 1),
                    (12349, 'F', 29, 'West', false, false, false, false, 0)
            """))
            
            # Создание демо-событий
            session.execute(text("""
                INSERT INTO events (event_type, patient_id, scheduled_time, description, priority, status, sms_received)
                VALUES 
                    ('appointment', 1, '2025-06-25 09:00:00', 'Плановый прием терапевта', 'medium', 'scheduled', true),
                    ('consultation', 2, '2025-06-25 10:30:00', 'Консультация кардиолога', 'high', 'scheduled', true),
                    ('examination', 3, '2025-06-25 14:00:00', 'Общее обследование', 'low', 'scheduled', false),
                    ('operation', 4, '2025-06-26 08:00:00', 'Плановая операция', 'critical', 'confirmed', true),
                    ('appointment', 5, '2025-06-26 11:00:00', 'Повторный прием', 'medium', 'scheduled', true)
            """))
            
            session.commit()
            logger.info("Начальные данные созданы успешно")
            
        else:
            logger.info("Данные уже существуют, пропускаем инициализацию")
            
    except Exception as e:
        logger.error(f"Ошибка создания начальных данных: {e}")
        session.rollback()
        raise
    finally:
        session.close()


async def init_db(drop_all: bool = False):
    """
    Инициализация базы данных
    
    Args:
        drop_all: Если True, удаляет все таблицы перед созданием
    """
    try:
        logger.info("Начинаем инициализацию базы данных...")
        
        if drop_all:
            logger.warning("Удаление всех таблиц...")
            with engine.connect() as conn:
                conn.execute(text("DROP SCHEMA public CASCADE"))
                conn.execute(text("CREATE SCHEMA public"))
                conn.commit()
        
        # Создание таблиц
        await init_database()
        
        # Создание начальных данных
        await create_initial_data()
        
        logger.info("Инициализация базы данных завершена успешно")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {e}")
        raise


def run_init():
    """Синхронная обертка для запуска инициализации"""
    asyncio.run(init_db())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_init()
