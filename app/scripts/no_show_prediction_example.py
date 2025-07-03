"""
Пример использования модуля прогнозирования неявок пациентов
Демонстрирует основные возможности модуля
"""

import asyncio
from datetime import datetime, timedelta
from sqlmodel import Session, create_engine
import logging

from services.no_show_prediction_service import (
    NoShowPredictionService, MedialogDatabaseConnector
)
from repositories.medialog_repository import MedialogRepository
from database.database import get_session
from models.medialog import (
    MedialogPatient, Medecin, Consultation, Schedule, Appointment, NoShowPrediction
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Основная функция примера"""
    
    logger.info("Запуск примера модуля прогнозирования неявок")
    
    try:
        # Получение сессии базы данных
        session = next(get_session())
        
        # Инициализация компонентов
        db_connector = MedialogDatabaseConnector(session)
        prediction_service = NoShowPredictionService(db_connector)
        repository = MedialogRepository(session)
        
        # Шаг 1: Обучение модели
        logger.info("Шаг 1: Обучение модели на исторических данных")
        try:
            metrics = prediction_service.train_model(days_back=365)
            logger.info(f"Модель обучена успешно:")
            logger.info(f"  - Точность: {metrics['accuracy']:.3f}")
            logger.info(f"  - Precision: {metrics['precision']:.3f}")
            logger.info(f"  - Recall: {metrics['recall']:.3f}")
            logger.info(f"  - F1-Score: {metrics['f1_score']:.3f}")
        except Exception as e:
            logger.warning(f"Ошибка обучения модели: {e}")
            logger.info("Продолжаем с предобученной моделью")
        
        # Шаг 2: Получение списка записей для прогнозирования
        logger.info("Шаг 2: Получение записей для прогнозирования")
        
        # Получаем записи на завтра
        tomorrow = datetime.utcnow() + timedelta(days=1)
        appointments = repository.get_appointments_by_date(tomorrow)
        
        if not appointments:
            logger.info("Нет записей на завтра, создаем тестовые данные")
            appointments = create_test_appointments(session)
        
        logger.info(f"Найдено {len(appointments)} записей для прогнозирования")
        
        # Шаг 3: Прогнозирование для каждой записи
        logger.info("Шаг 3: Выполнение прогнозирования")
        
        predictions = []
        for appointment in appointments[:10]:  # Ограничиваем 10 записями для примера
            try:
                result = prediction_service.predict_no_show(appointment.appointment_id)
                if result:
                    predictions.append(result)
                    
                    # Сохранение прогноза в БД
                    prediction_service.save_prediction_to_db(result)
                    
                    logger.info(f"Запись {appointment.appointment_id}: "
                              f"риск {result.risk_level} "
                              f"(вероятность: {result.no_show_probability:.3f})")
                    
            except Exception as e:
                logger.error(f"Ошибка прогнозирования для записи {appointment.appointment_id}: {e}")
        
        # Шаг 4: Анализ результатов
        logger.info("Шаг 4: Анализ результатов прогнозирования")
        
        if predictions:
            risk_distribution = {}
            for pred in predictions:
                risk_level = pred.risk_level
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            
            logger.info("Распределение по уровням риска:")
            for risk_level, count in risk_distribution.items():
                percentage = (count / len(predictions)) * 100
                logger.info(f"  - {risk_level}: {count} записей ({percentage:.1f}%)")
            
            # Средняя вероятность неявки
            avg_probability = sum(p.no_show_probability for p in predictions) / len(predictions)
            logger.info(f"Средняя вероятность неявки: {avg_probability:.3f}")
        
        # Шаг 5: Получение статистики
        logger.info("Шаг 5: Получение статистики неявок")
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        try:
            statistics = repository.get_no_show_statistics(start_date, end_date, "day")
            logger.info(f"Статистика за последние 30 дней:")
            
            if statistics:
                total_appointments = sum(s["total_appointments"] for s in statistics)
                total_no_shows = sum(s["no_shows"] for s in statistics)
                overall_no_show_rate = total_no_shows / total_appointments if total_appointments > 0 else 0
                
                logger.info(f"  - Всего записей: {total_appointments}")
                logger.info(f"  - Всего неявок: {total_no_shows}")
                logger.info(f"  - Общий процент неявок: {overall_no_show_rate:.3f}")
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
        
        # Шаг 6: Получение профилей риска
        logger.info("Шаг 6: Получение профилей риска пациентов")
        
        try:
            risk_profiles = repository.get_patient_risk_profiles(limit=20)
            logger.info(f"Получено {len(risk_profiles)} профилей риска")
            
            if risk_profiles:
                high_risk_patients = [p for p in risk_profiles if p["risk_level"] == "high"]
                logger.info(f"Пациентов с высоким риском: {len(high_risk_patients)}")
                
                if high_risk_patients:
                    logger.info("Примеры пациентов с высоким риском:")
                    for patient in high_risk_patients[:3]:
                        logger.info(f"  - {patient['name']} (ID: {patient['patient_id']}): "
                                  f"{patient['no_show_rate']:.3f}")
        
        except Exception as e:
            logger.error(f"Ошибка получения профилей риска: {e}")
        
        logger.info("Пример завершен успешно")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения примера: {e}")
        raise
    finally:
        session.close()


def create_test_appointments(session: Session) -> list:
    """Создание тестовых записей для демонстрации"""
    
    # Создаем тестового пациента
    patient = MedialogPatient(
        patients_id=99999,
        surname="Тестовый",
        name="Пациент",
        birth_date=datetime(1985, 5, 15),
        gender="M",
        phone="+79001234567",
        email="test@example.com"
    )
    
    # Создаем тестового врача
    medecin = Medecin(
        medecins_id=88888,
        surname="Тестовый",
        name="Врач",
        speciality="Терапевт",
        department="Терапевтическое отделение",
        position="Врач-терапевт"
    )
    
    # Создаем расписание
    schedule = Schedule(
        schedule_id=77777,
        medecins_id=medecin.medecins_id,
        date=datetime.utcnow() + timedelta(days=1),
        time_start=datetime.utcnow() + timedelta(days=1, hours=9),
        time_end=datetime.utcnow() + timedelta(days=1, hours=17),
        cabinet="101",
        slots_total=20,
        slots_booked=5
    )
    
    # Создаем записи
    appointments = []
    for i in range(5):
        appointment = Appointment(
            appointment_id=66666 + i,
            patients_id=patient.patients_id,
            schedule_id=schedule.schedule_id,
            appointment_time=datetime.utcnow() + timedelta(days=1, hours=9 + i),
            visit_type="consultation",
            status="confirmed"
        )
        appointments.append(appointment)
    
    # Сохраняем в БД
    try:
        session.add(patient)
        session.add(medecin)
        session.add(schedule)
        for appointment in appointments:
            session.add(appointment)
        session.commit()
        logger.info("Созданы тестовые данные")
    except Exception as e:
        logger.warning(f"Ошибка создания тестовых данных: {e}")
        session.rollback()
    
    return appointments


def demonstrate_batch_prediction():
    """Демонстрация массового прогнозирования"""
    
    logger.info("Демонстрация массового прогнозирования")
    
    try:
        session = next(get_session())
        db_connector = MedialogDatabaseConnector(session)
        prediction_service = NoShowPredictionService(db_connector)
        
        # Список ID записей для прогнозирования
        appointment_ids = [12345, 12346, 12347, 12348, 12349]
        
        # Массовое прогнозирование
        results = prediction_service.batch_predict(appointment_ids)
        
        logger.info(f"Выполнено прогнозирование для {len(results)} записей")
        
        # Анализ результатов
        high_risk_count = sum(1 for r in results if r.risk_level == "high")
        medium_risk_count = sum(1 for r in results if r.risk_level == "medium")
        low_risk_count = sum(1 for r in results if r.risk_level == "low")
        
        logger.info(f"Распределение рисков:")
        logger.info(f"  - Высокий: {high_risk_count}")
        logger.info(f"  - Средний: {medium_risk_count}")
        logger.info(f"  - Низкий: {low_risk_count}")
        
        # Рекомендации для высокого риска
        high_risk_results = [r for r in results if r.risk_level == "high"]
        if high_risk_results:
            logger.info("Рекомендации для записей с высоким риском:")
            for result in high_risk_results:
                logger.info(f"  - Запись {result.appointment_id}: {result.recommendation}")
        
    except Exception as e:
        logger.error(f"Ошибка массового прогнозирования: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    # Запуск основного примера
    asyncio.run(main())
    
    # Демонстрация массового прогнозирования
    demonstrate_batch_prediction() 