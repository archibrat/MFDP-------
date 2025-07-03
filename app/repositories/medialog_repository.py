"""
Оптимизированный репозиторий для работы с данными МИС Медиалог
Использует фабричный генератор условий и оптимизированные запросы
"""

from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from sqlmodel import Session, select, and_, or_
from sqlalchemy import desc, func
import logging

from app.models.medialog import (
    MedialogPatient, Medecin, Consultation, Schedule, Appointment,
    Direction, DataTransfer, NoShowPrediction, ScheduleModel, Planning
)
from app.models.medialog import MedialogPatient as Patient
from app.repositories.base_repository import BaseRepository


class ConditionBuilder:
    """Фабричный генератор условий для SQL-запросов"""
    
    @staticmethod
    def build_conditions(**kwargs) -> List[Callable]:
        """
        Строит список условий на основе переданных параметров
        
        Args:
            **kwargs: Параметры для построения условий
            
        Returns:
            Список функций-условий
        """
        conditions = []
        
        # Условия для пациентов
        if 'age_min' in kwargs and kwargs['age_min'] is not None:
            max_birth_date = datetime.utcnow() - timedelta(days=kwargs['age_min'] * 365)
            conditions.append(lambda: MedialogPatient.birth_date <= max_birth_date)
            
        if 'age_max' in kwargs and kwargs['age_max'] is not None:
            min_birth_date = datetime.utcnow() - timedelta(days=kwargs['age_max'] * 365)
            conditions.append(lambda: MedialogPatient.birth_date >= min_birth_date)
            
        if 'gender' in kwargs and kwargs['gender']:
            conditions.append(lambda: MedialogPatient.gender == kwargs['gender'])
            
        # Условия для дат
        if 'start_date' in kwargs and kwargs['start_date']:
            conditions.append(lambda: Consultation.visit_date >= kwargs['start_date'])
            
        if 'end_date' in kwargs and kwargs['end_date']:
            conditions.append(lambda: Consultation.visit_date <= kwargs['end_date'])
            
        # Условия для врачей
        if 'department' in kwargs and kwargs['department']:
            conditions.append(lambda: Medecin.department == kwargs['department'])
            
        if 'active_only' in kwargs and kwargs['active_only']:
            conditions.append(lambda: Medecin.active_flag == True)
            
        # Условия для записей
        if 'include_no_shows' in kwargs and not kwargs['include_no_shows']:
            conditions.append(lambda: Appointment.no_show_flag == False)
            
        return conditions


class MedialogRepository:
    """Оптимизированный репозиторий для работы с данными МИС Медиалог"""

    def __init__(self, session: Session):
        self.session = session
        self.logger = logging.getLogger(__name__)

    # Методы для работы с пациентами
    def get_patient_by_id(self, patients_id: int) -> Optional[MedialogPatient]:
        """Получение пациента по ID"""
        try:
            stmt = (
                select(MedialogPatient)
                .where(MedialogPatient.patients_id == patients_id)
            )
            return self.session.exec(stmt).first()
        except Exception as e:
            self.logger.error(f"Ошибка получения пациента {patients_id}: {e}")
            return None

    def get_patients_by_criteria(self, 
                                age_min: Optional[int] = None,
                                age_max: Optional[int] = None,
                                gender: Optional[str] = None,
                                limit: int = 100) -> List[MedialogPatient]:
        """Получение пациентов по критериям с использованием фабричного генератора"""
        try:
            stmt = select(MedialogPatient)
            
            # Используем фабричный генератор условий
            conditions = ConditionBuilder.build_conditions(
                age_min=age_min, age_max=age_max, gender=gender
            )
            
            if conditions:
                stmt = stmt.where(and_(*[cond() for cond in conditions]))

            stmt = stmt.limit(limit)
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения пациентов по критериям: {e}")
            return []

    # Методы для работы с врачами
    def get_medecin_by_id(self, medecins_id: int) -> Optional[Medecin]:
        """Получение врача по ID"""
        try:
            stmt = (
                select(Medecin)
                .where(Medecin.medecins_id == medecins_id)
            )
            return self.session.exec(stmt).first()
        except Exception as e:
            self.logger.error(f"Ошибка получения врача {medecins_id}: {e}")
            return None

    def get_medecins_by_department(self, department: str) -> List[Medecin]:
        """Получение врачей по отделению с использованием фабричного генератора"""
        try:
            stmt = select(Medecin)
            
            conditions = ConditionBuilder.build_conditions(
                department=department, active_only=True
            )
            
            if conditions:
                stmt = stmt.where(and_(*[cond() for cond in conditions]))
                
            return list(self.session.exec(stmt).all())
        except Exception as e:
            self.logger.error(f"Ошибка получения врачей отделения {department}: {e}")
            return []

    # Методы для работы с консультациями
    def get_patient_consultations(self, patients_id: int, 
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> List[Consultation]:
        """Получение консультаций пациента с оптимизированным запросом"""
        try:
            stmt = (
                select(Consultation)
                .where(Consultation.patients_id == patients_id)
            )
            
            conditions = ConditionBuilder.build_conditions(
                start_date=start_date, end_date=end_date
            )
            
            if conditions:
                stmt = stmt.where(and_(*[cond() for cond in conditions]))

            stmt = stmt.order_by(Consultation.visit_date.desc())
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения консультаций пациента {patients_id}: {e}")
            return []

    def get_consultations_by_medecin(self, medecins_id: int,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> List[Consultation]:
        """Получение консультаций врача с оптимизированным запросом"""
        try:
            stmt = (
                select(Consultation)
                .where(Consultation.medecins_id == medecins_id)
            )
            
            conditions = ConditionBuilder.build_conditions(
                start_date=start_date, end_date=end_date
            )
            
            if conditions:
                stmt = stmt.where(and_(*[cond() for cond in conditions]))

            stmt = stmt.order_by(Consultation.visit_date.desc())
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения консультаций врача {medecins_id}: {e}")
            return []

    # Методы для работы с расписанием
    def get_schedule_by_medecin(self, medecins_id: int,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> List[Schedule]:
        """Получение расписания врача с оптимизированным запросом"""
        try:
            stmt = (
                select(Schedule)
                .where(Schedule.medecins_id == medecins_id)
            )
            
            conditions = ConditionBuilder.build_conditions(
                start_date=start_date, end_date=end_date
            )
            
            if conditions:
                stmt = stmt.where(and_(*[cond() for cond in conditions]))

            stmt = stmt.order_by(Schedule.date.asc(), Schedule.time_start.asc())
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения расписания врача {medecins_id}: {e}")
            return []

    def get_available_slots(self, medecins_id: int, date: datetime) -> List[Schedule]:
        """Получение доступных слотов врача на дату"""
        try:
            stmt = (
                select(Schedule)
                .where(
                    and_(
                        Schedule.medecins_id == medecins_id,
                        Schedule.date == date,
                        Schedule.slots_booked < Schedule.slots_total,
                        Schedule.status == "active"
                    )
                )
            )
            return list(self.session.exec(stmt).all())
        except Exception as e:
            self.logger.error(f"Ошибка получения доступных слотов: {e}")
            return []

    # Методы для работы с записями
    def get_appointment_by_id(self, appointment_id: int) -> Optional[Appointment]:
        """Получение записи по ID"""
        try:
            stmt = (
                select(Appointment)
                .where(Appointment.appointment_id == appointment_id)
            )
            return self.session.exec(stmt).first()
        except Exception as e:
            self.logger.error(f"Ошибка получения записи {appointment_id}: {e}")
            return None

    def get_patient_appointments(self, patients_id: int,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                include_no_shows: bool = True) -> List[Appointment]:
        """Получение записей пациента с использованием фабричного генератора"""
        try:
            stmt = (
                select(Appointment)
                .where(Appointment.patients_id == patients_id)
            )
            
            conditions = ConditionBuilder.build_conditions(
                start_date=start_date, end_date=end_date, include_no_shows=include_no_shows
            )
            
            if conditions:
                stmt = stmt.where(and_(*[cond() for cond in conditions]))

            stmt = stmt.order_by(Appointment.appointment_time.desc())
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения записей пациента {patients_id}: {e}")
            return []

    def get_appointments_by_date(self, date: datetime, 
                                medecins_id: Optional[int] = None) -> List[Appointment]:
        """Получение записей на дату с оптимизированным JOIN"""
        try:
            stmt = (
                select(Appointment)
                .join(Schedule)
                .where(Schedule.date == date)
            )
            
            if medecins_id:
                stmt = stmt.where(Schedule.medecins_id == medecins_id)

            stmt = stmt.order_by(Appointment.appointment_time.asc())
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения записей на дату {date}: {e}")
            return []

    def get_no_show_appointments(self, start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> List[Appointment]:
        """Получение записей с неявками с использованием фабричного генератора"""
        try:
            stmt = (
                select(Appointment)
                .where(Appointment.no_show_flag == True)
            )
            
            conditions = ConditionBuilder.build_conditions(
                start_date=start_date, end_date=end_date
            )
            
            if conditions:
                stmt = stmt.where(and_(*[cond() for cond in conditions]))

            stmt = stmt.order_by(Appointment.appointment_time.desc())
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения записей с неявками: {e}")
            return []

    # Методы для работы с прогнозами
    def save_no_show_prediction(self, prediction: NoShowPrediction) -> bool:
        """Сохранение прогноза неявки"""
        try:
            self.session.add(prediction)
            self.session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка сохранения прогноза: {e}")
            self.session.rollback()
            return False

    def get_predictions_by_appointment(self, appointment_id: int) -> List[NoShowPrediction]:
        """Получение прогнозов для записи"""
        try:
            stmt = (
                select(NoShowPrediction)
                .where(NoShowPrediction.appointment_id == appointment_id)
                .order_by(NoShowPrediction.prediction_date.desc())
            )
            return list(self.session.exec(stmt).all())
        except Exception as e:
            self.logger.error(f"Ошибка получения прогнозов для записи {appointment_id}: {e}")
            return []

    def get_predictions_by_date_range(self, start_date: datetime,
                                    end_date: datetime) -> List[NoShowPrediction]:
        """Получение прогнозов за период"""
        try:
            stmt = (
                select(NoShowPrediction)
                .where(
                    and_(
                        NoShowPrediction.prediction_date >= start_date,
                        NoShowPrediction.prediction_date <= end_date
                    )
                )
                .order_by(NoShowPrediction.prediction_date.desc())
            )
            return list(self.session.exec(stmt).all())
        except Exception as e:
            self.logger.error(f"Ошибка получения прогнозов за период: {e}")
            return []

    # Методы для аналитики с оптимизированными запросами
    def get_no_show_statistics(self, start_date: datetime,
                              end_date: datetime,
                              group_by: str = "day") -> List[Dict[str, Any]]:
        """Получение статистики неявок с оптимизированным запросом"""
        try:
            # Используем агрегированный запрос для повышения производительности
            if group_by == "day":
                stmt = (
                    select(
                        func.date(Appointment.appointment_time).label('period'),
                        func.count(Appointment.appointment_id).label('total_appointments'),
                        func.sum(func.cast(Appointment.no_show_flag, func.Integer)).label('no_shows')
                    )
                    .where(
                        and_(
                            Appointment.appointment_time >= start_date,
                            Appointment.appointment_time <= end_date
                        )
                    )
                    .group_by(func.date(Appointment.appointment_time))
                    .order_by(func.date(Appointment.appointment_time))
                )
            else:
                # Для других группировок используем стандартный подход
                stmt = select(Appointment).where(
                    and_(
                        Appointment.appointment_time >= start_date,
                        Appointment.appointment_time <= end_date
                    )
                )
                appointments = self.session.exec(stmt).all()
                
                # Группировка данных
                stats = {}
                for appointment in appointments:
                    if group_by == "week":
                        key = appointment.appointment_time.isocalendar()[1]
                    elif group_by == "month":
                        key = (appointment.appointment_time.year, appointment.appointment_time.month)
                    else:
                        key = appointment.appointment_time.date()

                    if key not in stats:
                        stats[key] = {"total": 0, "no_shows": 0}

                    stats[key]["total"] += 1
                    if appointment.no_show_flag:
                        stats[key]["no_shows"] += 1

                # Формирование результата
                result = []
                for key, data in stats.items():
                    result.append({
                        "period": str(key),
                        "total_appointments": data["total"],
                        "no_shows": data["no_shows"],
                        "no_show_rate": data["no_shows"] / data["total"] if data["total"] > 0 else 0
                    })
                return result

            # Выполнение агрегированного запроса
            results = self.session.exec(stmt).all()
            
            # Формирование результата
            result = []
            for row in results:
                result.append({
                    "period": str(row.period),
                    "total_appointments": row.total_appointments,
                    "no_shows": row.no_shows,
                    "no_show_rate": row.no_shows / row.total_appointments if row.total_appointments > 0 else 0
                })

            return result

        except Exception as e:
            self.logger.error(f"Ошибка получения статистики неявок: {e}")
            return []

    def get_patient_risk_profiles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение профилей риска пациентов"""
        try:
            # Получаем пациентов с их историей неявок
            patients = self.get_patients_by_criteria(limit=limit)
            
            risk_profiles = []
            for patient in patients:
                # Получаем историю записей пациента
                appointments = self.get_patient_appointments(
                    patient.patients_id,
                    start_date=datetime.utcnow() - timedelta(days=365),
                    include_no_shows=True
                )
                
                # Рассчитываем метрики риска
                total_appointments = len(appointments)
                no_show_count = len([a for a in appointments if a.no_show_flag])
                no_show_rate = no_show_count / total_appointments if total_appointments > 0 else 0.0
                
                # Получаем последние консультации для анализа поведения
                recent_consultations = self.get_patient_consultations(
                    patient.patients_id,
                    start_date=datetime.utcnow() - timedelta(days=90)
                )
                
                risk_profiles.append({
                    'patient_id': patient.patients_id,
                    'age': patient.age,
                    'gender': patient.gender,
                    'total_appointments': total_appointments,
                    'no_show_count': no_show_count,
                    'no_show_rate': no_show_rate,
                    'recent_consultations': len(recent_consultations),
                    'risk_level': 'high' if no_show_rate > 0.3 else 'medium' if no_show_rate > 0.1 else 'low'
                })
            
            return risk_profiles

        except Exception as e:
            self.logger.error(f"Ошибка получения профилей риска: {e}")
            return []

    def get_schedules_by_date(self, date: datetime) -> List[Schedule]:
        """Получение расписания на конкретную дату"""
        try:
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            stmt = (
                select(Schedule)
                .where(
                    and_(
                        Schedule.date >= start_of_day,
                        Schedule.date < end_of_day
                    )
                )
                .order_by(Schedule.medecins_id, Schedule.time_start)
            )
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения расписания на {date}: {e}")
            return []

    def get_appointments_by_date(self, date: datetime) -> List[Appointment]:
        """Получение записей на конкретную дату"""
        try:
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            stmt = (
                select(Appointment)
                .where(
                    and_(
                        Appointment.appointment_time >= start_of_day,
                        Appointment.appointment_time < end_of_day
                    )
                )
                .order_by(Appointment.appointment_time)
            )
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения записей на {date}: {e}")
            return []

    def get_planning_by_id(self, planning_id: int) -> Optional[Planning]:
        """Получение записи по ID из таблицы PLANNING"""
        try:
            stmt = (
                select(Planning)
                .where(Planning.planning_id == planning_id)
            )
            return self.session.exec(stmt).first()
        except Exception as e:
            self.logger.error(f"Ошибка получения записи {planning_id}: {e}")
            return None

    def get_appointments_by_medecin(self, medecins_id: int, 
                                   start_date: datetime,
                                   end_date: datetime) -> List[Planning]:
        """Получение записей врача за период"""
        try:
            stmt = (
                select(Planning)
                .where(
                    and_(
                        Planning.medecins_creator_id == medecins_id,
                        Planning.date_cons >= start_date,
                        Planning.date_cons <= end_date,
                        Planning.cancelled != 'Y'
                    )
                )
                .order_by(Planning.date_cons)
            )
            return list(self.session.exec(stmt).all())
        except Exception as e:
            self.logger.error(f"Ошибка получения записей врача {medecins_id}: {e}")
            return [] 