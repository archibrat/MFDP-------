"""
Коннектор к базе данных для модуля оптимизации нагрузки врачей
Адаптирован под схему БД Медиалог
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlmodel import Session, select, and_, func
import pandas as pd
import logging

from app.models.medialog import Medecins, Planning, Motconsu, DirAnsw, Patients
from app.services.doctor_load_optimization.schemas import DoctorProfile


class DatabaseConnector(ABC):
    """Абстрактный коннектор к базе данных"""

    @abstractmethod
    def get_doctor_data(self, doctor_id: int) -> Optional[DoctorProfile]:
        """Получение данных врача"""
        pass

    @abstractmethod
    def get_doctors_by_criteria(self, specialities: Optional[List[str]] = None,
                               departments: Optional[List[str]] = None,
                               active_only: bool = True) -> List[DoctorProfile]:
        """Получение врачей по критериям"""
        pass

    @abstractmethod
    def get_schedule_data(self, doctor_id: int, start_date: datetime,
                         end_date: datetime) -> List[Planning]:
        """Получение данных расписания"""
        pass

    @abstractmethod
    def get_appointment_data(self, doctor_id: int, start_date: datetime,
                           end_date: datetime) -> List[Planning]:
        """Получение данных приемов"""
        pass

    @abstractmethod
    def get_consultation_data(self, doctor_id: int, start_date: datetime,
                            end_date: datetime) -> List[Motconsu]:
        """Получение данных консультаций"""
        pass

    @abstractmethod
    def get_historical_metrics(self, days_back: int = 30) -> pd.DataFrame:
        """Получение исторических метрик"""
        pass


class MedialogDatabaseConnector(DatabaseConnector):
    """Коннектор к базе данных МИС Медиалог"""

    def __init__(self, session: Session):
        self.session = session
        self.logger = logging.getLogger(__name__)

    def get_doctor_data(self, doctor_id: int) -> Optional[DoctorProfile]:
        """Получение данных врача из таблицы MEDECINS"""
        try:
            stmt = (
                select(Medecins)
                .where(Medecins.medecins_id == doctor_id)
            )
            doctor = self.session.exec(stmt).first()
            
            if not doctor:
                return None

            # Получение метрик врача
            metrics = self._calculate_doctor_metrics(doctor_id)
            
            return DoctorProfile(
                doctor_id=doctor.medecins_id,
                speciality=doctor.specialisation or 'general',
                department=doctor.fm_dep_id or 'unknown',
                current_load=metrics.get('current_load', 0.0),
                avg_wait_time=metrics.get('avg_wait_time', 0.0),
                utilization_rate=metrics.get('utilization_rate', 0.0),
                patient_satisfaction=metrics.get('patient_satisfaction', 0.5),
                complexity_score=metrics.get('complexity_score', 0.5),
                experience_years=metrics.get('experience_years', 5),
                max_patients_per_day=metrics.get('max_patients_per_day', 20)
            )

        except Exception as e:
            self.logger.error(f"Ошибка получения данных врача {doctor_id}: {e}")
            return None

    def get_doctors_by_criteria(self, specialities: Optional[List[str]] = None,
                               departments: Optional[List[str]] = None,
                               active_only: bool = True) -> List[DoctorProfile]:
        """Получение врачей по критериям"""
        try:
            stmt = select(Medecins)
            
            conditions = []
            if active_only:
                conditions.append(Medecins.archive != 'Y')
            if specialities:
                conditions.append(Medecins.specialisation.in_(specialities))
            if departments:
                conditions.append(Medecins.fm_dep_id.in_(departments))
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            doctors = self.session.exec(stmt).all()
            
            doctor_profiles = []
            for doctor in doctors:
                profile = self.get_doctor_data(doctor.medecins_id)
                if profile:
                    doctor_profiles.append(profile)
            
            return doctor_profiles

        except Exception as e:
            self.logger.error(f"Ошибка получения врачей по критериям: {e}")
            return []

    def get_schedule_data(self, doctor_id: int, start_date: datetime,
                         end_date: datetime) -> List[Planning]:
        """Получение данных расписания из таблицы PLANNING"""
        try:
            stmt = (
                select(Planning)
                .where(
                    and_(
                        Planning.medecins_creator_id == doctor_id,
                        Planning.date_cons >= start_date,
                        Planning.date_cons <= end_date,
                        Planning.cancelled != 'Y'
                    )
                )
                .order_by(Planning.date_cons, Planning.heure)
            )
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения расписания врача {doctor_id}: {e}")
            return []

    def get_appointment_data(self, doctor_id: int, start_date: datetime,
                           end_date: datetime) -> List[Planning]:
        """Получение данных приемов из таблицы PLANNING"""
        try:
            stmt = (
                select(Planning)
                .where(
                    and_(
                        Planning.medecins_creator_id == doctor_id,
                        Planning.date_cons >= start_date,
                        Planning.date_cons <= end_date,
                        Planning.cancelled != 'Y'
                    )
                )
                .order_by(Planning.date_cons)
            )
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения приемов врача {doctor_id}: {e}")
            return []

    def get_consultation_data(self, doctor_id: int, start_date: datetime,
                            end_date: datetime) -> List[Motconsu]:
        """Получение данных консультаций из таблицы MOTCONSU"""
        try:
            stmt = (
                select(Motconsu)
                .where(
                    and_(
                        Motconsu.medecins_id == doctor_id,
                        Motconsu.date_consultation >= start_date,
                        Motconsu.date_consultation <= end_date
                    )
                )
                .order_by(Motconsu.date_consultation)
            )
            return list(self.session.exec(stmt).all())

        except Exception as e:
            self.logger.error(f"Ошибка получения консультаций врача {doctor_id}: {e}")
            return []

    def get_historical_metrics(self, days_back: int = 30) -> pd.DataFrame:
        """Получение исторических метрик"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Получение данных о приемах
            stmt = (
                select(
                    Planning.planning_id,
                    Planning.date_cons,
                    Planning.motif,
                    Planning.cancelled,
                    Planning.medecins_creator_id,
                    Medecins.specialisation,
                    Medecins.fm_dep_id
                )
                .join(Medecins, Planning.medecins_creator_id == Medecins.medecins_id)
                .where(
                    and_(
                        Planning.date_cons >= start_date,
                        Planning.date_cons <= end_date
                    )
                )
            )
            
            results = self.session.exec(stmt).all()
            
            # Преобразование в DataFrame
            data = []
            for result in results:
                data.append({
                    'planning_id': result.planning_id,
                    'date_cons': result.date_cons,
                    'motif': result.motif,
                    'cancelled': result.cancelled,
                    'doctor_id': result.medecins_creator_id,
                    'speciality': result.specialisation,
                    'department': result.fm_dep_id
                })
            
            return pd.DataFrame(data)

        except Exception as e:
            self.logger.error(f"Ошибка получения исторических метрик: {e}")
            return pd.DataFrame()

    def _calculate_doctor_metrics(self, doctor_id: int) -> Dict[str, Any]:
        """Расчет метрик врача"""
        try:
            # Период для расчета метрик (последние 30 дней)
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            # Получение данных
            appointments = self.get_appointment_data(doctor_id, start_date, end_date)
            consultations = self.get_consultation_data(doctor_id, start_date, end_date)
            
            # Расчет метрик
            metrics = {}
            
            # Текущая загруженность (количество записей в день)
            if appointments:
                # Группировка по дням
                daily_appointments = {}
                for app in appointments:
                    day = app.date_cons.date()
                    if day not in daily_appointments:
                        daily_appointments[day] = []
                    daily_appointments[day].append(app)
                
                avg_daily_appointments = sum(len(apps) for apps in daily_appointments.values()) / len(daily_appointments)
                max_patients_per_day = max(len(apps) for apps in daily_appointments.values()) if daily_appointments else 20
                
                # Нормализация загрузки (предполагаем норму 20 пациентов в день)
                metrics['current_load'] = min(1.0, avg_daily_appointments / 20)
                metrics['max_patients_per_day'] = max_patients_per_day
            else:
                metrics['current_load'] = 0.0
                metrics['max_patients_per_day'] = 20
            
            # Среднее время ожидания (упрощенный расчет)
            if appointments:
                # Группировка по дням для расчета времени ожидания
                daily_appointments = {}
                for app in appointments:
                    day = app.date_cons.date()
                    if day not in daily_appointments:
                        daily_appointments[day] = []
                    daily_appointments[day].append(app)
                
                wait_times = []
                for day, day_apps in daily_appointments.items():
                    sorted_apps = sorted(day_apps, key=lambda x: x.date_cons)
                    for i in range(1, len(sorted_apps)):
                        wait_time = (sorted_apps[i].date_cons - sorted_apps[i-1].date_cons).total_seconds() / 60
                        wait_times.append(wait_time)
                
                metrics['avg_wait_time'] = sum(wait_times) / len(wait_times) if wait_times else 0.0
            else:
                metrics['avg_wait_time'] = 0.0
            
            # Коэффициент использования времени
            if consultations:
                # Расчет использованного времени на основе консультаций
                total_consultation_time = sum(
                    int(c.cons_duration or 30) for c in consultations
                )
                
                # Предполагаем 8-часовой рабочий день
                work_days = len(set(c.date_consultation.date() for c in consultations))
                total_available_minutes = work_days * 8 * 60
                
                metrics['utilization_rate'] = min(1.0, total_consultation_time / total_available_minutes) if total_available_minutes > 0 else 0.0
            else:
                metrics['utilization_rate'] = 0.0
            
            # Оценка удовлетворенности (упрощенная)
            metrics['patient_satisfaction'] = 0.7  # Заглушка
            
            # Сложность случаев (упрощенная)
            if consultations:
                complexity_scores = []
                for c in consultations:
                    # Простая логика определения сложности по типу приема
                    if 'консультация' in (c.vid_priema or '').lower():
                        complexity_scores.append(0.3)
                    elif 'обследование' in (c.vid_priema or '').lower():
                        complexity_scores.append(0.6)
                    else:
                        complexity_scores.append(0.5)
                metrics['complexity_score'] = sum(complexity_scores) / len(complexity_scores)
            else:
                metrics['complexity_score'] = 0.5
            
            # Дополнительные метрики
            metrics['experience_years'] = 5  # Заглушка
            
            return metrics

        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик врача {doctor_id}: {e}")
            return {
                'current_load': 0.0,
                'avg_wait_time': 0.0,
                'utilization_rate': 0.0,
                'patient_satisfaction': 0.5,
                'complexity_score': 0.5,
                'experience_years': 5,
                'max_patients_per_day': 20
            } 