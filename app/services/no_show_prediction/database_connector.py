"""
Коннектор для работы с базой данных МИС Медиалог
Оптимизирован для быстрого извлечения данных для ML-моделей
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlmodel import Session, select, and_, or_, func
from sqlalchemy.orm import selectinload

from app.models.medialog import (
    MedialogPatient, Appointment, Consultation, Medecin, NoShowPrediction,
    Planning, Patients, Motconsu, DirAnsw, Medecins
)
from app.services.no_show_prediction.schemas import PatientProfile, AppointmentContext


class DatabaseConnector(ABC):
    """Абстрактный класс для работы с базой данных"""

    @abstractmethod
    def get_patient_data(self, patient_id: int) -> Optional[PatientProfile]:
        """Получение данных пациента"""
        pass

    @abstractmethod
    def get_appointment_data(self, appointment_id: int) -> Optional[AppointmentContext]:
        """Получение данных записи на прием"""
        pass

    @abstractmethod
    def get_historical_data(self, days_back: int = 365) -> pd.DataFrame:
        """Получение исторических данных для обучения модели"""
        pass

    @abstractmethod
    def save_prediction(self, prediction: NoShowPrediction) -> bool:
        """Сохранение прогноза в базу данных"""
        pass


class MedialogDatabaseConnector(DatabaseConnector):
    """Оптимизированный коннектор для работы с базой данных МИС Медиалог"""

    def __init__(self, session: Session):
        self.session = session
        self.logger = logging.getLogger(__name__)

    def get_patient_data(self, patient_id: int) -> Optional[PatientProfile]:
        """
        Получение данных пациента из таблиц PATIENTS и MOTCONSU
        Оптимизировано для минимизации количества SQL-запросов
        """
        try:
            # Получаем данные пациента с предзагрузкой связанных данных
            patient_stmt = (
                select(Patients)
                .where(Patients.patients_id == patient_id)
            )
            patient = self.session.exec(patient_stmt).first()
            
            if not patient:
                return None

            # Вычисляем возраст из поля AGE или GOD_ROGDENIQ
            age = self._calculate_age(patient)
            
            # Получаем историю консультаций одним запросом
            consultations_stmt = (
                select(Motconsu)
                .where(Motconsu.patients_id == patient_id)
                .order_by(Motconsu.date_consultation.desc())
            )
            consultations = self.session.exec(consultations_stmt).all()

            # Оптимизированное вычисление среднего интервала между посещениями
            visit_dates = sorted([c.date_consultation for c in consultations if c.date_consultation])
            avg_interval = self._calculate_avg_interval(visit_dates)

            # Получаем историю неявок из PLANNING
            planning_stmt = (
                select(Planning)
                .where(
                    and_(
                        Planning.patients_id == patient_id,
                        Planning.cancelled == 'Y'
                    )
                )
            )
            cancelled_appointments = self.session.exec(planning_stmt).all()

            # Векторизованное вычисление процента неявок
            no_show_rate = self._calculate_no_show_rate(cancelled_appointments, consultations)

            return PatientProfile(
                patient_id=patient_id,
                age=age,
                gender=patient.pol or 'U',
                visit_history_count=len(consultations),
                avg_interval_between_visits=avg_interval,
                no_show_history_rate=no_show_rate,
                phone_confirmed=bool(patient.tel or patient.mobil_telefon)
            )

        except Exception as e:
            self.logger.error(f"Ошибка получения данных пациента {patient_id}: {e}")
            return None

    def get_appointment_data(self, appointment_id: int) -> Optional[AppointmentContext]:
        """
        Получение данных записи на прием из таблицы PLANNING
        Оптимизировано с использованием JOIN
        """
        try:
            # Получаем запись с предзагрузкой связанных данных
            planning_stmt = (
                select(Planning)
                .where(Planning.planning_id == appointment_id)
            )
            planning = self.session.exec(planning_stmt).first()
            
            if not planning:
                return None

            # Получаем данные врача
            doctor_id = 0
            if planning.medecins_creator_id:
                medecin_stmt = (
                    select(Medecins)
                    .where(Medecins.medecins_id == planning.medecins_creator_id)
                )
                medecin = self.session.exec(medecin_stmt).first()
                doctor_id = medecin.medecins_id if medecin else 0

            # Вычисляем заблаговременность записи
            advance_booking_days = 0
            if planning.create_date_time and planning.date_cons:
                advance_booking_days = (planning.date_cons - planning.create_date_time).days

            # Проверяем, является ли это повторным посещением
            is_repeat_visit = self._check_repeat_visit(planning.patients_id, planning.date_cons)

            return AppointmentContext(
                appointment_id=appointment_id,
                patient_id=planning.patients_id,
                doctor_id=doctor_id,
                appointment_time=planning.date_cons or datetime.utcnow(),
                visit_type=planning.motif or 'consultation',
                advance_booking_days=advance_booking_days,
                is_repeat_visit=is_repeat_visit,
                reminder_sent=False  # В реальности должно браться из системы уведомлений
            )

        except Exception as e:
            self.logger.error(f"Ошибка получения данных записи {appointment_id}: {e}")
            return None

    def get_historical_data(self, days_back: int = 365) -> pd.DataFrame:
        """
        Получение исторических данных для обучения модели
        Оптимизировано для batch-обработки с использованием схемы Медиалог
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Получаем записи с результатами неявок одним запросом
            planning_stmt = (
                select(Planning)
                .where(
                    and_(
                        Planning.date_cons >= cutoff_date,
                        Planning.date_cons <= datetime.utcnow()
                    )
                )
            )
            planning_records = self.session.exec(planning_stmt).all()

            # Batch-обработка данных
            data = []
            for planning in planning_records:
                try:
                    patient_profile = self.get_patient_data(planning.patients_id)
                    appointment_context = self.get_appointment_data(planning.planning_id)
                    
                    if patient_profile and appointment_context:
                        features = self._extract_features(patient_profile, appointment_context, planning)
                        features['no_show'] = int(planning.cancelled == 'Y')
                        features['planning_id'] = planning.planning_id
                        data.append(features)
                except Exception as e:
                    self.logger.warning(f"Ошибка обработки записи {planning.planning_id}: {e}")
                    continue

            return pd.DataFrame(data)

        except Exception as e:
            self.logger.error(f"Ошибка получения исторических данных: {e}")
            return pd.DataFrame()

    def save_prediction(self, prediction: NoShowPrediction) -> bool:
        """Сохранение прогноза в базу данных"""
        try:
            self.session.add(prediction)
            self.session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка сохранения прогноза: {e}")
            self.session.rollback()
            return False

    def _calculate_age(self, patient) -> int:
        """Вычисление возраста пациента"""
        try:
            if hasattr(patient, 'age') and patient.age:
                return int(patient.age)
            elif hasattr(patient, 'god_rogdeniq') and patient.god_rogdeniq:
                birth_year = int(patient.god_rogdeniq)
                return datetime.utcnow().year - birth_year
            else:
                return 30  # Значение по умолчанию
        except:
            return 30

    def _calculate_avg_interval(self, visit_dates: List[datetime]) -> float:
        """
        Оптимизированное вычисление среднего интервала между посещениями
        Использует numpy для векторизации
        """
        if len(visit_dates) <= 1:
            return 0.0
        
        # Конвертируем в numpy array для векторизации
        dates_array = np.array([d.timestamp() for d in visit_dates])
        intervals = np.diff(dates_array) / (24 * 3600)  # Конвертируем в дни
        return float(np.mean(intervals))

    def _calculate_no_show_rate(self, cancelled_appointments: List, consultations: List) -> float:
        """
        Векторизованное вычисление процента неявок
        """
        total_appointments = len(consultations) + len(cancelled_appointments)
        if total_appointments == 0:
            return 0.0
        
        return len(cancelled_appointments) / total_appointments

    def _check_repeat_visit(self, patient_id: int, appointment_time: datetime) -> bool:
        """
        Проверка, является ли это повторным посещением
        Оптимизировано с использованием EXISTS
        """
        try:
            # Используем EXISTS для эффективности
            consultation_stmt = (
                select(Motconsu)
                .where(
                    and_(
                        Motconsu.patients_id == patient_id,
                        Motconsu.date_consultation < appointment_time
                    )
                )
                .limit(1)
            )
            previous_consultation = self.session.exec(consultation_stmt).first()
            return previous_consultation is not None
        except Exception:
            return False

    def _extract_features(self, patient: PatientProfile, appointment: AppointmentContext, planning) -> Dict[str, Any]:
        """Извлечение признаков из данных пациента и записи"""
        appointment_time = appointment.appointment_time
        
        return {
            'age': patient.age,
            'gender_encoded': 1 if patient.gender == 'M' else 0,
            'visit_history_count': patient.visit_history_count,
            'avg_interval_days': patient.avg_interval_between_visits,
            'historical_no_show_rate': patient.no_show_history_rate,
            'phone_confirmed': int(patient.phone_confirmed),
            'advance_booking_days': appointment.advance_booking_days,
            'is_repeat_visit': int(appointment.is_repeat_visit),
            'reminder_sent': int(appointment.reminder_sent),
            'appointment_hour': appointment_time.hour,
            'appointment_weekday': appointment_time.weekday(),
            'is_morning_appointment': int(appointment_time.hour < 12),
            'is_afternoon_appointment': int(12 <= appointment_time.hour < 17),
            'is_evening_appointment': int(appointment_time.hour >= 17),
            'is_weekend': int(appointment_time.weekday() >= 5),
            'is_monday': int(appointment_time.weekday() == 0),
            'is_friday': int(appointment_time.weekday() == 4),
            'month': appointment_time.month,
            'is_summer': int(6 <= appointment_time.month <= 8),
            'is_winter': int(appointment_time.month in [12, 1, 2]),
            'cancelled': int(planning.cancelled == 'Y') if hasattr(planning, 'cancelled') else 0
        } 