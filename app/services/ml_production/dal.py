"""
Data Access Layer для ML-модуля
"""

from sqlalchemy import text, func
from sqlmodel import Session, select
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

from app.models.ml_production import NSFeatures, LoadStat, PLSnapshot
from app.database.database import get_session


logger = logging.getLogger(__name__)


class MLDataAccessLayer:
    """Data Access Layer для ML операций"""
    
    def __init__(self, session: Session):
        self.session = session

    def extract_noshow_features(self, planning_id: int) -> Optional[Dict[str, Any]]:
        """Извлекает признаки для прогнозирования неявок по planning_id"""
        query = text("""
            SELECT 
                p.planning_id,
                EXTRACT(DOW FROM p.date_cons) as weekday,
                EXTRACT(HOUR FROM p.heure) as hour,
                EXTRACT(MONTH FROM p.date_cons) as month,
                CASE WHEN EXTRACT(DOW FROM p.date_cons) IN (0, 6) THEN true ELSE false END as is_weekend,
                
                pat.age as patient_age,
                pat.pol as patient_gender,
                
                p.date_cons - p.create_date_time as advance_booking_days,
                
                m.specialisation_id as doctor_specialization_id,
                m.fm_dep_id as department_id,
                
                COALESCE(p.not_send_sms, false) as not_send_sms,
                CASE WHEN pat.email IS NOT NULL AND pat.email != '' THEN true ELSE false END as has_email,
                
                -- Подсчет прошлых неявок за 12 месяцев
                (SELECT COUNT(*) 
                 FROM planning p2 
                 WHERE p2.patients_id = p.patients_id 
                   AND p2.cancelled = 'Y' 
                   AND p2.date_cons >= p.date_cons - INTERVAL '12 months'
                   AND p2.date_cons < p.date_cons) as past_noshows_count
                   
            FROM planning p
            LEFT JOIN patients pat ON p.patients_id = pat.patients_id
            LEFT JOIN medecins m ON p.medecins_id = m.medecins_id
            WHERE p.planning_id = :planning_id
        """)
        
        result = self.session.execute(query, {"planning_id": planning_id}).fetchone()
        
        if not result:
            return None
            
        return {
            "planning_id": result.planning_id,
            "weekday": result.weekday,
            "hour": result.hour,
            "month": result.month,
            "is_weekend": result.is_weekend,
            "patient_age": result.patient_age,
            "patient_gender": result.patient_gender,
            "advance_booking_days": result.advance_booking_days.days if result.advance_booking_days else 0,
            "doctor_specialization_id": result.doctor_specialization_id,
            "department_id": result.department_id,
            "not_send_sms": result.not_send_sms,
            "has_email": result.has_email,
            "past_noshows_count": result.past_noshows_count or 0
        }

    def load_training_data(self, days_back: int = 365) -> pd.DataFrame:
        """Загружает обучающие данные для модели неявок"""
        query = text("""
            SELECT 
                nf.*,
                CASE WHEN p.cancelled = 'Y' THEN 1 ELSE 0 END as target
            FROM ns_features nf
            JOIN planning p ON nf.planning_id = p.planning_id
            WHERE nf.created_at >= :start_date
        """)
        
        start_date = datetime.now() - timedelta(days=days_back)
        result = self.session.execute(query, {"start_date": start_date})
        
        return pd.DataFrame(result.fetchall())

    def save_features(self, features: Dict[str, Any]) -> None:
        """Сохраняет признаки в витрину ns_features"""
        ns_feature = NSFeatures(**features)
        self.session.add(ns_feature)
        self.session.commit()

    def get_doctor_profiles(self, department_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Получает профили врачей для балансировки нагрузки"""
        query = text("""
            SELECT 
                m.medecins_id,
                m.specialisation_id,
                m.fm_dep_id as department_id,
                COALESCE(m.n_use_plan, 20) as daily_norm,
                
                -- Текущая загрузка (записи на сегодня)
                COUNT(p.planning_id) * 1.0 / COALESCE(m.n_use_plan, 20) as current_load,
                
                -- Среднее время консультации
                COALESCE(AVG(mc.duration), 30) as avg_consultation_time
                
            FROM medecins m
            LEFT JOIN planning p ON m.medecins_id = p.medecins_id 
                AND DATE(p.date_cons) = CURRENT_DATE
                AND p.cancelled != 'Y'
            LEFT JOIN motconsu mc ON p.planning_id = mc.planning_id
            WHERE m.active = true
            """ + ("AND m.fm_dep_id = ANY(:department_ids)" if department_ids else "") + """
            GROUP BY m.medecins_id, m.specialisation_id, m.fm_dep_id, m.n_use_plan
        """)
        
        params = {"department_ids": department_ids} if department_ids else {}
        result = self.session.execute(query, params)
        
        return [dict(row._mapping) for row in result.fetchall()]

    def get_schedule_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Получает данные расписания на период"""
        query = text("""
            SELECT 
                p.planning_id,
                p.patients_id,
                p.medecins_id,
                p.date_cons,
                p.heure,
                p.cancelled,
                p.cito,
                pat.vip_groups_id,
                m.specialisation_id,
                m.fm_dep_id,
                CASE WHEN p.cancelled = 'Y' THEN 'cancelled'
                     WHEN p.date_cons <= NOW() THEN 'completed'
                     ELSE 'scheduled' END as status
            FROM planning p
            JOIN patients pat ON p.patients_id = pat.patients_id  
            JOIN medecins m ON p.medecins_id = m.medecins_id
            WHERE p.date_cons BETWEEN :start_date AND :end_date
            ORDER BY p.date_cons, p.heure
        """)
        
        result = self.session.execute(query, {
            "start_date": start_date,
            "end_date": end_date
        })
        
        return [dict(row._mapping) for row in result.fetchall()]

    def get_waiting_patients(self) -> List[Dict[str, Any]]:
        """Получает пациентов в ожидании"""
        query = text("""
            SELECT 
                p.planning_id,
                p.patients_id,
                p.date_cons,
                p.heure,
                p.cito,
                pat.vip_groups_id,
                CASE WHEN p.cito = 'Y' THEN 10
                     WHEN pat.vip_groups_id IS NOT NULL THEN 5
                     ELSE 1 END as priority
            FROM planning p
            JOIN patients pat ON p.patients_id = pat.patients_id
            WHERE p.status = 'waiting'
              AND DATE(p.date_cons) = CURRENT_DATE
            ORDER BY priority DESC, p.heure ASC
        """)
        
        result = self.session.execute(query)
        return [dict(row._mapping) for row in result.fetchall()]

    def call_sp_calc_noshow(self, planning_id: int) -> None:
        """Вызывает хранимую процедуру расчета неявки"""
        self.session.execute(text("CALL sp_calc_noshow(:planning_id)"), 
                           {"planning_id": planning_id})
        self.session.commit()

    def call_sp_reschedule_batch(self, assignments: Dict[int, int]) -> None:
        """Вызывает процедуру пакетного переназначения"""
        for planning_id, medecins_id in assignments.items():
            self.session.execute(
                text("CALL sp_reschedule_batch(:planning_id, :medecins_id)"),
                {"planning_id": planning_id, "medecins_id": medecins_id}
            )
        self.session.commit()

    def save_load_stats(self, stats: Dict[str, Any]) -> None:
        """Сохраняет статистику загрузки"""
        load_stat = LoadStat(**stats)
        self.session.add(load_stat)
        self.session.commit()

    def save_snapshot(self, snapshot_data: Dict[str, Any]) -> None:
        """Сохраняет снимок изменений планирования"""
        snapshot = PLSnapshot(**snapshot_data)
        self.session.add(snapshot)
        self.session.commit()

    def update_planning_time(self, planning_id: int, new_time: datetime) -> None:
        """Обновляет время записи"""
        self.session.execute(
            text("UPDATE planning SET heure = :new_time WHERE planning_id = :planning_id"),
            {"new_time": new_time, "planning_id": planning_id}
        )
        self.session.commit()

    def trigger_job_notify_sms(self, planning_id: int) -> None:
        """Инициирует отправку SMS-уведомления"""
        self.session.execute(
            text("INSERT INTO job_queue (job_type, planning_id) VALUES ('notify_sms', :planning_id)"),
            {"planning_id": planning_id}
        )
        self.session.commit()

    def trigger_double_booking(self, planning_id: int) -> None:
        """Инициирует двойное бронирование"""
        self.session.execute(
            text("INSERT INTO job_queue (job_type, planning_id) VALUES ('double_booking', :planning_id)"),
            {"planning_id": planning_id}
        )
        self.session.commit() 