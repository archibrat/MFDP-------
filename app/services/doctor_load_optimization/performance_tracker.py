"""
Модуль отслеживания производительности врачей
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
import pandas as pd

from app.services.doctor_load_optimization.schemas import DoctorProfile, PerformanceMetrics


class PerformanceMetric(str, Enum):
    """Типы метрик производительности"""
    APPOINTMENT_COUNT = "appointment_count"
    CONSULTATION_TIME = "consultation_time"
    PATIENT_SATISFACTION = "patient_satisfaction"
    UTILIZATION_RATE = "utilization_rate"
    EFFICIENCY_SCORE = "efficiency_score"
    NO_SHOW_RATE = "no_show_rate"
    COMPLETION_RATE = "completion_rate"


class PerformanceTracker:
    """Отслеживатель производительности врачей"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = {}  # {doctor_id: [metrics]}
        self.benchmarks = {}  # {metric_type: benchmark_value}

    def track_doctor_performance(self, doctor_id: int, 
                               period_start: datetime,
                               period_end: datetime) -> PerformanceMetrics:
        """Отслеживание производительности врача за период"""
        try:
            # Получение данных о приемах (заглушка)
            appointments_data = self._get_appointments_data(doctor_id, period_start, period_end)
            consultations_data = self._get_consultations_data(doctor_id, period_start, period_end)
            
            # Расчет метрик
            total_appointments = len(appointments_data)
            completed_appointments = len([a for a in appointments_data if a.get('status') == 'completed'])
            avg_consultation_time = self._calculate_avg_consultation_time(consultations_data)
            patient_satisfaction = self._calculate_patient_satisfaction(appointments_data)
            utilization_rate = self._calculate_utilization_rate(appointments_data, period_start, period_end)
            efficiency_score = self._calculate_efficiency_score(
                total_appointments, completed_appointments, avg_consultation_time, 
                patient_satisfaction, utilization_rate
            )
            
            # Генерация рекомендаций
            recommendations = self._generate_performance_recommendations(
                total_appointments, completed_appointments, avg_consultation_time,
                patient_satisfaction, utilization_rate, efficiency_score
            )
            
            return PerformanceMetrics(
                doctor_id=doctor_id,
                period_start=period_start,
                period_end=period_end,
                total_appointments=total_appointments,
                completed_appointments=completed_appointments,
                avg_consultation_time=avg_consultation_time,
                patient_satisfaction=patient_satisfaction,
                utilization_rate=utilization_rate,
                efficiency_score=efficiency_score,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Ошибка отслеживания производительности врача {doctor_id}: {e}")
            raise

    def _get_appointments_data(self, doctor_id: int, start_date: datetime,
                             end_date: datetime) -> List[Dict[str, Any]]:
        """Получение данных о приемах (заглушка)"""
        try:
            # Симуляция данных о приемах
            appointments = []
            current_date = start_date
            
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Рабочие дни
                    # Генерация случайного количества приемов в день
                    daily_appointments = np.random.poisson(8)  # В среднем 8 приемов в день
                    
                    for i in range(daily_appointments):
                        appointment_time = current_date.replace(
                            hour=np.random.randint(9, 17),
                            minute=np.random.randint(0, 60)
                        )
                        
                        appointments.append({
                            'appointment_id': len(appointments) + 1,
                            'appointment_time': appointment_time,
                            'status': np.random.choice(['completed', 'cancelled', 'no_show'], p=[0.8, 0.1, 0.1]),
                            'duration_minutes': np.random.randint(15, 60),
                            'patient_satisfaction': np.random.uniform(0.6, 1.0)
                        })
                
                current_date += timedelta(days=1)
            
            return appointments

        except Exception as e:
            self.logger.error(f"Ошибка получения данных о приемах: {e}")
            return []

    def _get_consultations_data(self, doctor_id: int, start_date: datetime,
                              end_date: datetime) -> List[Dict[str, Any]]:
        """Получение данных о консультациях (заглушка)"""
        try:
            # Симуляция данных о консультациях
            consultations = []
            current_date = start_date
            
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Рабочие дни
                    # Генерация случайного количества консультаций в день
                    daily_consultations = np.random.poisson(5)  # В среднем 5 консультаций в день
                    
                    for i in range(daily_consultations):
                        consultation_time = current_date.replace(
                            hour=np.random.randint(9, 17),
                            minute=np.random.randint(0, 60)
                        )
                        
                        consultations.append({
                            'consultation_id': len(consultations) + 1,
                            'visit_date': consultation_time,
                            'duration_minutes': np.random.randint(20, 90),
                            'diagnosis': 'Общий осмотр',
                            'treatment': 'Рекомендации'
                        })
                
                current_date += timedelta(days=1)
            
            return consultations

        except Exception as e:
            self.logger.error(f"Ошибка получения данных о консультациях: {e}")
            return []

    def _calculate_avg_consultation_time(self, consultations_data: List[Dict[str, Any]]) -> float:
        """Расчет среднего времени консультации"""
        try:
            if not consultations_data:
                return 0.0
            
            total_time = sum(consultation.get('duration_minutes', 0) for consultation in consultations_data)
            return total_time / len(consultations_data)

        except Exception as e:
            self.logger.error(f"Ошибка расчета среднего времени консультации: {e}")
            return 0.0

    def _calculate_patient_satisfaction(self, appointments_data: List[Dict[str, Any]]) -> float:
        """Расчет удовлетворенности пациентов"""
        try:
            if not appointments_data:
                return 0.5  # Нейтральная оценка при отсутствии данных
            
            satisfaction_scores = [
                appointment.get('patient_satisfaction', 0.5) 
                for appointment in appointments_data
                if appointment.get('status') == 'completed'
            ]
            
            if not satisfaction_scores:
                return 0.5
            
            return sum(satisfaction_scores) / len(satisfaction_scores)

        except Exception as e:
            self.logger.error(f"Ошибка расчета удовлетворенности пациентов: {e}")
            return 0.5

    def _calculate_utilization_rate(self, appointments_data: List[Dict[str, Any]],
                                  period_start: datetime, period_end: datetime) -> float:
        """Расчет коэффициента использования времени"""
        try:
            if not appointments_data:
                return 0.0
            
            # Расчет общего рабочего времени за период
            work_days = 0
            current_date = period_start
            while current_date <= period_end:
                if current_date.weekday() < 5:  # Рабочие дни
                    work_days += 1
                current_date += timedelta(days=1)
            
            total_available_hours = work_days * 8  # 8 часов в день
            total_available_minutes = total_available_hours * 60
            
            # Расчет использованного времени
            total_used_minutes = sum(
                appointment.get('duration_minutes', 0) 
                for appointment in appointments_data
                if appointment.get('status') == 'completed'
            )
            
            utilization_rate = total_used_minutes / total_available_minutes if total_available_minutes > 0 else 0.0
            return min(1.0, utilization_rate)

        except Exception as e:
            self.logger.error(f"Ошибка расчета коэффициента использования: {e}")
            return 0.0

    def _calculate_efficiency_score(self, total_appointments: int, completed_appointments: int,
                                  avg_consultation_time: float, patient_satisfaction: float,
                                  utilization_rate: float) -> float:
        """Расчет общей оценки эффективности"""
        try:
            # Веса для различных факторов
            weights = {
                'completion_rate': 0.3,
                'consultation_time': 0.2,
                'satisfaction': 0.25,
                'utilization': 0.25
            }
            
            # Нормализованные значения
            completion_rate = completed_appointments / total_appointments if total_appointments > 0 else 0.0
            
            # Оптимальное время консультации - 30 минут
            time_efficiency = 1.0 - abs(avg_consultation_time - 30) / 30 if avg_consultation_time > 0 else 0.0
            time_efficiency = max(0.0, min(1.0, time_efficiency))
            
            # Взвешенная оценка эффективности
            efficiency_score = (
                completion_rate * weights['completion_rate'] +
                time_efficiency * weights['consultation_time'] +
                patient_satisfaction * weights['satisfaction'] +
                utilization_rate * weights['utilization']
            )
            
            return max(0.0, min(1.0, efficiency_score))

        except Exception as e:
            self.logger.error(f"Ошибка расчета оценки эффективности: {e}")
            return 0.5

    def _generate_performance_recommendations(self, total_appointments: int,
                                            completed_appointments: int,
                                            avg_consultation_time: float,
                                            patient_satisfaction: float,
                                            utilization_rate: float,
                                            efficiency_score: float) -> List[str]:
        """Генерация рекомендаций по улучшению производительности"""
        try:
            recommendations = []
            
            # Рекомендации по завершению приемов
            completion_rate = completed_appointments / total_appointments if total_appointments > 0 else 0.0
            if completion_rate < 0.8:
                recommendations.append("Улучшите планирование времени для снижения отмен приемов")
            
            # Рекомендации по времени консультации
            if avg_consultation_time < 20:
                recommendations.append("Рассмотрите увеличение времени консультации для более детального осмотра")
            elif avg_consultation_time > 60:
                recommendations.append("Оптимизируйте процесс консультации для сокращения времени приема")
            
            # Рекомендации по удовлетворенности
            if patient_satisfaction < 0.7:
                recommendations.append("Проведите анализ причин низкой удовлетворенности пациентов")
            
            # Рекомендации по использованию времени
            if utilization_rate < 0.6:
                recommendations.append("Повысьте эффективность использования рабочего времени")
            elif utilization_rate > 0.9:
                recommendations.append("Рассмотрите возможность увеличения времени на отдых между приемами")
            
            # Общие рекомендации
            if efficiency_score < 0.6:
                recommendations.append("Проведите комплексный анализ производительности и разработайте план улучшений")
            
            if not recommendations:
                recommendations.append("Производительность находится на оптимальном уровне")
            
            return recommendations

        except Exception as e:
            self.logger.error(f"Ошибка генерации рекомендаций: {e}")
            return ["Ошибка при анализе производительности"]

    def get_performance_trends(self, doctor_id: int, days: int = 30) -> Dict[str, List[float]]:
        """Получение трендов производительности"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            trends = {
                'appointment_counts': [],
                'satisfaction_scores': [],
                'utilization_rates': [],
                'efficiency_scores': []
            }
            
            # Разбиение периода на недели
            current_date = start_date
            while current_date <= end_date:
                week_end = min(current_date + timedelta(days=7), end_date)
                
                # Получение метрик за неделю
                weekly_metrics = self.track_doctor_performance(doctor_id, current_date, week_end)
                
                trends['appointment_counts'].append(weekly_metrics.total_appointments)
                trends['satisfaction_scores'].append(weekly_metrics.patient_satisfaction)
                trends['utilization_rates'].append(weekly_metrics.utilization_rate)
                trends['efficiency_scores'].append(weekly_metrics.efficiency_score)
                
                current_date = week_end
            
            return trends

        except Exception as e:
            self.logger.error(f"Ошибка получения трендов производительности: {e}")
            return {}

    def compare_with_benchmarks(self, doctor_metrics: PerformanceMetrics) -> Dict[str, Dict[str, Any]]:
        """Сравнение с эталонными показателями"""
        try:
            benchmarks = {
                'appointment_count': {'target': 20, 'excellent': 25},
                'consultation_time': {'target': 30, 'excellent': 25},
                'satisfaction': {'target': 0.8, 'excellent': 0.9},
                'utilization': {'target': 0.75, 'excellent': 0.85},
                'efficiency': {'target': 0.7, 'excellent': 0.8}
            }
            
            comparison = {}
            
            # Сравнение количества приемов
            comparison['appointment_count'] = {
                'current': doctor_metrics.total_appointments,
                'target': benchmarks['appointment_count']['target'],
                'excellent': benchmarks['appointment_count']['excellent'],
                'status': self._get_performance_status(
                    doctor_metrics.total_appointments,
                    benchmarks['appointment_count']['target'],
                    benchmarks['appointment_count']['excellent']
                )
            }
            
            # Сравнение времени консультации
            comparison['consultation_time'] = {
                'current': doctor_metrics.avg_consultation_time,
                'target': benchmarks['consultation_time']['target'],
                'excellent': benchmarks['consultation_time']['excellent'],
                'status': self._get_performance_status(
                    doctor_metrics.avg_consultation_time,
                    benchmarks['consultation_time']['target'],
                    benchmarks['consultation_time']['excellent'],
                    reverse=True  # Меньше время - лучше
                )
            }
            
            # Сравнение удовлетворенности
            comparison['satisfaction'] = {
                'current': doctor_metrics.patient_satisfaction,
                'target': benchmarks['satisfaction']['target'],
                'excellent': benchmarks['satisfaction']['excellent'],
                'status': self._get_performance_status(
                    doctor_metrics.patient_satisfaction,
                    benchmarks['satisfaction']['target'],
                    benchmarks['satisfaction']['excellent']
                )
            }
            
            # Сравнение использования времени
            comparison['utilization'] = {
                'current': doctor_metrics.utilization_rate,
                'target': benchmarks['utilization']['target'],
                'excellent': benchmarks['utilization']['excellent'],
                'status': self._get_performance_status(
                    doctor_metrics.utilization_rate,
                    benchmarks['utilization']['target'],
                    benchmarks['utilization']['excellent']
                )
            }
            
            # Сравнение эффективности
            comparison['efficiency'] = {
                'current': doctor_metrics.efficiency_score,
                'target': benchmarks['efficiency']['target'],
                'excellent': benchmarks['efficiency']['excellent'],
                'status': self._get_performance_status(
                    doctor_metrics.efficiency_score,
                    benchmarks['efficiency']['target'],
                    benchmarks['efficiency']['excellent']
                )
            }
            
            return comparison

        except Exception as e:
            self.logger.error(f"Ошибка сравнения с эталонами: {e}")
            return {}

    def _get_performance_status(self, current: float, target: float, excellent: float, 
                              reverse: bool = False) -> str:
        """Получение статуса производительности"""
        try:
            if reverse:
                # Для метрик, где меньшее значение лучше
                if current <= excellent:
                    return "excellent"
                elif current <= target:
                    return "good"
                else:
                    return "needs_improvement"
            else:
                # Для метрик, где большее значение лучше
                if current >= excellent:
                    return "excellent"
                elif current >= target:
                    return "good"
                else:
                    return "needs_improvement"

        except Exception as e:
            self.logger.error(f"Ошибка определения статуса производительности: {e}")
            return "unknown"

    def save_metrics_history(self, doctor_id: int, metrics: PerformanceMetrics):
        """Сохранение метрик в историю"""
        try:
            if doctor_id not in self.metrics_history:
                self.metrics_history[doctor_id] = []
            
            self.metrics_history[doctor_id].append(metrics)
            
            # Ограничение истории (последние 100 записей)
            if len(self.metrics_history[doctor_id]) > 100:
                self.metrics_history[doctor_id] = self.metrics_history[doctor_id][-100:]
                
        except Exception as e:
            self.logger.error(f"Ошибка сохранения истории метрик: {e}")

    def get_metrics_history(self, doctor_id: int, limit: int = 10) -> List[PerformanceMetrics]:
        """Получение истории метрик"""
        try:
            if doctor_id in self.metrics_history:
                return self.metrics_history[doctor_id][-limit:]
            return []

        except Exception as e:
            self.logger.error(f"Ошибка получения истории метрик: {e}")
            return [] 