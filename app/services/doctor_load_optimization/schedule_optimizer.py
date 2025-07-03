"""
Оптимизатор расписания для врачей
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd

from app.services.doctor_load_optimization.schemas import (
    DoctorProfile, ScheduleSlot, OptimizationResult, OptimizationObjective
)
from app.services.doctor_load_optimization.load_balancer import (
    LoadBalancer, create_load_balancer, LoadBalancingStrategy
)


class ScheduleOptimizer:
    """Оптимизатор расписания в реальном времени"""

    def __init__(self, load_balancer: Optional[LoadBalancer] = None):
        self.load_balancer = load_balancer or create_load_balancer(LoadBalancingStrategy.ADAPTIVE)
        self.logger = logging.getLogger(__name__)

    def optimize_doctor_load(self, doctor_profiles: List[DoctorProfile],
                           date_range: Tuple[datetime, datetime],
                           objective: OptimizationObjective = OptimizationObjective.BALANCE_LOAD) -> OptimizationResult:
        """Оптимизация загрузки врачей"""
        try:
            start_time = datetime.utcnow()
            
            # Анализ паттернов записей
            appointment_patterns = self._analyze_appointment_patterns(doctor_profiles, date_range)
            
            # Оптимизация распределения
            if objective == OptimizationObjective.BALANCE_LOAD:
                optimal_distribution = self._balance_load_optimization(doctor_profiles)
            elif objective == OptimizationObjective.MINIMIZE_WAIT_TIME:
                optimal_distribution = self._minimize_wait_time_optimization(doctor_profiles, appointment_patterns)
            elif objective == OptimizationObjective.MAXIMIZE_UTILIZATION:
                optimal_distribution = self._maximize_utilization_optimization(doctor_profiles)
            elif objective == OptimizationObjective.MINIMIZE_COST:
                optimal_distribution = self._minimize_cost_optimization(doctor_profiles)
            else:
                optimal_distribution = self._balance_load_optimization(doctor_profiles)
            
            # Создание оптимизированного расписания
            optimized_schedule = self._create_optimized_schedule(optimal_distribution, date_range)
            
            # Расчет улучшений
            improvements = self._calculate_improvements(doctor_profiles, optimal_distribution)
            
            # Генерация рекомендаций
            recommendations = self._generate_recommendations(doctor_profiles, optimal_distribution)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return OptimizationResult(
                optimized_schedule=optimized_schedule,
                load_distribution=optimal_distribution,
                wait_time_reduction=improvements.get('wait_time_reduction', 0.0),
                utilization_improvement=improvements.get('utilization_improvement', 0.0),
                recommendations=recommendations,
                efficiency_score=improvements.get('efficiency_score', 0.0),
                cost_savings=improvements.get('cost_savings', 0.0),
                processing_time=processing_time
            )

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации загрузки врачей: {e}")
            raise

    def _analyze_appointment_patterns(self, doctor_profiles: List[DoctorProfile],
                                    date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Анализ паттернов записей"""
        try:
            patterns = {
                'peak_hours': [],
                'speciality_distribution': {},
                'complexity_distribution': {},
                'wait_time_patterns': {}
            }
            
            # Анализ по специальностям
            for doctor in doctor_profiles:
                if doctor.speciality not in patterns['speciality_distribution']:
                    patterns['speciality_distribution'][doctor.speciality] = {
                        'doctor_count': 0,
                        'avg_load': 0.0,
                        'avg_wait_time': 0.0
                    }
                
                patterns['speciality_distribution'][doctor.speciality]['doctor_count'] += 1
                patterns['speciality_distribution'][doctor.speciality]['avg_load'] += doctor.current_load
                patterns['speciality_distribution'][doctor.speciality]['avg_wait_time'] += doctor.avg_wait_time
            
            # Нормализация средних значений
            for speciality in patterns['speciality_distribution']:
                count = patterns['speciality_distribution'][speciality]['doctor_count']
                if count > 0:
                    patterns['speciality_distribution'][speciality]['avg_load'] /= count
                    patterns['speciality_distribution'][speciality]['avg_wait_time'] /= count
            
            # Анализ пиковых часов (упрощенный)
            patterns['peak_hours'] = [9, 10, 11, 14, 15, 16]  # Заглушка
            
            return patterns

        except Exception as e:
            self.logger.error(f"Ошибка анализа паттернов: {e}")
            return {}

    def _balance_load_optimization(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Оптимизация балансировки нагрузки"""
        try:
            return self.load_balancer.balance_load(doctor_profiles)
        except Exception as e:
            self.logger.error(f"Ошибка балансировки нагрузки: {e}")
            return {}

    def _minimize_wait_time_optimization(self, doctor_profiles: List[DoctorProfile],
                                       patterns: Dict[str, Any]) -> Dict[int, float]:
        """Оптимизация для минимизации времени ожидания"""
        try:
            # Сортировка по времени ожидания
            sorted_doctors = sorted(doctor_profiles, key=lambda x: x.avg_wait_time)
            
            optimal_distribution = {}
            total_doctors = len(sorted_doctors)
            
            for i, doctor in enumerate(sorted_doctors):
                # Приоритет врачам с меньшим временем ожидания
                priority_factor = (total_doctors - i) / total_doctors
                target_load = 0.6 + priority_factor * 0.3  # 0.6-0.9 диапазон
                optimal_distribution[doctor.doctor_id] = max(0.1, min(0.95, target_load))
            
            return optimal_distribution

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации времени ожидания: {e}")
            return {}

    def _maximize_utilization_optimization(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Оптимизация для максимизации использования ресурсов"""
        try:
            # Сортировка по коэффициенту использования
            sorted_doctors = sorted(doctor_profiles, key=lambda x: x.utilization_rate, reverse=True)
            
            optimal_distribution = {}
            total_doctors = len(sorted_doctors)
            
            for i, doctor in enumerate(sorted_doctors):
                # Приоритет врачам с высоким использованием
                priority_factor = (i + 1) / total_doctors
                target_load = 0.7 + priority_factor * 0.2  # 0.7-0.9 диапазон
                optimal_distribution[doctor.doctor_id] = max(0.1, min(0.95, target_load))
            
            return optimal_distribution

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации использования: {e}")
            return {}

    def _minimize_cost_optimization(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Оптимизация для минимизации затрат"""
        try:
            # Сортировка по эффективности (соотношение качества и затрат)
            sorted_doctors = sorted(doctor_profiles, key=lambda x: x.patient_satisfaction / (x.current_load + 0.1), reverse=True)
            
            optimal_distribution = {}
            total_doctors = len(sorted_doctors)
            
            for i, doctor in enumerate(sorted_doctors):
                # Приоритет эффективным врачам
                efficiency_factor = (i + 1) / total_doctors
                target_load = 0.6 + efficiency_factor * 0.3  # 0.6-0.9 диапазон
                optimal_distribution[doctor.doctor_id] = max(0.1, min(0.95, target_load))
            
            return optimal_distribution

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации затрат: {e}")
            return {}

    def _create_optimized_schedule(self, optimal_distribution: Dict[int, float],
                                 date_range: Tuple[datetime, datetime]) -> Dict[int, List[ScheduleSlot]]:
        """Создание оптимизированного расписания"""
        try:
            optimized_schedule = {}
            start_date, end_date = date_range
            
            # Создание слотов для каждого врача
            for doctor_id, target_load in optimal_distribution.items():
                slots = []
                current_date = start_date
                
                while current_date <= end_date:
                    # Создание дневных слотов
                    daily_slots = self._create_daily_slots(doctor_id, current_date, target_load)
                    slots.extend(daily_slots)
                    current_date += timedelta(days=1)
                
                optimized_schedule[doctor_id] = slots
            
            return optimized_schedule

        except Exception as e:
            self.logger.error(f"Ошибка создания оптимизированного расписания: {e}")
            return {}

    def _create_daily_slots(self, doctor_id: int, date: datetime, target_load: float) -> List[ScheduleSlot]:
        """Создание дневных слотов для врача"""
        try:
            slots = []
            
            # Рабочие часы (8:00 - 18:00)
            work_start = datetime.combine(date.date(), datetime.min.time().replace(hour=8))
            work_end = datetime.combine(date.date(), datetime.min.time().replace(hour=18))
            
            # Длительность слота (30 минут)
            slot_duration = timedelta(minutes=30)
            
            current_time = work_start
            while current_time < work_end:
                # Определение доступности слота на основе целевой нагрузки
                is_available = np.random.random() > target_load
                
                slot = ScheduleSlot(
                    start_time=current_time,
                    end_time=current_time + slot_duration,
                    doctor_id=doctor_id,
                    patient_id=None,
                    appointment_type="consultation",
                    complexity=0.5,
                    is_available=is_available
                )
                
                slots.append(slot)
                current_time += slot_duration
            
            return slots

        except Exception as e:
            self.logger.error(f"Ошибка создания дневных слотов: {e}")
            return []

    def _calculate_improvements(self, current_profiles: List[DoctorProfile],
                              optimal_distribution: Dict[int, float]) -> Dict[str, float]:
        """Расчет улучшений от оптимизации"""
        try:
            improvements = {}
            
            # Текущие метрики
            current_avg_load = sum(d.current_load for d in current_profiles) / len(current_profiles)
            current_avg_wait = sum(d.avg_wait_time for d in current_profiles) / len(current_profiles)
            current_avg_utilization = sum(d.utilization_rate for d in current_profiles) / len(current_profiles)
            
            # Оптимальные метрики
            optimal_avg_load = sum(optimal_distribution.values()) / len(optimal_distribution) if optimal_distribution else 0.0
            
            # Расчет улучшений
            load_improvement = (optimal_avg_load - current_avg_load) / current_avg_load if current_avg_load > 0 else 0.0
            wait_time_reduction = max(0.0, (current_avg_wait - 15) / current_avg_wait) if current_avg_wait > 0 else 0.0
            utilization_improvement = max(0.0, (0.85 - current_avg_utilization) / current_avg_utilization) if current_avg_utilization > 0 else 0.0
            
            # Общая оценка эффективности
            efficiency_score = (
                load_improvement * 0.4 +
                wait_time_reduction * 0.3 +
                utilization_improvement * 0.3
            )
            
            # Оценка экономии затрат
            cost_savings = efficiency_score * 1000  # Упрощенная оценка
            
            improvements.update({
                'wait_time_reduction': wait_time_reduction * 100,  # В процентах
                'utilization_improvement': utilization_improvement * 100,  # В процентах
                'efficiency_score': max(0.0, min(1.0, efficiency_score)),
                'cost_savings': cost_savings
            })
            
            return improvements

        except Exception as e:
            self.logger.error(f"Ошибка расчета улучшений: {e}")
            return {
                'wait_time_reduction': 0.0,
                'utilization_improvement': 0.0,
                'efficiency_score': 0.0,
                'cost_savings': 0.0
            }

    def _generate_recommendations(self, doctor_profiles: List[DoctorProfile],
                                optimal_distribution: Dict[int, float]) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        try:
            recommendations = []
            
            # Анализ загруженности
            overloaded_doctors = [d for d in doctor_profiles if d.current_load > 0.8]
            underloaded_doctors = [d for d in doctor_profiles if d.current_load < 0.3]
            
            if overloaded_doctors:
                recommendations.append(
                    f"Перераспределить нагрузку с {len(overloaded_doctors)} перегруженных врачей"
                )
            
            if underloaded_doctors:
                recommendations.append(
                    f"Увеличить нагрузку для {len(underloaded_doctors)} недостаточно загруженных врачей"
                )
            
            # Анализ времени ожидания
            high_wait_doctors = [d for d in doctor_profiles if d.avg_wait_time > 30]
            if high_wait_doctors:
                recommendations.append(
                    f"Оптимизировать расписание для {len(high_wait_doctors)} врачей с высоким временем ожидания"
                )
            
            # Анализ использования ресурсов
            low_utilization_doctors = [d for d in doctor_profiles if d.utilization_rate < 0.5]
            if low_utilization_doctors:
                recommendations.append(
                    f"Повысить эффективность использования времени для {len(low_utilization_doctors)} врачей"
                )
            
            # Общие рекомендации
            if not recommendations:
                recommendations.append("Текущее распределение нагрузки оптимально")
            
            return recommendations

        except Exception as e:
            self.logger.error(f"Ошибка генерации рекомендаций: {e}")
            return ["Ошибка при анализе данных для генерации рекомендаций"]

    def get_optimal_doctor_for_appointment(self, speciality: str, complexity: float,
                                         doctor_profiles: List[DoctorProfile]) -> Optional[int]:
        """Получение оптимального врача для конкретного приема"""
        try:
            return self.load_balancer.get_optimal_doctor(speciality, complexity, doctor_profiles)
        except Exception as e:
            self.logger.error(f"Ошибка выбора оптимального врача: {e}")
            return None 