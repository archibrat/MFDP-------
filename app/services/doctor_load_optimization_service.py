"""
Модуль динамической оптимизации загрузки врачей
Реализует балансировку нагрузки, оптимизацию расписания и распределение ресурсов
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
import pandas as pd
from sqlmodel import Session, select

from models.medialog import Medecin, Schedule, Appointment, Consultation
from repositories.medialog_repository import MedialogRepository


class OptimizationObjective(Enum):
    MINIMIZE_WAIT_TIME = "minimize_wait_time"
    MAXIMIZE_UTILIZATION = "maximize_utilization"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_COST = "minimize_cost"


@dataclass
class DoctorMetrics:
    """Метрики врача для оптимизации"""
    doctor_id: int
    speciality: str
    department: str
    current_load: float  # Коэффициент загруженности (0-1)
    avg_wait_time: float  # Среднее время ожидания (минуты)
    utilization_rate: float  # Коэффициент использования времени
    patient_satisfaction: float  # Оценка удовлетворенности (0-1)
    complexity_score: float  # Сложность случаев (0-1)


@dataclass
class ScheduleSlot:
    """Слот расписания"""
    start_time: datetime
    end_time: datetime
    doctor_id: int
    patient_id: Optional[int]
    appointment_type: str
    complexity: float
    is_available: bool = True


@dataclass
class OptimizationResult:
    """Результат оптимизации"""
    optimized_schedule: Dict[int, List[ScheduleSlot]]
    load_distribution: Dict[int, float]
    wait_time_reduction: float
    utilization_improvement: float
    recommendations: List[str]
    efficiency_score: float


class LoadBalancer(ABC):
    """Абстрактный класс балансировщика нагрузки"""

    @abstractmethod
    def balance_load(self, doctor_metrics: List[DoctorMetrics]) -> Dict[int, float]:
        """Балансировка нагрузки между врачами"""
        pass

    @abstractmethod
    def get_optimal_doctor(self, speciality: str, complexity: float) -> Optional[int]:
        """Получение оптимального врача для приема"""
        pass


class RoundRobinLoadBalancer(LoadBalancer):
    """Балансировщик нагрузки по принципу Round Robin"""

    def __init__(self):
        self.current_index = 0
        self.logger = logging.getLogger(__name__)

    def balance_load(self, doctor_metrics: List[DoctorMetrics]) -> Dict[int, float]:
        """Балансировка нагрузки методом Round Robin"""
        try:
            if not doctor_metrics:
                return {}

            # Сортировка по загруженности
            sorted_doctors = sorted(doctor_metrics, key=lambda x: x.current_load)
            
            # Равномерное распределение нагрузки
            total_load = sum(d.current_load for d in sorted_doctors)
            avg_load = total_load / len(sorted_doctors)
            
            balanced_loads = {}
            for i, doctor in enumerate(sorted_doctors):
                # Небольшая вариация для равномерности
                variation = (i - len(sorted_doctors) / 2) * 0.05
                balanced_loads[doctor.doctor_id] = max(0, avg_load + variation)
            
            return balanced_loads

        except Exception as e:
            self.logger.error(f"Ошибка балансировки нагрузки: {e}")
            return {}

    def get_optimal_doctor(self, speciality: str, complexity: float) -> Optional[int]:
        """Получение оптимального врача для приема"""
        # Упрощенная логика - возвращает следующего врача в очереди
        self.current_index = (self.current_index + 1) % 100  # Заглушка
        return self.current_index


class WeightedLoadBalancer(LoadBalancer):
    """Балансировщик нагрузки с учетом весов"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def balance_load(self, doctor_metrics: List[DoctorMetrics]) -> Dict[int, float]:
        """Балансировка нагрузки с учетом весов"""
        try:
            if not doctor_metrics:
                return {}

            balanced_loads = {}
            
            for doctor in doctor_metrics:
                # Веса для различных факторов
                load_weight = 0.4
                satisfaction_weight = 0.3
                utilization_weight = 0.3
                
                # Нормализованные значения
                normalized_load = 1 - doctor.current_load  # Инвертируем для минимизации
                normalized_satisfaction = doctor.patient_satisfaction
                normalized_utilization = doctor.utilization_rate
                
                # Взвешенная оценка
                weighted_score = (
                    normalized_load * load_weight +
                    normalized_satisfaction * satisfaction_weight +
                    normalized_utilization * utilization_weight
                )
                
                # Преобразование в целевую нагрузку
                target_load = 0.7 + (weighted_score - 0.5) * 0.4  # 0.5-0.9 диапазон
                balanced_loads[doctor.doctor_id] = max(0.1, min(0.95, target_load))
            
            return balanced_loads

        except Exception as e:
            self.logger.error(f"Ошибка взвешенной балансировки: {e}")
            return {}

    def get_optimal_doctor(self, speciality: str, complexity: float) -> Optional[int]:
        """Получение оптимального врача с учетом сложности случая"""
        # Логика выбора врача на основе сложности
        return None


class ScheduleOptimizer:
    """Оптимизатор расписания в реальном времени"""

    def __init__(self, repository: MedialogRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def optimize_doctor_load(self, date_range: Tuple[datetime, datetime], 
                           objective: OptimizationObjective = OptimizationObjective.BALANCE_LOAD) -> OptimizationResult:
        """Оптимизация загрузки врачей"""
        try:
            # Получение метрик врачей
            doctor_metrics = self.calculate_current_load(date_range)
            
            # Анализ паттернов записей
            appointment_patterns = self.analyze_appointment_patterns(date_range)
            
            # Оптимизация распределения
            if objective == OptimizationObjective.BALANCE_LOAD:
                optimal_distribution = self.balance_load_optimization(doctor_metrics)
            elif objective == OptimizationObjective.MINIMIZE_WAIT_TIME:
                optimal_distribution = self.minimize_wait_time_optimization(doctor_metrics, appointment_patterns)
            elif objective == OptimizationObjective.MAXIMIZE_UTILIZATION:
                optimal_distribution = self.maximize_utilization_optimization(doctor_metrics)
            else:
                optimal_distribution = self.balance_load_optimization(doctor_metrics)
            
            # Создание оптимизированного расписания
            optimized_schedule = self.create_optimized_schedule(optimal_distribution, date_range)
            
            # Расчет улучшений
            improvements = self.calculate_improvements(doctor_metrics, optimal_distribution)
            
            # Генерация рекомендаций
            recommendations = self.generate_recommendations(doctor_metrics, optimal_distribution)
            
            return OptimizationResult(
                optimized_schedule=optimized_schedule,
                load_distribution=optimal_distribution,
                wait_time_reduction=improvements['wait_time_reduction'],
                utilization_improvement=improvements['utilization_improvement'],
                recommendations=recommendations,
                efficiency_score=improvements['efficiency_score']
            )

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации загрузки врачей: {e}")
            raise

    def calculate_current_load(self, date_range: Tuple[datetime, datetime]) -> List[DoctorMetrics]:
        """Расчет текущей загрузки врачей"""
        try:
            start_date, end_date = date_range
            
            # Получение всех врачей
            doctors = self.repository.get_all_medecins()
            doctor_metrics = []
            
            for doctor in doctors:
                # Получение расписания врача
                schedules = self.repository.get_schedule_by_medecin(
                    doctor.medecins_id, start_date, end_date
                )
                
                # Получение записей врача
                appointments = self.repository.get_appointments_by_medecin(
                    doctor.medecins_id, start_date, end_date
                )
                
                # Расчет метрик
                total_slots = sum(s.slots_total for s in schedules)
                booked_slots = sum(s.slots_booked for s in schedules)
                current_load = booked_slots / total_slots if total_slots > 0 else 0
                
                # Расчет среднего времени ожидания
                avg_wait_time = self.calculate_average_wait_time(appointments)
                
                # Расчет коэффициента использования
                utilization_rate = self.calculate_utilization_rate(schedules, appointments)
                
                # Оценка удовлетворенности (упрощенная)
                patient_satisfaction = self.estimate_patient_satisfaction(appointments)
                
                # Оценка сложности случаев
                complexity_score = self.calculate_complexity_score(appointments)
                
                metrics = DoctorMetrics(
                    doctor_id=doctor.medecins_id,
                    speciality=doctor.speciality,
                    department=doctor.department,
                    current_load=current_load,
                    avg_wait_time=avg_wait_time,
                    utilization_rate=utilization_rate,
                    patient_satisfaction=patient_satisfaction,
                    complexity_score=complexity_score
                )
                
                doctor_metrics.append(metrics)
            
            return doctor_metrics

        except Exception as e:
            self.logger.error(f"Ошибка расчета загрузки врачей: {e}")
            return []

    def analyze_appointment_patterns(self, date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Анализ паттернов записей"""
        try:
            start_date, end_date = date_range
            
            # Получение всех записей за период
            all_appointments = []
            for day in pd.date_range(start_date, end_date):
                appointments = self.repository.get_appointments_by_date(day)
                all_appointments.extend(appointments)
            
            # Анализ по часам
            hourly_patterns = {}
            for hour in range(8, 18):  # Рабочие часы
                hour_appointments = [
                    a for a in all_appointments 
                    if a.appointment_time.hour == hour
                ]
                hourly_patterns[hour] = len(hour_appointments)
            
            # Анализ по дням недели
            daily_patterns = {}
            for day in range(7):
                day_appointments = [
                    a for a in all_appointments 
                    if a.appointment_time.weekday() == day
                ]
                daily_patterns[day] = len(day_appointments)
            
            # Анализ по типам приемов
            type_patterns = {}
            for appointment in all_appointments:
                visit_type = appointment.visit_type
                type_patterns[visit_type] = type_patterns.get(visit_type, 0) + 1
            
            return {
                'hourly_patterns': hourly_patterns,
                'daily_patterns': daily_patterns,
                'type_patterns': type_patterns,
                'total_appointments': len(all_appointments)
            }

        except Exception as e:
            self.logger.error(f"Ошибка анализа паттернов записей: {e}")
            return {}

    def balance_load_optimization(self, doctor_metrics: List[DoctorMetrics]) -> Dict[int, float]:
        """Оптимизация балансировки нагрузки"""
        try:
            # Группировка врачей по специальностям
            speciality_groups = {}
            for doctor in doctor_metrics:
                if doctor.speciality not in speciality_groups:
                    speciality_groups[doctor.speciality] = []
                speciality_groups[doctor.speciality].append(doctor)
            
            optimal_distribution = {}
            
            for speciality, doctors in speciality_groups.items():
                # Сортировка по текущей загрузке
                sorted_doctors = sorted(doctors, key=lambda x: x.current_load)
                
                # Равномерное распределение
                total_load = sum(d.current_load for d in sorted_doctors)
                target_load_per_doctor = total_load / len(sorted_doctors)
                
                for doctor in sorted_doctors:
                    # Небольшая вариация для учета индивидуальных особенностей
                    variation = (doctor.patient_satisfaction - 0.5) * 0.1
                    optimal_load = max(0.1, min(0.95, target_load_per_doctor + variation))
                    optimal_distribution[doctor.doctor_id] = optimal_load
            
            return optimal_distribution

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации балансировки: {e}")
            return {}

    def minimize_wait_time_optimization(self, doctor_metrics: List[DoctorMetrics], 
                                      patterns: Dict[str, Any]) -> Dict[int, float]:
        """Оптимизация для минимизации времени ожидания"""
        try:
            optimal_distribution = {}
            
            for doctor in doctor_metrics:
                # Базовый коэффициент загрузки
                base_load = 0.7
                
                # Корректировка на основе времени ожидания
                wait_time_factor = max(0.5, min(1.5, 1 - doctor.avg_wait_time / 60))
                
                # Корректировка на основе удовлетворенности
                satisfaction_factor = 0.8 + doctor.patient_satisfaction * 0.4
                
                # Финальная загрузка
                optimal_load = base_load * wait_time_factor * satisfaction_factor
                optimal_distribution[doctor.doctor_id] = max(0.1, min(0.95, optimal_load))
            
            return optimal_distribution

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации времени ожидания: {e}")
            return {}

    def maximize_utilization_optimization(self, doctor_metrics: List[DoctorMetrics]) -> Dict[int, float]:
        """Оптимизация для максимизации использования"""
        try:
            optimal_distribution = {}
            
            for doctor in doctor_metrics:
                # Целевая загрузка на основе текущего использования
                target_utilization = 0.85  # 85% - оптимальная загрузка
                
                # Корректировка на основе сложности случаев
                complexity_factor = 0.9 + doctor.complexity_score * 0.2
                
                # Корректировка на основе специальности
                speciality_factor = 1.0
                if doctor.speciality in ['хирург', 'кардиолог']:
                    speciality_factor = 0.8  # Снижаем загрузку для сложных специальностей
                
                optimal_load = target_utilization * complexity_factor * speciality_factor
                optimal_distribution[doctor.doctor_id] = max(0.1, min(0.95, optimal_load))
            
            return optimal_distribution

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации использования: {e}")
            return {}

    def create_optimized_schedule(self, optimal_distribution: Dict[int, float], 
                                date_range: Tuple[datetime, datetime]) -> Dict[int, List[ScheduleSlot]]:
        """Создание оптимизированного расписания"""
        try:
            start_date, end_date = date_range
            optimized_schedule = {}
            
            for doctor_id, target_load in optimal_distribution.items():
                # Получение текущего расписания врача
                schedules = self.repository.get_schedule_by_medecin(doctor_id, start_date, end_date)
                
                doctor_slots = []
                for schedule in schedules:
                    # Расчет количества слотов на основе целевой загрузки
                    target_slots = int(schedule.slots_total * target_load)
                    
                    # Создание слотов
                    slot_duration = 30  # минуты
                    current_time = schedule.time_start
                    
                    for i in range(schedule.slots_total):
                        slot_end = current_time + timedelta(minutes=slot_duration)
                        
                        slot = ScheduleSlot(
                            start_time=current_time,
                            end_time=slot_end,
                            doctor_id=doctor_id,
                            patient_id=None,
                            appointment_type="consultation",
                            complexity=0.5,
                            is_available=i < target_slots
                        )
                        
                        doctor_slots.append(slot)
                        current_time = slot_end
                
                optimized_schedule[doctor_id] = doctor_slots
            
            return optimized_schedule

        except Exception as e:
            self.logger.error(f"Ошибка создания оптимизированного расписания: {e}")
            return {}

    def calculate_improvements(self, current_metrics: List[DoctorMetrics], 
                             optimal_distribution: Dict[int, float]) -> Dict[str, float]:
        """Расчет улучшений от оптимизации"""
        try:
            # Текущие метрики
            current_avg_load = np.mean([d.current_load for d in current_metrics])
            current_avg_wait = np.mean([d.avg_wait_time for d in current_metrics])
            current_avg_utilization = np.mean([d.utilization_rate for d in current_metrics])
            
            # Оптимальные метрики
            optimal_avg_load = np.mean(list(optimal_distribution.values()))
            
            # Расчет улучшений
            load_improvement = (optimal_avg_load - current_avg_load) / current_avg_load if current_avg_load > 0 else 0
            wait_time_reduction = max(0, (current_avg_wait - 30) / current_avg_wait) if current_avg_wait > 30 else 0
            utilization_improvement = max(0, (0.85 - current_avg_utilization) / current_avg_utilization) if current_avg_utilization > 0 else 0
            
            # Общий коэффициент эффективности
            efficiency_score = (load_improvement + wait_time_reduction + utilization_improvement) / 3
            
            return {
                'wait_time_reduction': wait_time_reduction,
                'utilization_improvement': utilization_improvement,
                'efficiency_score': efficiency_score
            }

        except Exception as e:
            self.logger.error(f"Ошибка расчета улучшений: {e}")
            return {'wait_time_reduction': 0, 'utilization_improvement': 0, 'efficiency_score': 0}

    def generate_recommendations(self, doctor_metrics: List[DoctorMetrics], 
                               optimal_distribution: Dict[int, float]) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        try:
            recommendations = []
            
            # Анализ перегруженных врачей
            overloaded_doctors = [
                d for d in doctor_metrics 
                if d.current_load > 0.9
            ]
            
            if overloaded_doctors:
                recommendations.append(
                    f"Выявлено {len(overloaded_doctors)} перегруженных врачей. "
                    "Рекомендуется перераспределить нагрузку."
                )
            
            # Анализ недозагруженных врачей
            underloaded_doctors = [
                d for d in doctor_metrics 
                if d.current_load < 0.3
            ]
            
            if underloaded_doctors:
                recommendations.append(
                    f"Выявлено {len(underloaded_doctors)} недозагруженных врачей. "
                    "Рекомендуется увеличить количество приемов."
                )
            
            # Анализ времени ожидания
            high_wait_doctors = [
                d for d in doctor_metrics 
                if d.avg_wait_time > 60
            ]
            
            if high_wait_doctors:
                recommendations.append(
                    f"У {len(high_wait_doctors)} врачей высокое время ожидания. "
                    "Рекомендуется оптимизировать расписание."
                )
            
            # Общие рекомендации
            avg_load = np.mean([d.current_load for d in doctor_metrics])
            if avg_load < 0.6:
                recommendations.append("Общая загрузка низкая. Рекомендуется привлечь больше пациентов.")
            elif avg_load > 0.85:
                recommendations.append("Общая загрузка высокая. Рекомендуется расширить штат.")
            
            return recommendations

        except Exception as e:
            self.logger.error(f"Ошибка генерации рекомендаций: {e}")
            return ["Ошибка при генерации рекомендаций"]

    def calculate_average_wait_time(self, appointments: List[Appointment]) -> float:
        """Расчет среднего времени ожидания"""
        try:
            if not appointments:
                return 0.0
            
            wait_times = []
            for appointment in appointments:
                # Упрощенный расчет времени ожидания
                # В реальности должно учитываться время записи и время приема
                wait_time = 30  # Заглушка - 30 минут
                wait_times.append(wait_time)
            
            return np.mean(wait_times)

        except Exception as e:
            self.logger.error(f"Ошибка расчета времени ожидания: {e}")
            return 0.0

    def calculate_utilization_rate(self, schedules: List[Schedule], 
                                 appointments: List[Appointment]) -> float:
        """Расчет коэффициента использования времени"""
        try:
            if not schedules:
                return 0.0
            
            total_slots = sum(s.slots_total for s in schedules)
            booked_slots = sum(s.slots_booked for s in schedules)
            
            return booked_slots / total_slots if total_slots > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Ошибка расчета коэффициента использования: {e}")
            return 0.0

    def estimate_patient_satisfaction(self, appointments: List[Appointment]) -> float:
        """Оценка удовлетворенности пациентов"""
        try:
            if not appointments:
                return 0.5
            
            # Упрощенная оценка на основе отсутствия неявок
            no_shows = sum(1 for a in appointments if a.no_show_flag)
            satisfaction = 1 - (no_shows / len(appointments))
            
            return max(0.1, min(1.0, satisfaction))

        except Exception as e:
            self.logger.error(f"Ошибка оценки удовлетворенности: {e}")
            return 0.5

    def calculate_complexity_score(self, appointments: List[Appointment]) -> float:
        """Расчет сложности случаев"""
        try:
            if not appointments:
                return 0.5
            
            # Упрощенная оценка сложности на основе типа приема
            complexity_scores = {
                'consultation': 0.3,
                'examination': 0.5,
                'procedure': 0.8,
                'surgery': 0.9
            }
            
            scores = []
            for appointment in appointments:
                score = complexity_scores.get(appointment.visit_type, 0.5)
                scores.append(score)
            
            return np.mean(scores)

        except Exception as e:
            self.logger.error(f"Ошибка расчета сложности: {e}")
            return 0.5


class ResourceAllocator:
    """Распределитель ресурсов (кабинеты, оборудование)"""

    def __init__(self, repository: MedialogRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def allocate_resources(self, date: datetime, speciality: str, 
                          appointment_type: str) -> Dict[str, Any]:
        """Распределение ресурсов для приема"""
        try:
            # Получение доступных кабинетов
            available_rooms = self.get_available_rooms(date, speciality)
            
            # Получение доступного оборудования
            available_equipment = self.get_available_equipment(date, speciality)
            
            # Выбор оптимальных ресурсов
            optimal_room = self.select_optimal_room(available_rooms, appointment_type)
            optimal_equipment = self.select_optimal_equipment(available_equipment, appointment_type)
            
            return {
                'room': optimal_room,
                'equipment': optimal_equipment,
                'allocation_time': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Ошибка распределения ресурсов: {e}")
            return {}

    def get_available_rooms(self, date: datetime, speciality: str) -> List[Dict[str, Any]]:
        """Получение доступных кабинетов"""
        try:
            # Упрощенная логика - возвращаем заглушки
            rooms = [
                {'id': 1, 'name': 'Кабинет 101', 'speciality': 'терапевт', 'available': True},
                {'id': 2, 'name': 'Кабинет 102', 'speciality': 'кардиолог', 'available': True},
                {'id': 3, 'name': 'Кабинет 103', 'speciality': 'хирург', 'available': True}
            ]
            
            return [r for r in rooms if r['speciality'] == speciality and r['available']]

        except Exception as e:
            self.logger.error(f"Ошибка получения кабинетов: {e}")
            return []

    def get_available_equipment(self, date: datetime, speciality: str) -> List[Dict[str, Any]]:
        """Получение доступного оборудования"""
        try:
            # Упрощенная логика - возвращаем заглушки
            equipment = [
                {'id': 1, 'name': 'ЭКГ', 'speciality': 'кардиолог', 'available': True},
                {'id': 2, 'name': 'УЗИ', 'speciality': 'общий', 'available': True},
                {'id': 3, 'name': 'Рентген', 'speciality': 'общий', 'available': True}
            ]
            
            return [e for e in equipment if (e['speciality'] == speciality or e['speciality'] == 'общий') and e['available']]

        except Exception as e:
            self.logger.error(f"Ошибка получения оборудования: {e}")
            return []

    def select_optimal_room(self, available_rooms: List[Dict[str, Any]], 
                          appointment_type: str) -> Optional[Dict[str, Any]]:
        """Выбор оптимального кабинета"""
        try:
            if not available_rooms:
                return None
            
            # Простая логика - выбираем первый доступный
            return available_rooms[0]

        except Exception as e:
            self.logger.error(f"Ошибка выбора кабинета: {e}")
            return None

    def select_optimal_equipment(self, available_equipment: List[Dict[str, Any]], 
                               appointment_type: str) -> Optional[Dict[str, Any]]:
        """Выбор оптимального оборудования"""
        try:
            if not available_equipment:
                return None
            
            # Простая логика - выбираем первое доступное
            return available_equipment[0]

        except Exception as e:
            self.logger.error(f"Ошибка выбора оборудования: {e}")
            return None


class PerformanceTracker:
    """Трекер производительности врачей"""

    def __init__(self, repository: MedialogRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def track_doctor_performance(self, doctor_id: int, 
                               date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Отслеживание производительности врача"""
        try:
            start_date, end_date = date_range
            
            # Получение данных врача
            appointments = self.repository.get_appointments_by_medecin(doctor_id, start_date, end_date)
            consultations = self.repository.get_consultations_by_medecin(doctor_id, start_date, end_date)
            
            # Расчет метрик производительности
            total_appointments = len(appointments)
            completed_appointments = len([a for a in appointments if not a.no_show_flag])
            completion_rate = completed_appointments / total_appointments if total_appointments > 0 else 0
            
            # Среднее время приема
            avg_consultation_time = self.calculate_avg_consultation_time(consultations)
            
            # Удовлетворенность пациентов
            patient_satisfaction = self.calculate_patient_satisfaction(appointments)
            
            # Эффективность использования времени
            time_efficiency = self.calculate_time_efficiency(appointments, consultations)
            
            return {
                'doctor_id': doctor_id,
                'total_appointments': total_appointments,
                'completion_rate': completion_rate,
                'avg_consultation_time': avg_consultation_time,
                'patient_satisfaction': patient_satisfaction,
                'time_efficiency': time_efficiency,
                'performance_score': (completion_rate + patient_satisfaction + time_efficiency) / 3
            }

        except Exception as e:
            self.logger.error(f"Ошибка отслеживания производительности: {e}")
            return {}

    def calculate_avg_consultation_time(self, consultations: List[Consultation]) -> float:
        """Расчет среднего времени консультации"""
        try:
            if not consultations:
                return 30.0  # Стандартное время
            
            total_time = sum(c.duration or 30 for c in consultations)
            return total_time / len(consultations)

        except Exception as e:
            self.logger.error(f"Ошибка расчета времени консультации: {e}")
            return 30.0

    def calculate_patient_satisfaction(self, appointments: List[Appointment]) -> float:
        """Расчет удовлетворенности пациентов"""
        try:
            if not appointments:
                return 0.5
            
            # Упрощенная оценка на основе отсутствия неявок
            no_shows = sum(1 for a in appointments if a.no_show_flag)
            satisfaction = 1 - (no_shows / len(appointments))
            
            return max(0.1, min(1.0, satisfaction))

        except Exception as e:
            self.logger.error(f"Ошибка расчета удовлетворенности: {e}")
            return 0.5

    def calculate_time_efficiency(self, appointments: List[Appointment], 
                                consultations: List[Consultation]) -> float:
        """Расчет эффективности использования времени"""
        try:
            if not appointments:
                return 0.5
            
            # Упрощенная оценка эффективности
            total_slots = len(appointments)
            used_slots = len([a for a in appointments if not a.no_show_flag])
            
            efficiency = used_slots / total_slots if total_slots > 0 else 0.5
            return max(0.1, min(1.0, efficiency))

        except Exception as e:
            self.logger.error(f"Ошибка расчета эффективности времени: {e}")
            return 0.5


class DoctorLoadOptimizationService:
    """Основной сервис оптимизации загрузки врачей"""

    def __init__(self, repository: MedialogRepository):
        self.repository = repository
        self.schedule_optimizer = ScheduleOptimizer(repository)
        self.resource_allocator = ResourceAllocator(repository)
        self.performance_tracker = PerformanceTracker(repository)
        self.load_balancer = WeightedLoadBalancer()
        self.logger = logging.getLogger(__name__)

    def optimize_doctor_schedule(self, date_range: Tuple[datetime, datetime], 
                               objective: OptimizationObjective = OptimizationObjective.BALANCE_LOAD) -> OptimizationResult:
        """Основной метод оптимизации расписания врачей"""
        try:
            # Оптимизация расписания
            optimization_result = self.schedule_optimizer.optimize_doctor_load(date_range, objective)
            
            # Дополнительная оптимизация ресурсов
            self.optimize_resource_allocation(optimization_result.optimized_schedule)
            
            return optimization_result

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации расписания врачей: {e}")
            raise

    def optimize_resource_allocation(self, optimized_schedule: Dict[int, List[ScheduleSlot]]):
        """Оптимизация распределения ресурсов"""
        try:
            for doctor_id, slots in optimized_schedule.items():
                for slot in slots:
                    if not slot.is_available:
                        # Распределение ресурсов для занятого слота
                        resources = self.resource_allocator.allocate_resources(
                            slot.start_time, 
                            "терапевт",  # Упрощенно
                            slot.appointment_type
                        )
                        # Здесь можно сохранить распределение ресурсов

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации ресурсов: {e}")

    def get_doctor_performance_report(self, doctor_id: int, 
                                    date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Получение отчета о производительности врача"""
        try:
            return self.performance_tracker.track_doctor_performance(doctor_id, date_range)

        except Exception as e:
            self.logger.error(f"Ошибка получения отчета производительности: {e}")
            return {}

    def get_optimal_doctor_for_appointment(self, speciality: str, complexity: float, 
                                         date: datetime) -> Optional[int]:
        """Получение оптимального врача для записи"""
        try:
            # Получение доступных врачей специальности
            doctors = self.repository.get_medecins_by_department(speciality)
            
            if not doctors:
                return None
            
            # Выбор оптимального врача
            return self.load_balancer.get_optimal_doctor(speciality, complexity)

        except Exception as e:
            self.logger.error(f"Ошибка выбора оптимального врача: {e}")
            return None 