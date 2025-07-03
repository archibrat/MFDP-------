"""
Балансировщики нагрузки для врачей
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from enum import Enum
import logging
import numpy as np

from app.services.doctor_load_optimization.schemas import DoctorProfile


class LoadBalancingStrategy(str, Enum):
    """Стратегии балансировки нагрузки"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    ADAPTIVE = "adaptive"


class LoadBalancer(ABC):
    """Абстрактный балансировщик нагрузки"""

    @abstractmethod
    def balance_load(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Балансировка нагрузки между врачами"""
        pass

    @abstractmethod
    def get_optimal_doctor(self, speciality: str, complexity: float,
                          doctor_profiles: List[DoctorProfile]) -> Optional[int]:
        """Получение оптимального врача для приема"""
        pass


class RoundRobinLoadBalancer(LoadBalancer):
    """Балансировщик нагрузки по принципу Round Robin"""

    def __init__(self):
        self.current_index = 0
        self.logger = logging.getLogger(__name__)

    def balance_load(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Балансировка нагрузки методом Round Robin"""
        try:
            if not doctor_profiles:
                return {}

            # Сортировка по загруженности
            sorted_doctors = sorted(doctor_profiles, key=lambda x: x.current_load)
            
            # Равномерное распределение нагрузки
            total_load = sum(d.current_load for d in sorted_doctors)
            avg_load = total_load / len(sorted_doctors)
            
            balanced_loads = {}
            for i, doctor in enumerate(sorted_doctors):
                # Небольшая вариация для равномерности
                variation = (i - len(sorted_doctors) / 2) * 0.05
                balanced_loads[doctor.doctor_id] = max(0.1, min(0.95, avg_load + variation))
            
            return balanced_loads

        except Exception as e:
            self.logger.error(f"Ошибка балансировки нагрузки Round Robin: {e}")
            return {}

    def get_optimal_doctor(self, speciality: str, complexity: float,
                          doctor_profiles: List[DoctorProfile]) -> Optional[int]:
        """Получение оптимального врача для приема"""
        try:
            # Фильтрация по специальности
            suitable_doctors = [
                d for d in doctor_profiles 
                if d.speciality.lower() == speciality.lower()
            ]
            
            if not suitable_doctors:
                return None
            
            # Round Robin выбор
            self.current_index = (self.current_index + 1) % len(suitable_doctors)
            return suitable_doctors[self.current_index].doctor_id

        except Exception as e:
            self.logger.error(f"Ошибка выбора врача Round Robin: {e}")
            return None


class WeightedLoadBalancer(LoadBalancer):
    """Балансировщик нагрузки с учетом весов"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'load': 0.3,
            'satisfaction': 0.2,
            'utilization': 0.2,
            'experience': 0.15,
            'complexity': 0.15
        }
        self.logger = logging.getLogger(__name__)

    def balance_load(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Балансировка нагрузки с учетом весов"""
        try:
            if not doctor_profiles:
                return {}

            balanced_loads = {}
            
            for doctor in doctor_profiles:
                # Расчет взвешенной оценки
                weighted_score = self._calculate_weighted_score(doctor)
                
                # Преобразование в целевую нагрузку
                target_load = 0.7 + (weighted_score - 0.5) * 0.4  # 0.5-0.9 диапазон
                balanced_loads[doctor.doctor_id] = max(0.1, min(0.95, target_load))
            
            return balanced_loads

        except Exception as e:
            self.logger.error(f"Ошибка взвешенной балансировки: {e}")
            return {}

    def get_optimal_doctor(self, speciality: str, complexity: float,
                          doctor_profiles: List[DoctorProfile]) -> Optional[int]:
        """Получение оптимального врача с учетом сложности случая"""
        try:
            # Фильтрация по специальности
            suitable_doctors = [
                d for d in doctor_profiles 
                if d.speciality.lower() == speciality.lower()
            ]
            
            if not suitable_doctors:
                return None
            
            # Расчет оценок для каждого врача
            doctor_scores = []
            for doctor in suitable_doctors:
                score = self._calculate_doctor_score(doctor, complexity)
                doctor_scores.append((doctor.doctor_id, score))
            
            # Выбор врача с наилучшей оценкой
            if doctor_scores:
                best_doctor = max(doctor_scores, key=lambda x: x[1])
                return best_doctor[0]
            
            return None

        except Exception as e:
            self.logger.error(f"Ошибка выбора врача Weighted: {e}")
            return None

    def _calculate_weighted_score(self, doctor: DoctorProfile) -> float:
        """Расчет взвешенной оценки врача"""
        try:
            # Нормализованные значения
            normalized_load = 1 - doctor.current_load  # Инвертируем для минимизации
            normalized_satisfaction = doctor.patient_satisfaction
            normalized_utilization = doctor.utilization_rate
            normalized_experience = min(doctor.experience_years / 20, 1.0)  # Нормализация опыта
            normalized_complexity = doctor.complexity_score
            
            # Взвешенная оценка
            weighted_score = (
                normalized_load * self.weights['load'] +
                normalized_satisfaction * self.weights['satisfaction'] +
                normalized_utilization * self.weights['utilization'] +
                normalized_experience * self.weights['experience'] +
                normalized_complexity * self.weights['complexity']
            )
            
            return max(0.0, min(1.0, weighted_score))

        except Exception as e:
            self.logger.error(f"Ошибка расчета взвешенной оценки: {e}")
            return 0.5

    def _calculate_doctor_score(self, doctor: DoctorProfile, complexity: float) -> float:
        """Расчет оценки врача для конкретного случая"""
        try:
            # Базовый взвешенный скор
            base_score = self._calculate_weighted_score(doctor)
            
            # Корректировка по сложности
            complexity_match = 1.0 - abs(doctor.complexity_score - complexity)
            
            # Корректировка по загруженности
            load_penalty = doctor.current_load * 0.3
            
            # Итоговая оценка
            final_score = base_score * 0.6 + complexity_match * 0.3 - load_penalty * 0.1
            
            return max(0.0, min(1.0, final_score))

        except Exception as e:
            self.logger.error(f"Ошибка расчета оценки врача: {e}")
            return 0.5


class LeastConnectionsLoadBalancer(LoadBalancer):
    """Балансировщик нагрузки по принципу наименьшего количества соединений"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def balance_load(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Балансировка нагрузки по наименьшему количеству соединений"""
        try:
            if not doctor_profiles:
                return {}

            # Сортировка по загруженности (наименее загруженные первые)
            sorted_doctors = sorted(doctor_profiles, key=lambda x: x.current_load)
            
            balanced_loads = {}
            total_doctors = len(sorted_doctors)
            
            for i, doctor in enumerate(sorted_doctors):
                # Распределение нагрузки с приоритетом для менее загруженных
                priority_factor = (total_doctors - i) / total_doctors
                target_load = 0.5 + priority_factor * 0.3  # 0.5-0.8 диапазон
                balanced_loads[doctor.doctor_id] = max(0.1, min(0.95, target_load))
            
            return balanced_loads

        except Exception as e:
            self.logger.error(f"Ошибка балансировки Least Connections: {e}")
            return {}

    def get_optimal_doctor(self, speciality: str, complexity: float,
                          doctor_profiles: List[DoctorProfile]) -> Optional[int]:
        """Получение наименее загруженного врача"""
        try:
            # Фильтрация по специальности
            suitable_doctors = [
                d for d in doctor_profiles 
                if d.speciality.lower() == speciality.lower()
            ]
            
            if not suitable_doctors:
                return None
            
            # Выбор наименее загруженного врача
            least_loaded = min(suitable_doctors, key=lambda x: x.current_load)
            return least_loaded.doctor_id

        except Exception as e:
            self.logger.error(f"Ошибка выбора врача Least Connections: {e}")
            return None


class ResponseTimeLoadBalancer(LoadBalancer):
    """Балансировщик нагрузки по времени отклика"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def balance_load(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Балансировка нагрузки по времени отклика"""
        try:
            if not doctor_profiles:
                return {}

            # Сортировка по времени ожидания (наименьшее время первые)
            sorted_doctors = sorted(doctor_profiles, key=lambda x: x.avg_wait_time)
            
            balanced_loads = {}
            total_doctors = len(sorted_doctors)
            
            for i, doctor in enumerate(sorted_doctors):
                # Распределение нагрузки с приоритетом для быстрых врачей
                speed_factor = (total_doctors - i) / total_doctors
                target_load = 0.6 + speed_factor * 0.3  # 0.6-0.9 диапазон
                balanced_loads[doctor.doctor_id] = max(0.1, min(0.95, target_load))
            
            return balanced_loads

        except Exception as e:
            self.logger.error(f"Ошибка балансировки Response Time: {e}")
            return {}

    def get_optimal_doctor(self, speciality: str, complexity: float,
                          doctor_profiles: List[DoctorProfile]) -> Optional[int]:
        """Получение врача с наименьшим временем ожидания"""
        try:
            # Фильтрация по специальности
            suitable_doctors = [
                d for d in doctor_profiles 
                if d.speciality.lower() == speciality.lower()
            ]
            
            if not suitable_doctors:
                return None
            
            # Выбор врача с наименьшим временем ожидания
            fastest_doctor = min(suitable_doctors, key=lambda x: x.avg_wait_time)
            return fastest_doctor.doctor_id

        except Exception as e:
            self.logger.error(f"Ошибка выбора врача Response Time: {e}")
            return None


class AdaptiveLoadBalancer(LoadBalancer):
    """Адаптивный балансировщик нагрузки"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategies = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinLoadBalancer(),
            LoadBalancingStrategy.WEIGHTED: WeightedLoadBalancer(),
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsLoadBalancer(),
            LoadBalancingStrategy.RESPONSE_TIME: ResponseTimeLoadBalancer()
        }

    def balance_load(self, doctor_profiles: List[DoctorProfile]) -> Dict[int, float]:
        """Адаптивная балансировка нагрузки"""
        try:
            if not doctor_profiles:
                return {}

            # Анализ текущего состояния
            strategy = self._select_best_strategy(doctor_profiles)
            
            # Применение выбранной стратегии
            balancer = self.strategies[strategy]
            return balancer.balance_load(doctor_profiles)

        except Exception as e:
            self.logger.error(f"Ошибка адаптивной балансировки: {e}")
            return {}

    def get_optimal_doctor(self, speciality: str, complexity: float,
                          doctor_profiles: List[DoctorProfile]) -> Optional[int]:
        """Адаптивный выбор оптимального врача"""
        try:
            # Анализ текущего состояния
            strategy = self._select_best_strategy(doctor_profiles)
            
            # Применение выбранной стратегии
            balancer = self.strategies[strategy]
            return balancer.get_optimal_doctor(speciality, complexity, doctor_profiles)

        except Exception as e:
            self.logger.error(f"Ошибка адаптивного выбора врача: {e}")
            return None

    def _select_best_strategy(self, doctor_profiles: List[DoctorProfile]) -> LoadBalancingStrategy:
        """Выбор лучшей стратегии на основе текущего состояния"""
        try:
            if not doctor_profiles:
                return LoadBalancingStrategy.ROUND_ROBIN
            
            # Анализ метрик
            avg_load = sum(d.current_load for d in doctor_profiles) / len(doctor_profiles)
            avg_wait_time = sum(d.avg_wait_time for d in doctor_profiles) / len(doctor_profiles)
            load_variance = np.var([d.current_load for d in doctor_profiles])
            
            # Выбор стратегии на основе метрик
            if load_variance > 0.2:  # Высокая неравномерность нагрузки
                return LoadBalancingStrategy.LEAST_CONNECTIONS
            elif avg_wait_time > 30:  # Высокое время ожидания
                return LoadBalancingStrategy.RESPONSE_TIME
            elif avg_load > 0.8:  # Высокая загруженность
                return LoadBalancingStrategy.WEIGHTED
            else:
                return LoadBalancingStrategy.ROUND_ROBIN

        except Exception as e:
            self.logger.error(f"Ошибка выбора стратегии: {e}")
            return LoadBalancingStrategy.ROUND_ROBIN


def create_load_balancer(strategy: LoadBalancingStrategy) -> LoadBalancer:
    """Фабрика для создания балансировщиков нагрузки"""
    if strategy == LoadBalancingStrategy.ROUND_ROBIN:
        return RoundRobinLoadBalancer()
    elif strategy == LoadBalancingStrategy.WEIGHTED:
        return WeightedLoadBalancer()
    elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
        return LeastConnectionsLoadBalancer()
    elif strategy == LoadBalancingStrategy.RESPONSE_TIME:
        return ResponseTimeLoadBalancer()
    elif strategy == LoadBalancingStrategy.ADAPTIVE:
        return AdaptiveLoadBalancer()
    else:
        raise ValueError(f"Неизвестная стратегия балансировки: {strategy}") 