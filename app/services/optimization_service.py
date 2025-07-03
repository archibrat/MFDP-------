"""
Сервис оптимизации ресурсов и распределения нагрузки
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    DOCTOR = "doctor"
    ROOM = "room"
    EQUIPMENT = "equipment"
    NURSE = "nurse"


@dataclass
class Resource:
    id: str
    name: str
    type: ResourceType
    capacity: int
    availability_hours: List[Tuple[int, int]]  # (start_hour, end_hour)
    cost_per_hour: float = 0.0
    specialization: Optional[str] = None


@dataclass
class OptimizationRequest:
    date: datetime
    patient_count: int
    appointment_types: Dict[str, int]
    constraints: Dict[str, any]


@dataclass
class OptimizationResult:
    schedule: Dict[str, List[Dict]]
    resource_utilization: Dict[str, float]
    efficiency_score: float
    cost_estimate: float
    recommendations: List[str]


class ResourceOptimizer:
    """Оптимизатор ресурсов медицинского учреждения"""
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.historical_data = []
        self.optimization_cache = {}
    
    def add_resource(self, resource: Resource):
        """Добавление ресурса в систему"""
        self.resources[resource.id] = resource
        logger.info(f"Added resource: {resource.name} ({resource.type.value})")
    
    def optimize_schedule(self, request: OptimizationRequest) -> OptimizationResult:
        """Основной метод оптимизации расписания"""
        try:
            # Анализ текущих ресурсов
            available_resources = self._get_available_resources(request.date)
            
            # Прогнозирование потребности в ресурсах
            resource_demand = self._calculate_resource_demand(request)
            
            # Оптимизация распределения
            optimal_schedule = self._optimize_resource_allocation(
                available_resources, 
                resource_demand,
                request.constraints
            )
            
            # Расчет метрик эффективности
            utilization = self._calculate_utilization(optimal_schedule)
            efficiency = self._calculate_efficiency_score(optimal_schedule, resource_demand)
            cost = self._calculate_cost_estimate(optimal_schedule)
            
            # Генерация рекомендаций
            recommendations = self._generate_recommendations(
                optimal_schedule, 
                resource_demand, 
                utilization
            )
            
            return OptimizationResult(
                schedule=optimal_schedule,
                resource_utilization=utilization,
                efficiency_score=efficiency,
                cost_estimate=cost,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def _get_available_resources(self, date: datetime) -> Dict[str, Resource]:
        """Получение доступных ресурсов на дату"""
        available = {}
        for resource_id, resource in self.resources.items():
            # Проверка доступности (упрощенная логика)
            if self._is_resource_available(resource, date):
                available[resource_id] = resource
        return available
    
    def _is_resource_available(self, resource: Resource, date: datetime) -> bool:
        """Проверка доступности ресурса"""
        # Упрощенная проверка - ресурс доступен в рабочие часы
        current_hour = date.hour
        for start_hour, end_hour in resource.availability_hours:
            if start_hour <= current_hour <= end_hour:
                return True
        return False
    
    def _calculate_resource_demand(self, request: OptimizationRequest) -> Dict[str, int]:
        """Расчет потребности в ресурсах"""
        demand = {}
        
        # Базовые коэффициенты потребности
        base_ratios = {
            ResourceType.DOCTOR.value: 0.8,  # 1 врач на 1.25 пациента
            ResourceType.ROOM.value: 0.9,    # 1 кабинет на 1.1 пациента
            ResourceType.NURSE.value: 0.4,   # 1 медсестра на 2.5 пациента
            ResourceType.EQUIPMENT.value: 0.3 # 1 оборудование на 3.3 пациента
        }
        
        for resource_type, ratio in base_ratios.items():
            demand[resource_type] = max(1, int(request.patient_count * ratio))
        
        # Корректировка по типам приемов
        for appointment_type, count in request.appointment_types.items():
            if appointment_type == "specialist":
                demand[ResourceType.DOCTOR.value] += count * 0.2
            elif appointment_type == "diagnostic":
                demand[ResourceType.EQUIPMENT.value] += count * 0.5
        
        return demand
    
    def _optimize_resource_allocation(self, 
                                   available_resources: Dict[str, Resource],
                                   demand: Dict[str, int],
                                   constraints: Dict) -> Dict[str, List[Dict]]:
        """Оптимизация распределения ресурсов"""
        schedule = {resource_type: [] for resource_type in demand.keys()}
        
        # Простой алгоритм распределения по приоритетам
        priority_order = [
            ResourceType.DOCTOR.value,
            ResourceType.ROOM.value,
            ResourceType.NURSE.value,
            ResourceType.EQUIPMENT.value
        ]
        
        for resource_type in priority_order:
            if resource_type in demand:
                allocated = self._allocate_resource_type(
                    available_resources,
                    resource_type,
                    demand[resource_type],
                    constraints
                )
                schedule[resource_type] = allocated
        
        return schedule
    
    def _allocate_resource_type(self, 
                              available_resources: Dict[str, Resource],
                              resource_type: str,
                              needed_count: int,
                              constraints: Dict) -> List[Dict]:
        """Распределение конкретного типа ресурса"""
        allocated = []
        resources_of_type = [
            r for r in available_resources.values() 
            if r.type.value == resource_type
        ]
        
        # Сортировка по эффективности (cost_per_hour)
        resources_of_type.sort(key=lambda x: x.cost_per_hour)
        
        for resource in resources_of_type[:needed_count]:
            allocation = {
                "resource_id": resource.id,
                "resource_name": resource.name,
                "utilization_hours": 8,  # Стандартный рабочий день
                "capacity_used": min(resource.capacity, 100),
                "cost": resource.cost_per_hour * 8
            }
            allocated.append(allocation)
        
        return allocated
    
    def _calculate_utilization(self, schedule: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Расчет коэффициента использования ресурсов"""
        utilization = {}
        
        for resource_type, allocations in schedule.items():
            if allocations:
                total_capacity = sum(alloc["capacity_used"] for alloc in allocations)
                total_available = len(allocations) * 100  # 100% capacity per resource
                utilization[resource_type] = (total_capacity / total_available) * 100
            else:
                utilization[resource_type] = 0.0
        
        return utilization
    
    def _calculate_efficiency_score(self, 
                                  schedule: Dict[str, List[Dict]], 
                                  demand: Dict[str, int]) -> float:
        """Расчет общего показателя эффективности"""
        efficiency_scores = []
        
        for resource_type, demand_count in demand.items():
            allocated_count = len(schedule.get(resource_type, []))
            if demand_count > 0:
                score = min(allocated_count / demand_count, 1.0) * 100
                efficiency_scores.append(score)
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
    
    def _calculate_cost_estimate(self, schedule: Dict[str, List[Dict]]) -> float:
        """Расчет стоимости использования ресурсов"""
        total_cost = 0.0
        
        for resource_type, allocations in schedule.items():
            for allocation in allocations:
                total_cost += allocation.get("cost", 0.0)
        
        return total_cost
    
    def _generate_recommendations(self, 
                                schedule: Dict[str, List[Dict]],
                                demand: Dict[str, int],
                                utilization: Dict[str, float]) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        recommendations = []
        
        # Анализ загруженности
        for resource_type, util_percent in utilization.items():
            if util_percent > 90:
                recommendations.append(
                    f"Высокая загрузка {resource_type}: {util_percent:.1f}%. "
                    f"Рекомендуется увеличить количество ресурсов."
                )
            elif util_percent < 50:
                recommendations.append(
                    f"Низкая загрузка {resource_type}: {util_percent:.1f}%. "
                    f"Возможна оптимизация расходов."
                )
        
        # Анализ нехватки ресурсов
        for resource_type, demand_count in demand.items():
            allocated_count = len(schedule.get(resource_type, []))
            if allocated_count < demand_count:
                shortage = demand_count - allocated_count
                recommendations.append(
                    f"Нехватка {resource_type}: {shortage} единиц. "
                    f"Необходимо дополнительное планирование."
                )
        
        # Общие рекомендации
        if not recommendations:
            recommendations.append("Оптимальное распределение ресурсов достигнуто.")
        
        return recommendations


class LoadBalancer:
    """Балансировщик нагрузки между ресурсами"""
    
    def __init__(self):
        self.load_metrics = {}
        self.thresholds = {
            "high_load": 80.0,
            "critical_load": 95.0
        }
    
    def balance_load(self, current_loads: Dict[str, float]) -> Dict[str, str]:
        """Балансировка нагрузки между ресурсами"""
        recommendations = {}
        
        # Поиск перегруженных ресурсов
        overloaded = {
            resource_id: load 
            for resource_id, load in current_loads.items() 
            if load > self.thresholds["high_load"]
        }
        
        # Поиск недогруженных ресурсов
        underloaded = {
            resource_id: load 
            for resource_id, load in current_loads.items() 
            if load < 50.0
        }
        
        # Генерация рекомендаций по перераспределению
        if overloaded and underloaded:
            for overloaded_id, over_load in overloaded.items():
                for underloaded_id, under_load in underloaded.items():
                    if over_load - under_load > 30:
                        recommendations[f"{overloaded_id}_to_{underloaded_id}"] = (
                            f"Перенести часть нагрузки с {overloaded_id} "
                            f"({over_load:.1f}%) на {underloaded_id} ({under_load:.1f}%)"
                        )
        
        return recommendations


# Глобальный экземпляр оптимизатора
optimizer_service = ResourceOptimizer()
load_balancer = LoadBalancer()


def get_optimizer_service() -> ResourceOptimizer:
    """Получение экземпляра сервиса оптимизации"""
    return optimizer_service


def get_load_balancer() -> LoadBalancer:
    """Получение экземпляра балансировщика нагрузки"""
    return load_balancer 