"""
Модуль распределения ресурсов для врачей
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import random

from app.services.doctor_load_optimization.schemas import (
    ResourceAllocationRequest, ResourceAllocationResult, ScheduleSlot
)


class ResourceType(str, Enum):
    """Типы ресурсов"""
    DOCTOR = "doctor"
    ROOM = "room"
    EQUIPMENT = "equipment"
    NURSE = "nurse"


class Resource:
    """Модель ресурса"""
    
    def __init__(self, resource_id: int, name: str, resource_type: ResourceType,
                 capacity: int = 1, speciality: Optional[str] = None):
        self.resource_id = resource_id
        self.name = name
        self.resource_type = resource_type
        self.capacity = capacity
        self.speciality = speciality
        self.availability_schedule = {}  # {date: [time_slots]}
        self.current_utilization = 0.0


class ResourceAllocator:
    """Распределитель ресурсов"""

    def __init__(self):
        self.resources = {}  # {resource_id: Resource}
        self.logger = logging.getLogger(__name__)
        self._initialize_default_resources()

    def _initialize_default_resources(self):
        """Инициализация ресурсов по умолчанию"""
        try:
            # Кабинеты
            for i in range(1, 21):
                room = Resource(
                    resource_id=i,
                    name=f"Кабинет {i}",
                    resource_type=ResourceType.ROOM,
                    capacity=1
                )
                self.resources[i] = room
            
            # Оборудование
            equipment_list = [
                "ЭКГ", "УЗИ", "Рентген", "Тонометр", "Стетоскоп",
                "Отоскоп", "Офтальмоскоп", "Неврологический молоток"
            ]
            
            for i, equipment_name in enumerate(equipment_list, start=100):
                equipment = Resource(
                    resource_id=i,
                    name=equipment_name,
                    resource_type=ResourceType.EQUIPMENT,
                    capacity=1
                )
                self.resources[i] = equipment
            
            # Медсестры
            for i in range(1, 11):
                nurse = Resource(
                    resource_id=i + 200,
                    name=f"Медсестра {i}",
                    resource_type=ResourceType.NURSE,
                    capacity=1
                )
                self.resources[i + 200] = nurse
                
        except Exception as e:
            self.logger.error(f"Ошибка инициализации ресурсов: {e}")

    def allocate_resources(self, request: ResourceAllocationRequest) -> ResourceAllocationResult:
        """Распределение ресурсов для приема"""
        try:
            # Получение доступных ресурсов
            available_doctors = self._get_available_doctors(request.speciality, request.date)
            available_rooms = self._get_available_rooms(request.date, request.speciality)
            available_equipment = self._get_available_equipment(request.date, request.speciality)
            
            if not available_doctors or not available_rooms:
                raise ValueError("Недостаточно доступных ресурсов")
            
            # Выбор оптимальных ресурсов
            selected_doctor = self._select_optimal_doctor(available_doctors, request)
            selected_room = self._select_optimal_room(available_rooms, request)
            selected_equipment = self._select_optimal_equipment(available_equipment, request)
            
            # Создание слотов расписания
            schedule_slots = self._create_schedule_slots(
                selected_doctor, selected_room, selected_equipment, request
            )
            
            # Расчет эффективности
            efficiency_score = self._calculate_allocation_efficiency(
                selected_doctor, selected_room, selected_equipment, request
            )
            
            # Расчет затрат
            cost_estimate = self._calculate_cost_estimate(
                selected_doctor, selected_room, selected_equipment, request
            )
            
            # Генерация рекомендаций
            recommendations = self._generate_allocation_recommendations(
                selected_doctor, selected_room, selected_equipment, request
            )
            
            return ResourceAllocationResult(
                allocated_doctors=[selected_doctor],
                allocated_rooms=[selected_room],
                schedule_slots=schedule_slots,
                efficiency_score=efficiency_score,
                cost_estimate=cost_estimate,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Ошибка распределения ресурсов: {e}")
            raise

    def _get_available_doctors(self, speciality: str, date: datetime) -> List[int]:
        """Получение доступных врачей"""
        try:
            # Заглушка - в реальности здесь был бы запрос к БД
            available_doctors = []
            
            # Симуляция доступности врачей
            for i in range(1, 11):
                if random.random() > 0.3:  # 70% вероятность доступности
                    available_doctors.append(i)
            
            return available_doctors

        except Exception as e:
            self.logger.error(f"Ошибка получения доступных врачей: {e}")
            return []

    def _get_available_rooms(self, date: datetime, speciality: str) -> List[int]:
        """Получение доступных кабинетов"""
        try:
            available_rooms = []
            
            # Получение кабинетов по типу ресурса
            for resource_id, resource in self.resources.items():
                if (resource.resource_type == ResourceType.ROOM and 
                    self._is_resource_available(resource, date)):
                    available_rooms.append(resource_id)
            
            return available_rooms

        except Exception as e:
            self.logger.error(f"Ошибка получения доступных кабинетов: {e}")
            return []

    def _get_available_equipment(self, date: datetime, speciality: str) -> List[int]:
        """Получение доступного оборудования"""
        try:
            available_equipment = []
            
            # Получение оборудования по типу ресурса
            for resource_id, resource in self.resources.items():
                if (resource.resource_type == ResourceType.EQUIPMENT and 
                    self._is_resource_available(resource, date)):
                    available_equipment.append(resource_id)
            
            return available_equipment

        except Exception as e:
            self.logger.error(f"Ошибка получения доступного оборудования: {e}")
            return []

    def _is_resource_available(self, resource: Resource, date: datetime) -> bool:
        """Проверка доступности ресурса"""
        try:
            # Упрощенная проверка доступности
            if date.weekday() >= 5:  # Выходные
                return False
            
            # Проверка по расписанию
            if date.date() in resource.availability_schedule:
                return len(resource.availability_schedule[date.date()]) > 0
            
            # По умолчанию доступен в рабочие часы
            return True

        except Exception as e:
            self.logger.error(f"Ошибка проверки доступности ресурса: {e}")
            return False

    def _select_optimal_doctor(self, available_doctors: List[int], 
                             request: ResourceAllocationRequest) -> int:
        """Выбор оптимального врача"""
        try:
            if not available_doctors:
                raise ValueError("Нет доступных врачей")
            
            # Упрощенная логика выбора - случайный выбор
            # В реальности здесь была бы сложная логика с учетом:
            # - Опыта врача
            # - Специализации
            # - Текущей загруженности
            # - Сложности случая
            
            return random.choice(available_doctors)

        except Exception as e:
            self.logger.error(f"Ошибка выбора врача: {e}")
            raise

    def _select_optimal_room(self, available_rooms: List[int], 
                           request: ResourceAllocationRequest) -> int:
        """Выбор оптимального кабинета"""
        try:
            if not available_rooms:
                raise ValueError("Нет доступных кабинетов")
            
            # Упрощенная логика выбора
            # В реальности учитывались бы:
            # - Размер кабинета
            # - Оборудование в кабинете
            # - Расположение
            # - Специализация кабинета
            
            return random.choice(available_rooms)

        except Exception as e:
            self.logger.error(f"Ошибка выбора кабинета: {e}")
            raise

    def _select_optimal_equipment(self, available_equipment: List[int], 
                                request: ResourceAllocationRequest) -> Optional[int]:
        """Выбор оптимального оборудования"""
        try:
            if not available_equipment:
                return None
            
            # Выбор оборудования в зависимости от типа приема
            if "обследование" in request.appointment_type.lower():
                # Для обследований нужна диагностическая аппаратура
                diagnostic_equipment = [e for e in available_equipment if e in [101, 102, 103]]  # ЭКГ, УЗИ, Рентген
                if diagnostic_equipment:
                    return random.choice(diagnostic_equipment)
            
            # Для обычных приемов - базовое оборудование
            basic_equipment = [e for e in available_equipment if e in [104, 105]]  # Тонометр, Стетоскоп
            if basic_equipment:
                return random.choice(basic_equipment)
            
            return random.choice(available_equipment) if available_equipment else None

        except Exception as e:
            self.logger.error(f"Ошибка выбора оборудования: {e}")
            return None

    def _create_schedule_slots(self, doctor_id: int, room_id: int, 
                             equipment_id: Optional[int], 
                             request: ResourceAllocationRequest) -> List[ScheduleSlot]:
        """Создание слотов расписания"""
        try:
            slots = []
            
            # Создание слотов на основе количества пациентов и длительности
            slot_duration = timedelta(minutes=request.duration_minutes)
            patients_per_slot = min(request.patient_count, 3)  # Максимум 3 пациента на слот
            
            current_time = request.date.replace(hour=9, minute=0)  # Начало рабочего дня
            end_time = request.date.replace(hour=17, minute=0)  # Конец рабочего дня
            
            while current_time < end_time and len(slots) < request.patient_count:
                slot = ScheduleSlot(
                    start_time=current_time,
                    end_time=current_time + slot_duration,
                    doctor_id=doctor_id,
                    patient_id=None,  # Будет назначен позже
                    appointment_type=request.appointment_type,
                    complexity=request.complexity_level,
                    is_available=True,
                    room_id=room_id
                )
                
                slots.append(slot)
                current_time += slot_duration
            
            return slots

        except Exception as e:
            self.logger.error(f"Ошибка создания слотов расписания: {e}")
            return []

    def _calculate_allocation_efficiency(self, doctor_id: int, room_id: int,
                                       equipment_id: Optional[int],
                                       request: ResourceAllocationRequest) -> float:
        """Расчет эффективности распределения ресурсов"""
        try:
            efficiency_factors = []
            
            # Фактор использования времени
            time_efficiency = min(1.0, request.duration_minutes / 60.0)
            efficiency_factors.append(time_efficiency)
            
            # Фактор сложности случая
            complexity_efficiency = 1.0 - abs(request.complexity_level - 0.5)
            efficiency_factors.append(complexity_efficiency)
            
            # Фактор количества пациентов
            patient_efficiency = min(1.0, request.patient_count / 10.0)
            efficiency_factors.append(patient_efficiency)
            
            # Фактор оборудования
            equipment_efficiency = 1.0 if equipment_id else 0.8
            efficiency_factors.append(equipment_efficiency)
            
            # Средняя эффективность
            overall_efficiency = sum(efficiency_factors) / len(efficiency_factors)
            
            return max(0.0, min(1.0, overall_efficiency))

        except Exception as e:
            self.logger.error(f"Ошибка расчета эффективности: {e}")
            return 0.5

    def _calculate_cost_estimate(self, doctor_id: int, room_id: int,
                               equipment_id: Optional[int],
                               request: ResourceAllocationRequest) -> float:
        """Расчет оценки затрат"""
        try:
            base_cost = 1000  # Базовая стоимость приема
            
            # Дополнительные затраты за длительность
            duration_cost = request.duration_minutes * 10
            
            # Затраты на оборудование
            equipment_cost = 200 if equipment_id else 0
            
            # Затраты на сложность
            complexity_cost = request.complexity_level * 500
            
            # Затраты на количество пациентов
            patient_cost = request.patient_count * 100
            
            total_cost = base_cost + duration_cost + equipment_cost + complexity_cost + patient_cost
            
            return total_cost

        except Exception as e:
            self.logger.error(f"Ошибка расчета затрат: {e}")
            return 1000.0

    def _generate_allocation_recommendations(self, doctor_id: int, room_id: int,
                                           equipment_id: Optional[int],
                                           request: ResourceAllocationRequest) -> List[str]:
        """Генерация рекомендаций по распределению ресурсов"""
        try:
            recommendations = []
            
            # Рекомендации по времени
            if request.duration_minutes < 30:
                recommendations.append("Рассмотрите увеличение времени приема для лучшего качества обслуживания")
            elif request.duration_minutes > 120:
                recommendations.append("Рассмотрите разбиение длительного приема на несколько сессий")
            
            # Рекомендации по сложности
            if request.complexity_level > 0.8:
                recommendations.append("Для сложных случаев рекомендуется привлечение дополнительного специалиста")
            
            # Рекомендации по оборудованию
            if not equipment_id and request.complexity_level > 0.6:
                recommendations.append("Рекомендуется использование диагностического оборудования")
            
            # Рекомендации по количеству пациентов
            if request.patient_count > 5:
                recommendations.append("Рассмотрите распределение пациентов на несколько временных слотов")
            
            if not recommendations:
                recommendations.append("Распределение ресурсов оптимально")
            
            return recommendations

        except Exception as e:
            self.logger.error(f"Ошибка генерации рекомендаций: {e}")
            return ["Ошибка при анализе распределения ресурсов"]

    def add_resource(self, resource: Resource):
        """Добавление нового ресурса"""
        try:
            self.resources[resource.resource_id] = resource
            self.logger.info(f"Добавлен ресурс: {resource.name} (ID: {resource.resource_id})")
        except Exception as e:
            self.logger.error(f"Ошибка добавления ресурса: {e}")

    def remove_resource(self, resource_id: int):
        """Удаление ресурса"""
        try:
            if resource_id in self.resources:
                resource_name = self.resources[resource_id].name
                del self.resources[resource_id]
                self.logger.info(f"Удален ресурс: {resource_name} (ID: {resource_id})")
        except Exception as e:
            self.logger.error(f"Ошибка удаления ресурса: {e}")

    def get_resource_utilization(self, resource_id: int, date: datetime) -> float:
        """Получение коэффициента использования ресурса"""
        try:
            if resource_id not in self.resources:
                return 0.0
            
            resource = self.resources[resource_id]
            return resource.current_utilization

        except Exception as e:
            self.logger.error(f"Ошибка получения использования ресурса: {e}")
            return 0.0

    def update_resource_availability(self, resource_id: int, date: datetime, 
                                   available_slots: List[datetime]):
        """Обновление доступности ресурса"""
        try:
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                resource.availability_schedule[date.date()] = available_slots
                self.logger.info(f"Обновлена доступность ресурса {resource_id} на {date.date()}")
        except Exception as e:
            self.logger.error(f"Ошибка обновления доступности ресурса: {e}") 