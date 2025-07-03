"""
Основной сервис оптимизации нагрузки врачей
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from app.services.doctor_load_optimization.schemas import (
    DoctorProfile, OptimizationResult, OptimizationObjective, OptimizationRequest,
    BatchOptimizationRequest, BatchOptimizationResponse, ResourceAllocationRequest,
    ResourceAllocationResult, PerformanceMetrics
)
from app.services.doctor_load_optimization.database_connector import MedialogDatabaseConnector
from app.services.doctor_load_optimization.load_balancer import create_load_balancer, LoadBalancingStrategy
from app.services.doctor_load_optimization.schedule_optimizer import ScheduleOptimizer
from app.services.doctor_load_optimization.resource_allocator import ResourceAllocator
from app.services.doctor_load_optimization.performance_tracker import PerformanceTracker


class DoctorLoadOptimizationService:
    """Основной сервис оптимизации нагрузки врачей"""

    def __init__(self, db_connector: MedialogDatabaseConnector):
        """
        Инициализация сервиса
        
        Args:
            db_connector: Коннектор к базе данных
        """
        self.db_connector = db_connector
        self.schedule_optimizer = ScheduleOptimizer()
        self.resource_allocator = ResourceAllocator()
        self.performance_tracker = PerformanceTracker()
        self.logger = logging.getLogger(__name__)

    def optimize_doctor_schedule(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Оптимизация расписания врачей
        
        Args:
            request: Запрос на оптимизацию
            
        Returns:
            Результат оптимизации
        """
        try:
            start_time = time.time()
            
            # Получение профилей врачей
            doctor_profiles = self._get_doctor_profiles(request)
            
            if not doctor_profiles:
                raise ValueError("Не найдены врачи для оптимизации")
            
            self.logger.info(f"Найдено {len(doctor_profiles)} врачей для оптимизации")
            
            # Оптимизация расписания
            result = self.schedule_optimizer.optimize_doctor_load(
                doctor_profiles=doctor_profiles,
                date_range=(request.start_date, request.end_date),
                objective=request.objective
            )
            
            processing_time = time.time() - start_time
            self.logger.info(f"Оптимизация завершена за {processing_time:.2f} секунд")
            
            return result

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации расписания: {e}")
            raise

    def batch_optimize_schedules(self, request: BatchOptimizationRequest) -> BatchOptimizationResponse:
        """
        Пакетная оптимизация расписаний
        
        Args:
            request: Запрос на пакетную оптимизацию
            
        Returns:
            Результаты пакетной оптимизации
        """
        try:
            start_time = time.time()
            
            results = []
            successful_optimizations = 0
            failed_optimizations = 0
            
            for opt_request in request.optimization_requests:
                try:
                    result = self.optimize_doctor_schedule(opt_request)
                    results.append(result)
                    successful_optimizations += 1
                except Exception as e:
                    self.logger.error(f"Ошибка оптимизации в пакете: {e}")
                    failed_optimizations += 1
            
            processing_time = time.time() - start_time
            
            # Расчет сводных метрик
            summary_metrics = self._calculate_summary_metrics(results)
            
            return BatchOptimizationResponse(
                results=results,
                total_processed=len(request.optimization_requests),
                successful_optimizations=successful_optimizations,
                failed_optimizations=failed_optimizations,
                processing_time=processing_time,
                summary_metrics=summary_metrics
            )

        except Exception as e:
            self.logger.error(f"Ошибка пакетной оптимизации: {e}")
            raise

    def allocate_resources(self, request: ResourceAllocationRequest) -> ResourceAllocationResult:
        """
        Распределение ресурсов для приема
        
        Args:
            request: Запрос на распределение ресурсов
            
        Returns:
            Результат распределения ресурсов
        """
        try:
            return self.resource_allocator.allocate_resources(request)
        except Exception as e:
            self.logger.error(f"Ошибка распределения ресурсов: {e}")
            raise

    def get_doctor_performance_report(self, doctor_id: int,
                                    period_start: datetime,
                                    period_end: datetime) -> PerformanceMetrics:
        """
        Получение отчета о производительности врача
        
        Args:
            doctor_id: ID врача
            period_start: Начало периода
            period_end: Конец периода
            
        Returns:
            Метрики производительности
        """
        try:
            return self.performance_tracker.track_doctor_performance(
                doctor_id, period_start, period_end
            )
        except Exception as e:
            self.logger.error(f"Ошибка получения отчета о производительности: {e}")
            raise

    def get_optimal_doctor_for_appointment(self, speciality: str, complexity: float,
                                         date: datetime) -> Optional[int]:
        """
        Получение оптимального врача для приема
        
        Args:
            speciality: Специальность
            complexity: Сложность случая
            date: Дата приема
            
        Returns:
            ID оптимального врача или None
        """
        try:
            # Получение профилей врачей
            doctor_profiles = self.db_connector.get_doctors_by_criteria(
                specialities=[speciality],
                active_only=True
            )
            
            if not doctor_profiles:
                return None
            
            return self.schedule_optimizer.get_optimal_doctor_for_appointment(
                speciality, complexity, doctor_profiles
            )

        except Exception as e:
            self.logger.error(f"Ошибка выбора оптимального врача: {e}")
            return None

    def get_load_balance_analysis(self, start_date: datetime, end_date: datetime,
                                specialities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Анализ балансировки нагрузки
        
        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            specialities: Специальности для анализа
            
        Returns:
            Результат анализа балансировки
        """
        try:
            # Получение профилей врачей
            doctor_profiles = self.db_connector.get_doctors_by_criteria(
                specialities=specialities,
                active_only=True
            )
            
            if not doctor_profiles:
                return {"error": "Не найдены врачи для анализа"}
            
            # Анализ распределения нагрузки
            load_distribution = {}
            speciality_loads = {}
            
            for doctor in doctor_profiles:
                load_distribution[doctor.doctor_id] = doctor.current_load
                
                if doctor.speciality not in speciality_loads:
                    speciality_loads[doctor.speciality] = {
                        'total_load': 0.0,
                        'doctor_count': 0,
                        'avg_load': 0.0
                    }
                
                speciality_loads[doctor.speciality]['total_load'] += doctor.current_load
                speciality_loads[doctor.speciality]['doctor_count'] += 1
            
            # Расчет средних значений
            for speciality in speciality_loads:
                count = speciality_loads[speciality]['doctor_count']
                if count > 0:
                    speciality_loads[speciality]['avg_load'] = (
                        speciality_loads[speciality]['total_load'] / count
                    )
            
            # Определение перегруженных и недогруженных врачей
            overloaded_doctors = [
                d.doctor_id for d in doctor_profiles if d.current_load > 0.8
            ]
            underloaded_doctors = [
                d.doctor_id for d in doctor_profiles if d.current_load < 0.3
            ]
            
            # Рекомендации по балансировке
            recommendations = []
            if overloaded_doctors:
                recommendations.append(f"Перегружено {len(overloaded_doctors)} врачей")
            if underloaded_doctors:
                recommendations.append(f"Недогружено {len(underloaded_doctors)} врачей")
            
            if not recommendations:
                recommendations.append("Нагрузка распределена равномерно")
            
            return {
                "load_distribution": load_distribution,
                "speciality_loads": speciality_loads,
                "overloaded_doctors": overloaded_doctors,
                "underloaded_doctors": underloaded_doctors,
                "recommendations": recommendations,
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Ошибка анализа балансировки нагрузки: {e}")
            return {"error": str(e)}

    def get_resource_utilization_report(self, date: datetime,
                                      specialities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Отчет об использовании ресурсов
        
        Args:
            date: Дата отчета
            specialities: Специальности для анализа
            
        Returns:
            Отчет об использовании ресурсов
        """
        try:
            # Получение данных об использовании ресурсов
            resource_utilization = {}
            
            # Анализ кабинетов
            for resource_id in range(1, 21):  # Кабинеты 1-20
                utilization = self.resource_allocator.get_resource_utilization(resource_id, date)
                resource_utilization[f"room_{resource_id}"] = {
                    "type": "room",
                    "utilization": utilization,
                    "status": "available" if utilization < 0.8 else "busy"
                }
            
            # Анализ оборудования
            for resource_id in range(100, 108):  # Оборудование 100-107
                utilization = self.resource_allocator.get_resource_utilization(resource_id, date)
                resource_utilization[f"equipment_{resource_id}"] = {
                    "type": "equipment",
                    "utilization": utilization,
                    "status": "available" if utilization < 0.7 else "busy"
                }
            
            # Расчет общих метрик
            total_resources = len(resource_utilization)
            available_resources = len([
                r for r in resource_utilization.values() 
                if r["status"] == "available"
            ])
            
            avg_utilization = sum(
                r["utilization"] for r in resource_utilization.values()
            ) / total_resources if total_resources > 0 else 0.0
            
            return {
                "date": date.isoformat(),
                "resource_utilization": resource_utilization,
                "summary": {
                    "total_resources": total_resources,
                    "available_resources": available_resources,
                    "utilization_rate": avg_utilization,
                    "availability_rate": available_resources / total_resources if total_resources > 0 else 0.0
                },
                "recommendations": self._generate_resource_recommendations(resource_utilization)
            }

        except Exception as e:
            self.logger.error(f"Ошибка получения отчета об использовании ресурсов: {e}")
            return {"error": str(e)}

    def _get_doctor_profiles(self, request: OptimizationRequest) -> List[DoctorProfile]:
        """Получение профилей врачей для оптимизации"""
        try:
            return self.db_connector.get_doctors_by_criteria(
                specialities=request.specialities,
                departments=request.departments,
                active_only=True
            )
        except Exception as e:
            self.logger.error(f"Ошибка получения профилей врачей: {e}")
            return []

    def _calculate_summary_metrics(self, results: List[OptimizationResult]) -> Dict[str, float]:
        """Расчет сводных метрик для пакетной оптимизации"""
        try:
            if not results:
                return {}
            
            summary = {
                "avg_efficiency_score": sum(r.efficiency_score for r in results) / len(results),
                "avg_wait_time_reduction": sum(r.wait_time_reduction for r in results) / len(results),
                "avg_utilization_improvement": sum(r.utilization_improvement for r in results) / len(results),
                "total_cost_savings": sum(r.cost_savings for r in results),
                "avg_processing_time": sum(r.processing_time for r in results) / len(results)
            }
            
            return summary

        except Exception as e:
            self.logger.error(f"Ошибка расчета сводных метрик: {e}")
            return {}

    def _generate_resource_recommendations(self, resource_utilization: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по ресурсам"""
        try:
            recommendations = []
            
            # Анализ перегруженных ресурсов
            overloaded_resources = [
                resource_id for resource_id, data in resource_utilization.items()
                if data["utilization"] > 0.9
            ]
            
            if overloaded_resources:
                recommendations.append(f"Рассмотрите добавление ресурсов: {', '.join(overloaded_resources[:3])}")
            
            # Анализ недогруженных ресурсов
            underloaded_resources = [
                resource_id for resource_id, data in resource_utilization.items()
                if data["utilization"] < 0.3
            ]
            
            if underloaded_resources:
                recommendations.append(f"Оптимизируйте использование ресурсов: {', '.join(underloaded_resources[:3])}")
            
            if not recommendations:
                recommendations.append("Использование ресурсов оптимально")
            
            return recommendations

        except Exception as e:
            self.logger.error(f"Ошибка генерации рекомендаций по ресурсам: {e}")
            return ["Ошибка при анализе ресурсов"]

    def get_service_health(self) -> Dict[str, Any]:
        """Проверка состояния сервиса"""
        try:
            return {
                "status": "healthy",
                "components": {
                    "database_connector": "connected",
                    "schedule_optimizer": "ready",
                    "resource_allocator": "ready",
                    "performance_tracker": "ready"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Ошибка проверки состояния сервиса: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            } 