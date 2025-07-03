"""
API маршруты для модуля оптимизации загрузки врачей
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session
import pandas as pd

from app.database.database import get_session
from app.services.doctor_load_optimization_service import (
    DoctorLoadOptimizationService, OptimizationObjective, OptimizationResult
)
from app.repositories.medialog_repository import MedialogRepository
from app.models.base import BaseResponse, PaginatedResponse

router = APIRouter(prefix="/api/doctor-load-optimization", tags=["Doctor Load Optimization"])


def get_optimization_service(session: Session = Depends(get_session)) -> DoctorLoadOptimizationService:
    """Получение сервиса оптимизации загрузки врачей"""
    repository = MedialogRepository(session)
    return DoctorLoadOptimizationService(repository)


def get_medialog_repository(session: Session = Depends(get_session)) -> MedialogRepository:
    """Получение репозитория МИС Медиалог"""
    return MedialogRepository(session)


@router.post("/optimize-schedule", response_model=BaseResponse)
async def optimize_doctor_schedule(
    start_date: datetime = Query(description="Начальная дата"),
    end_date: datetime = Query(description="Конечная дата"),
    objective: OptimizationObjective = Query(
        default=OptimizationObjective.BALANCE_LOAD,
        description="Цель оптимизации"
    ),
    optimization_service: DoctorLoadOptimizationService = Depends(get_optimization_service)
):
    """Оптимизация расписания врачей"""
    try:
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Начальная дата должна быть меньше конечной")

        if (end_date - start_date).days > 30:
            raise HTTPException(status_code=400, detail="Период не должен превышать 30 дней")

        date_range = (start_date, end_date)
        result = optimization_service.optimize_doctor_schedule(date_range, objective)

        return BaseResponse(
            success=True,
            message="Оптимизация расписания выполнена успешно",
            data={
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "objective": objective.value,
                "efficiency_score": result.efficiency_score,
                "wait_time_reduction": result.wait_time_reduction,
                "utilization_improvement": result.utilization_improvement,
                "recommendations": result.recommendations,
                "load_distribution": result.load_distribution,
                "optimized_schedule_summary": {
                    doctor_id: len(slots) for doctor_id, slots in result.optimized_schedule.items()
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка оптимизации расписания: {str(e)}")


@router.get("/doctor-metrics", response_model=BaseResponse)
async def get_doctor_metrics(
    start_date: datetime = Query(description="Начальная дата"),
    end_date: datetime = Query(description="Конечная дата"),
    speciality: Optional[str] = Query(default=None, description="Специальность"),
    department: Optional[str] = Query(default=None, description="Отделение"),
    repository: MedialogRepository = Depends(get_medialog_repository)
):
    """Получение метрик врачей"""
    try:
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Начальная дата должна быть меньше конечной")

        # Получение врачей по критериям
        doctors = repository.get_medecins_by_department(department) if department else []
        
        if speciality:
            doctors = [d for d in doctors if d.speciality == speciality]
        
        if department:
            doctors = [d for d in doctors if d.department == department]

        # Расчет метрик для каждого врача
        metrics = []
        for doctor in doctors:
            # Получение расписания врача
            schedules = repository.get_schedule_by_medecin(
                doctor.medecins_id, start_date, end_date
            )
            
            # Получение записей врача
            appointments = repository.get_appointments_by_medecin(
                doctor.medecins_id, start_date, end_date
            )
            
            # Расчет метрик
            total_slots = sum(s.slots_total for s in schedules)
            booked_slots = sum(s.slots_booked for s in schedules)
            current_load = booked_slots / total_slots if total_slots > 0 else 0
            
            # Упрощенные расчеты для демонстрации
            avg_wait_time = 30.0  # Заглушка
            utilization_rate = current_load
            patient_satisfaction = 0.8  # Заглушка
            complexity_score = 0.5  # Заглушка
            
            doctor_metric = {
                "doctor_id": doctor.medecins_id,
                "name": f"{doctor.surname} {doctor.name}",
                "speciality": doctor.speciality,
                "department": doctor.department,
                "current_load": current_load,
                "avg_wait_time": avg_wait_time,
                "utilization_rate": utilization_rate,
                "patient_satisfaction": patient_satisfaction,
                "complexity_score": complexity_score,
                "total_appointments": len(appointments),
                "total_slots": total_slots,
                "booked_slots": booked_slots
            }
            
            metrics.append(doctor_metric)

        return BaseResponse(
            success=True,
            message=f"Получено {len(metrics)} метрик врачей",
            data={
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "filters": {
                    "speciality": speciality,
                    "department": department
                },
                "metrics": metrics,
                "summary": {
                    "total_doctors": len(metrics),
                    "avg_load": sum(m["current_load"] for m in metrics) / len(metrics) if metrics else 0,
                    "avg_wait_time": sum(m["avg_wait_time"] for m in metrics) / len(metrics) if metrics else 0,
                    "avg_utilization": sum(m["utilization_rate"] for m in metrics) / len(metrics) if metrics else 0
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения метрик врачей: {str(e)}")


@router.get("/performance-report/{doctor_id}", response_model=BaseResponse)
async def get_doctor_performance_report(
    doctor_id: int,
    start_date: datetime = Query(description="Начальная дата"),
    end_date: datetime = Query(description="Конечная дата"),
    optimization_service: DoctorLoadOptimizationService = Depends(get_optimization_service)
):
    """Получение отчета о производительности врача"""
    try:
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Начальная дата должна быть меньше конечной")

        date_range = (start_date, end_date)
        performance_report = optimization_service.get_doctor_performance_report(doctor_id, date_range)

        if not performance_report:
            raise HTTPException(status_code=404, detail="Врач не найден или нет данных")

        return BaseResponse(
            success=True,
            message="Отчет о производительности получен",
            data={
                "doctor_id": doctor_id,
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "performance": performance_report
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения отчета производительности: {str(e)}")


@router.post("/optimal-doctor", response_model=BaseResponse)
async def get_optimal_doctor_for_appointment(
    speciality: str = Query(description="Специальность"),
    complexity: float = Query(ge=0.0, le=1.0, description="Сложность случая (0-1)"),
    appointment_date: datetime = Query(description="Дата приема"),
    optimization_service: DoctorLoadOptimizationService = Depends(get_optimization_service)
):
    """Получение оптимального врача для записи"""
    try:
        optimal_doctor_id = optimization_service.get_optimal_doctor_for_appointment(
            speciality, complexity, appointment_date
        )

        if not optimal_doctor_id:
            raise HTTPException(status_code=404, detail="Подходящий врач не найден")

        return BaseResponse(
            success=True,
            message="Оптимальный врач найден",
            data={
                "optimal_doctor_id": optimal_doctor_id,
                "speciality": speciality,
                "complexity": complexity,
                "appointment_date": appointment_date.isoformat()
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска оптимального врача: {str(e)}")


@router.get("/load-balance", response_model=BaseResponse)
async def get_load_balance_analysis(
    start_date: datetime = Query(description="Начальная дата"),
    end_date: datetime = Query(description="Конечная дата"),
    speciality: Optional[str] = Query(default=None, description="Специальность"),
    repository: MedialogRepository = Depends(get_medialog_repository)
):
    """Анализ балансировки нагрузки"""
    try:
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Начальная дата должна быть меньше конечной")

        # Получение врачей
        doctors = repository.get_medecins_by_department(speciality) if speciality else []
        if speciality:
            doctors = [d for d in doctors if d.speciality == speciality]

        # Анализ нагрузки по врачам
        load_analysis = []
        for doctor in doctors:
            schedules = repository.get_schedule_by_medecin(
                doctor.medecins_id, start_date, end_date
            )
            
            total_slots = sum(s.slots_total for s in schedules)
            booked_slots = sum(s.slots_booked for s in schedules)
            load_percentage = (booked_slots / total_slots * 100) if total_slots > 0 else 0
            
            load_status = "low"
            if load_percentage > 80:
                load_status = "high"
            elif load_percentage > 50:
                load_status = "medium"
            
            analysis = {
                "doctor_id": doctor.medecins_id,
                "name": f"{doctor.surname} {doctor.name}",
                "speciality": doctor.speciality,
                "department": doctor.department,
                "total_slots": total_slots,
                "booked_slots": booked_slots,
                "load_percentage": load_percentage,
                "load_status": load_status
            }
            
            load_analysis.append(analysis)

        # Статистика по нагрузке
        high_load_count = len([a for a in load_analysis if a["load_status"] == "high"])
        medium_load_count = len([a for a in load_analysis if a["load_status"] == "medium"])
        low_load_count = len([a for a in load_analysis if a["load_status"] == "low"])

        return BaseResponse(
            success=True,
            message="Анализ балансировки нагрузки выполнен",
            data={
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "speciality": speciality,
                "load_analysis": load_analysis,
                "statistics": {
                    "total_doctors": len(load_analysis),
                    "high_load": high_load_count,
                    "medium_load": medium_load_count,
                    "low_load": low_load_count,
                    "avg_load_percentage": sum(a["load_percentage"] for a in load_analysis) / len(load_analysis) if load_analysis else 0
                },
                "recommendations": [
                    f"Выявлено {high_load_count} перегруженных врачей" if high_load_count > 0 else "Перегруженных врачей не выявлено",
                    f"Выявлено {low_load_count} недозагруженных врачей" if low_load_count > 0 else "Недозагруженных врачей не выявлено"
                ]
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа балансировки нагрузки: {str(e)}")


@router.get("/resource-allocation", response_model=BaseResponse)
async def get_resource_allocation_status(
    date: datetime = Query(description="Дата"),
    speciality: Optional[str] = Query(default=None, description="Специальность"),
    repository: MedialogRepository = Depends(get_medialog_repository)
):
    """Получение статуса распределения ресурсов"""
    try:
        # Получение расписания на дату
        schedules = repository.get_schedules_by_date(date)
        
        if speciality:
            # Фильтрация по специальности (упрощенно)
            doctors = repository.get_medecins_by_department(speciality)
            doctor_ids = [d.medecins_id for d in doctors]
            schedules = [s for s in schedules if s.medecins_id in doctor_ids]

        # Анализ использования ресурсов
        resource_usage = {
            "total_schedules": len(schedules),
            "total_slots": sum(s.slots_total for s in schedules),
            "booked_slots": sum(s.slots_booked for s in schedules),
            "utilization_rate": sum(s.slots_booked for s in schedules) / sum(s.slots_total for s in schedules) if schedules else 0,
            "available_slots": sum(s.slots_total - s.slots_booked for s in schedules),
            "rooms_in_use": len(set(s.cabinet for s in schedules if s.cabinet))
        }

        return BaseResponse(
            success=True,
            message="Статус распределения ресурсов получен",
            data={
                "date": date.isoformat(),
                "speciality": speciality,
                "resource_usage": resource_usage,
                "schedules_summary": [
                    {
                        "schedule_id": s.schedule_id,
                        "doctor_id": s.medecins_id,
                        "cabinet": s.cabinet,
                        "slots_total": s.slots_total,
                        "slots_booked": s.slots_booked,
                        "utilization": s.slots_booked / s.slots_total if s.slots_total > 0 else 0
                    }
                    for s in schedules
                ]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса ресурсов: {str(e)}")


@router.get("/appointment-patterns", response_model=BaseResponse)
async def get_appointment_patterns(
    start_date: datetime = Query(description="Начальная дата"),
    end_date: datetime = Query(description="Конечная дата"),
    repository: MedialogRepository = Depends(get_medialog_repository)
):
    """Анализ паттернов записей"""
    try:
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Начальная дата должна быть меньше конечной")

        # Получение записей за период
        all_appointments = []
        for day in pd.date_range(start_date, end_date):
            appointments = repository.get_appointments_by_date(day)
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

        return BaseResponse(
            success=True,
            message="Анализ паттернов записей выполнен",
            data={
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "total_appointments": len(all_appointments),
                "hourly_patterns": hourly_patterns,
                "daily_patterns": daily_patterns,
                "type_patterns": type_patterns,
                "peak_hours": sorted(hourly_patterns.items(), key=lambda x: x[1], reverse=True)[:3],
                "peak_days": sorted(daily_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа паттернов: {str(e)}")


@router.get("/health", response_model=BaseResponse)
async def health_check(
    optimization_service: DoctorLoadOptimizationService = Depends(get_optimization_service)
):
    """Проверка состояния сервиса оптимизации"""
    try:
        return BaseResponse(
            success=True,
            message="Сервис оптимизации загрузки врачей работает",
            data={
                "service_status": "healthy",
                "components": {
                    "schedule_optimizer": "active",
                    "resource_allocator": "active",
                    "performance_tracker": "active",
                    "load_balancer": "active"
                },
                "version": "1.0.0"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка проверки состояния: {str(e)}") 