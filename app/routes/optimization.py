"""
API endpoints для оптимизации ресурсов
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from services.optimization_service import (
    get_optimizer_service, 
    get_load_balancer,
    ResourceOptimizer,
    LoadBalancer,
    Resource,
    ResourceType,
    OptimizationRequest,
    OptimizationResult
)

optimization_router = APIRouter()


class ResourceCreateRequest(BaseModel):
    name: str
    type: str
    capacity: int
    availability_hours: List[List[int]]
    cost_per_hour: float = 0.0
    specialization: Optional[str] = None


class OptimizationRequestModel(BaseModel):
    date: datetime
    patient_count: int
    appointment_types: Dict[str, int]
    constraints: Dict[str, any] = {}


class LoadBalanceRequest(BaseModel):
    current_loads: Dict[str, float]


@optimization_router.post("/resources/")
async def create_resource(
    request: ResourceCreateRequest,
    optimizer: ResourceOptimizer = Depends(get_optimizer_service)
):
    """Создание нового ресурса"""
    try:
        # Преобразование availability_hours
        availability_hours = [
            (hours[0], hours[1]) for hours in request.availability_hours
        ]
        
        resource = Resource(
            id=f"resource_{len(optimizer.resources) + 1}",
            name=request.name,
            type=ResourceType(request.type),
            capacity=request.capacity,
            availability_hours=availability_hours,
            cost_per_hour=request.cost_per_hour,
            specialization=request.specialization
        )
        
        optimizer.add_resource(resource)
        
        return {
            "status": "success",
            "message": f"Ресурс {request.name} создан успешно",
            "resource_id": resource.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания ресурса: {str(e)}")


@optimization_router.get("/resources/")
async def get_resources(
    optimizer: ResourceOptimizer = Depends(get_optimizer_service)
):
    """Получение списка всех ресурсов"""
    resources_data = []
    
    for resource_id, resource in optimizer.resources.items():
        resources_data.append({
            "id": resource.id,
            "name": resource.name,
            "type": resource.type.value,
            "capacity": resource.capacity,
            "availability_hours": resource.availability_hours,
            "cost_per_hour": resource.cost_per_hour,
            "specialization": resource.specialization
        })
    
    return {
        "resources": resources_data,
        "total_count": len(resources_data)
    }


@optimization_router.post("/optimize/")
async def optimize_schedule(
    request: OptimizationRequestModel,
    optimizer: ResourceOptimizer = Depends(get_optimizer_service)
) -> Dict:
    """Оптимизация расписания и распределения ресурсов"""
    try:
        optimization_request = OptimizationRequest(
            date=request.date,
            patient_count=request.patient_count,
            appointment_types=request.appointment_types,
            constraints=request.constraints
        )
        
        result = optimizer.optimize_schedule(optimization_request)
        
        return {
            "status": "success",
            "optimization_result": {
                "schedule": result.schedule,
                "resource_utilization": result.resource_utilization,
                "efficiency_score": result.efficiency_score,
                "cost_estimate": result.cost_estimate,
                "recommendations": result.recommendations
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка оптимизации: {str(e)}")


@optimization_router.get("/utilization/")
async def get_resource_utilization(
    date: Optional[str] = None,
    optimizer: ResourceOptimizer = Depends(get_optimizer_service)
):
    """Получение данных о загруженности ресурсов"""
    target_date = datetime.fromisoformat(date) if date else datetime.now()
    
    # Демонстрационные данные загруженности
    utilization_data = {}
    
    for resource_id, resource in optimizer.resources.items():
        # Симуляция текущей загруженности
        import random
        utilization_data[resource_id] = {
            "name": resource.name,
            "type": resource.type.value,
            "current_load": round(random.uniform(30, 95), 1),
            "capacity": resource.capacity,
            "cost_per_hour": resource.cost_per_hour,
            "availability_status": "available" if random.random() > 0.1 else "busy"
        }
    
    return {
        "date": target_date.isoformat(),
        "utilization": utilization_data,
        "summary": {
            "total_resources": len(utilization_data),
            "average_utilization": sum(
                data["current_load"] for data in utilization_data.values()
            ) / len(utilization_data) if utilization_data else 0,
            "overloaded_count": sum(
                1 for data in utilization_data.values() 
                if data["current_load"] > 80
            )
        }
    }


@optimization_router.post("/balance-load/")
async def balance_load(
    request: LoadBalanceRequest,
    balancer: LoadBalancer = Depends(get_load_balancer)
):
    """Балансировка нагрузки между ресурсами"""
    try:
        recommendations = balancer.balance_load(request.current_loads)
        
        return {
            "status": "success",
            "current_loads": request.current_loads,
            "balance_recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка балансировки: {str(e)}")


@optimization_router.get("/analytics/efficiency/")
async def get_efficiency_analytics(
    days: int = 7,
    optimizer: ResourceOptimizer = Depends(get_optimizer_service)
):
    """Аналитика эффективности использования ресурсов"""
    import random
    
    # Генерация демонстрационных данных
    dates = [
        (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") 
        for i in range(days-1, -1, -1)
    ]
    
    efficiency_data = {
        "dates": dates,
        "efficiency_scores": [round(random.uniform(75, 95), 1) for _ in range(days)],
        "cost_per_day": [round(random.uniform(15000, 25000), 2) for _ in range(days)],
        "resource_utilization": {
            "doctors": [round(random.uniform(70, 90), 1) for _ in range(days)],
            "rooms": [round(random.uniform(60, 85), 1) for _ in range(days)],
            "equipment": [round(random.uniform(45, 75), 1) for _ in range(days)],
            "nurses": [round(random.uniform(55, 80), 1) for _ in range(days)]
        }
    }
    
    return {
        "period": f"{days} дней",
        "analytics": efficiency_data,
        "summary": {
            "average_efficiency": sum(efficiency_data["efficiency_scores"]) / len(efficiency_data["efficiency_scores"]),
            "total_cost": sum(efficiency_data["cost_per_day"]),
            "cost_per_patient": sum(efficiency_data["cost_per_day"]) / (days * 50)  # Предполагаем 50 пациентов в день
        }
    }


@optimization_router.get("/recommendations/")
async def get_optimization_recommendations(
    optimizer: ResourceOptimizer = Depends(get_optimizer_service)
):
    """Получение общих рекомендаций по оптимизации"""
    
    # Анализ текущего состояния ресурсов
    total_resources = len(optimizer.resources)
    resource_types = {}
    
    for resource in optimizer.resources.values():
        resource_type = resource.type.value
        if resource_type not in resource_types:
            resource_types[resource_type] = 0
        resource_types[resource_type] += 1
    
    recommendations = []
    
    # Базовые рекомендации
    if total_resources < 10:
        recommendations.append("Рекомендуется расширить базу ресурсов для повышения эффективности")
    
    # Рекомендации по типам ресурсов
    if resource_types.get("doctor", 0) < 3:
        recommendations.append("Недостаточное количество врачей для оптимальной работы")
    
    if resource_types.get("room", 0) < 5:
        recommendations.append("Рекомендуется увеличить количество кабинетов")
    
    # Добавляем общие рекомендации
    recommendations.extend([
        "Регулярно анализируйте загруженность ресурсов",
        "Используйте прогнозирование для планирования ресурсов",
        "Оптимизируйте расписание на основе исторических данных"
    ])
    
    return {
        "current_state": {
            "total_resources": total_resources,
            "resource_breakdown": resource_types
        },
        "recommendations": recommendations,
        "priority_actions": recommendations[:3] if len(recommendations) >= 3 else recommendations
    } 