"""
Пример использования модуля оптимизации загрузки врачей
"""

from datetime import datetime, timedelta
import sys
import os

# Добавление пути к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.doctor_load_optimization_service import (
    DoctorLoadOptimizationService, OptimizationObjective
)
from repositories.medialog_repository import MedialogRepository
from database.database import get_session


def main():
    """Основная функция примера"""
    print("=== Пример использования модуля оптимизации загрузки врачей ===\n")
    
    try:
        # Инициализация сервиса
        print("1. Инициализация сервиса оптимизации...")
        session = next(get_session())
        repository = MedialogRepository(session)
        optimization_service = DoctorLoadOptimizationService(repository)
        print("✓ Сервис инициализирован\n")
        
        # Настройка параметров
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)
        date_range = (start_date, end_date)
        
        print(f"2. Параметры оптимизации:")
        print(f"   - Период: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")
        print(f"   - Длительность: {(end_date - start_date).days} дней\n")
        
        # Оптимизация с разными целями
        objectives = [
            OptimizationObjective.BALANCE_LOAD,
            OptimizationObjective.MINIMIZE_WAIT_TIME,
            OptimizationObjective.MAXIMIZE_UTILIZATION
        ]
        
        for i, objective in enumerate(objectives, 1):
            print(f"3.{i} Оптимизация с целью: {objective.value}")
            
            result = optimization_service.optimize_doctor_schedule(date_range, objective)
            
            print(f"   Результаты:")
            print(f"   - Коэффициент эффективности: {result.efficiency_score:.2f}")
            print(f"   - Сокращение времени ожидания: {result.wait_time_reduction:.1%}")
            print(f"   - Улучшение использования: {result.utilization_improvement:.1%}")
            print(f"   - Количество врачей в расписании: {len(result.optimized_schedule)}")
            
            if result.recommendations:
                print(f"   - Рекомендации:")
                for rec in result.recommendations[:3]:  # Показываем первые 3
                    print(f"     • {rec}")
            
            print()
        
        # Анализ метрик врачей
        print("4. Анализ метрик врачей...")
        
        # Получение всех врачей
        doctors = repository.get_all_medecins()
        if doctors:
            print(f"   Найдено врачей: {len(doctors)}")
            
            # Анализ по специальностям
            specialities = {}
            for doctor in doctors:
                if doctor.speciality not in specialities:
                    specialities[doctor.speciality] = 0
                specialities[doctor.speciality] += 1
            
            print(f"   Распределение по специальностям:")
            for speciality, count in specialities.items():
                print(f"     - {speciality}: {count} врачей")
            
            # Анализ производительности для первых 3 врачей
            print(f"\n   Анализ производительности (первые 3 врача):")
            for i, doctor in enumerate(doctors[:3]):
                performance = optimization_service.get_doctor_performance_report(
                    doctor.medecins_id, date_range
                )
                
                if performance:
                    print(f"     Врач {doctor.medecins_id} ({doctor.surname} {doctor.name}):")
                    print(f"       - Специальность: {doctor.speciality}")
                    print(f"       - Отделение: {doctor.department}")
                    print(f"       - Оценка производительности: {performance.get('performance_score', 0):.1%}")
                    print(f"       - Удовлетворенность пациентов: {performance.get('patient_satisfaction', 0):.1%}")
                    print()
        
        # Поиск оптимального врача
        print("5. Поиск оптимального врача для записи...")
        
        test_cases = [
            ("терапевт", 0.3, "простая консультация"),
            ("кардиолог", 0.7, "сложное обследование"),
            ("хирург", 0.9, "плановая операция")
        ]
        
        for speciality, complexity, description in test_cases:
            appointment_date = datetime(2024, 1, 15, 10, 0)
            
            optimal_doctor_id = optimization_service.get_optimal_doctor_for_appointment(
                speciality, complexity, appointment_date
            )
            
            print(f"   {description}:")
            print(f"     - Специальность: {speciality}")
            print(f"     - Сложность: {complexity}")
            print(f"     - Оптимальный врач: {optimal_doctor_id or 'не найден'}")
            print()
        
        # Демонстрация API endpoints
        print("6. Доступные API endpoints:")
        endpoints = [
            "POST /api/doctor-load-optimization/optimize-schedule",
            "GET /api/doctor-load-optimization/doctor-metrics",
            "GET /api/doctor-load-optimization/performance-report/{doctor_id}",
            "POST /api/doctor-load-optimization/optimal-doctor",
            "GET /api/doctor-load-optimization/load-balance",
            "GET /api/doctor-load-optimization/resource-allocation",
            "GET /api/doctor-load-optimization/appointment-patterns",
            "GET /api/doctor-load-optimization/health"
        ]
        
        for endpoint in endpoints:
            print(f"   - {endpoint}")
        
        print("\n=== Пример завершен успешно ===")
        
    except Exception as e:
        print(f"Ошибка при выполнении примера: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'session' in locals():
            session.close()


def demonstrate_optimization_algorithms():
    """Демонстрация алгоритмов оптимизации"""
    print("\n=== Демонстрация алгоритмов оптимизации ===\n")
    
    try:
        session = next(get_session())
        repository = MedialogRepository(session)
        optimization_service = DoctorLoadOptimizationService(repository)
        
        # Тестовые данные
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 3)
        date_range = (start_date, end_date)
        
        print("Сравнение алгоритмов оптимизации:")
        print("=" * 50)
        
        algorithms = {
            "Балансировка нагрузки": OptimizationObjective.BALANCE_LOAD,
            "Минимизация времени ожидания": OptimizationObjective.MINIMIZE_WAIT_TIME,
            "Максимизация использования": OptimizationObjective.MAXIMIZE_UTILIZATION
        }
        
        results = {}
        
        for name, objective in algorithms.items():
            print(f"\n{name}:")
            result = optimization_service.optimize_doctor_schedule(date_range, objective)
            results[name] = result
            
            print(f"  - Эффективность: {result.efficiency_score:.3f}")
            print(f"  - Сокращение ожидания: {result.wait_time_reduction:.1%}")
            print(f"  - Улучшение использования: {result.utilization_improvement:.1%}")
            
            if result.recommendations:
                print(f"  - Первая рекомендация: {result.recommendations[0]}")
        
        # Сравнение результатов
        print(f"\nСравнение результатов:")
        print("=" * 50)
        
        best_efficiency = max(results.items(), key=lambda x: x[1].efficiency_score)
        best_wait_reduction = max(results.items(), key=lambda x: x[1].wait_time_reduction)
        best_utilization = max(results.items(), key=lambda x: x[1].utilization_improvement)
        
        print(f"Лучшая эффективность: {best_efficiency[0]} ({best_efficiency[1].efficiency_score:.3f})")
        print(f"Лучшее сокращение ожидания: {best_wait_reduction[0]} ({best_wait_reduction[1].wait_time_reduction:.1%})")
        print(f"Лучшее улучшение использования: {best_utilization[0]} ({best_utilization[1].utilization_improvement:.1%})")
        
    except Exception as e:
        print(f"Ошибка демонстрации алгоритмов: {e}")
    
    finally:
        if 'session' in locals():
            session.close()


def demonstrate_load_balancing():
    """Демонстрация балансировки нагрузки"""
    print("\n=== Демонстрация балансировки нагрузки ===\n")
    
    try:
        session = next(get_session())
        repository = MedialogRepository(session)
        
        # Получение данных врачей
        doctors = repository.get_all_medecins()
        
        if not doctors:
            print("Врачи не найдены в базе данных")
            return
        
        print(f"Анализ нагрузки для {len(doctors)} врачей:")
        print("=" * 60)
        
        # Анализ по специальностям
        speciality_loads = {}
        
        for doctor in doctors:
            if doctor.speciality not in speciality_loads:
                speciality_loads[doctor.speciality] = {
                    'doctors': [],
                    'total_load': 0,
                    'avg_satisfaction': 0
                }
            
            speciality_loads[doctor.speciality]['doctors'].append(doctor)
        
        # Расчет средних показателей
        for speciality, data in speciality_loads.items():
            doctor_count = len(data['doctors'])
            print(f"\n{speciality} ({doctor_count} врачей):")
            
            # Здесь можно добавить расчет реальных метрик
            # Для демонстрации используем заглушки
            avg_load = 0.7 + (hash(speciality) % 30) / 100  # Псевдослучайная нагрузка
            avg_satisfaction = 0.8 + (hash(speciality) % 20) / 100
            
            print(f"  - Средняя загрузка: {avg_load:.1%}")
            print(f"  - Средняя удовлетворенность: {avg_satisfaction:.1%}")
            
            if avg_load > 0.85:
                print(f"  - ⚠️  Высокая загрузка - требуется оптимизация")
            elif avg_load < 0.5:
                print(f"  - ⚠️  Низкая загрузка - можно увеличить приемы")
            else:
                print(f"  - ✅ Оптимальная загрузка")
        
        print(f"\nОбщие рекомендации:")
        print("=" * 60)
        
        recommendations = [
            "Мониторить загрузку врачей еженедельно",
            "Перераспределять нагрузку при превышении 85%",
            "Учитывать сложность случаев при планировании",
            "Анализировать удовлетворенность пациентов"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
    except Exception as e:
        print(f"Ошибка демонстрации балансировки: {e}")
    
    finally:
        if 'session' in locals():
            session.close()


if __name__ == "__main__":
    # Основной пример
    main()
    
    # Дополнительные демонстрации
    demonstrate_optimization_algorithms()
    demonstrate_load_balancing()
    
    print("\n" + "="*60)
    print("Все примеры завершены. Модуль готов к использованию!")
    print("="*60) 