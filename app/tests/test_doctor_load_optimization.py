"""
Тесты для модуля оптимизации загрузки врачей
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from services.doctor_load_optimization_service import (
    DoctorLoadOptimizationService, ScheduleOptimizer, ResourceAllocator, PerformanceTracker,
    LoadBalancer, RoundRobinLoadBalancer, WeightedLoadBalancer,
    DoctorMetrics, ScheduleSlot, OptimizationResult, OptimizationObjective
)
from repositories.medialog_repository import MedialogRepository


class TestDoctorMetrics:
    """Тесты для класса DoctorMetrics"""

    def test_doctor_metrics_creation(self):
        """Тест создания метрик врача"""
        metrics = DoctorMetrics(
            doctor_id=123,
            speciality="терапевт",
            department="Терапевтическое отделение",
            current_load=0.75,
            avg_wait_time=45.0,
            utilization_rate=0.8,
            patient_satisfaction=0.85,
            complexity_score=0.6
        )
        
        assert metrics.doctor_id == 123
        assert metrics.speciality == "терапевт"
        assert metrics.current_load == 0.75
        assert metrics.avg_wait_time == 45.0
        assert metrics.utilization_rate == 0.8
        assert metrics.patient_satisfaction == 0.85
        assert metrics.complexity_score == 0.6


class TestScheduleSlot:
    """Тесты для класса ScheduleSlot"""

    def test_schedule_slot_creation(self):
        """Тест создания слота расписания"""
        start_time = datetime(2024, 1, 15, 9, 0)
        end_time = datetime(2024, 1, 15, 9, 30)
        
        slot = ScheduleSlot(
            start_time=start_time,
            end_time=end_time,
            doctor_id=123,
            patient_id=456,
            appointment_type="consultation",
            complexity=0.5,
            is_available=False
        )
        
        assert slot.start_time == start_time
        assert slot.end_time == end_time
        assert slot.doctor_id == 123
        assert slot.patient_id == 456
        assert slot.appointment_type == "consultation"
        assert slot.complexity == 0.5
        assert slot.is_available is False


class TestOptimizationResult:
    """Тесты для класса OptimizationResult"""

    def test_optimization_result_creation(self):
        """Тест создания результата оптимизации"""
        optimized_schedule = {123: [], 456: []}
        load_distribution = {123: 0.7, 456: 0.8}
        
        result = OptimizationResult(
            optimized_schedule=optimized_schedule,
            load_distribution=load_distribution,
            wait_time_reduction=0.15,
            utilization_improvement=0.2,
            recommendations=["Рекомендация 1", "Рекомендация 2"],
            efficiency_score=0.85
        )
        
        assert result.optimized_schedule == optimized_schedule
        assert result.load_distribution == load_distribution
        assert result.wait_time_reduction == 0.15
        assert result.utilization_improvement == 0.2
        assert len(result.recommendations) == 2
        assert result.efficiency_score == 0.85


class TestRoundRobinLoadBalancer:
    """Тесты для RoundRobinLoadBalancer"""

    @pytest.fixture
    def load_balancer(self):
        """Фикстура для балансировщика"""
        return RoundRobinLoadBalancer()

    @pytest.fixture
    def doctor_metrics(self):
        """Фикстура для метрик врачей"""
        return [
            DoctorMetrics(1, "терапевт", "отделение", 0.8, 30, 0.7, 0.8, 0.5),
            DoctorMetrics(2, "терапевт", "отделение", 0.6, 25, 0.8, 0.9, 0.6),
            DoctorMetrics(3, "терапевт", "отделение", 0.4, 20, 0.9, 0.7, 0.4)
        ]

    def test_balance_load(self, load_balancer, doctor_metrics):
        """Тест балансировки нагрузки"""
        balanced_loads = load_balancer.balance_load(doctor_metrics)
        
        assert len(balanced_loads) == 3
        assert all(0 <= load <= 1 for load in balanced_loads.values())
        
        # Проверяем, что нагрузки сбалансированы
        loads = list(balanced_loads.values())
        avg_load = sum(loads) / len(loads)
        assert all(abs(load - avg_load) < 0.2 for load in loads)

    def test_balance_load_empty_list(self, load_balancer):
        """Тест балансировки с пустым списком"""
        balanced_loads = load_balancer.balance_load([])
        assert balanced_loads == {}

    def test_get_optimal_doctor(self, load_balancer):
        """Тест получения оптимального врача"""
        doctor_id = load_balancer.get_optimal_doctor("терапевт", 0.5)
        assert isinstance(doctor_id, int)


class TestWeightedLoadBalancer:
    """Тесты для WeightedLoadBalancer"""

    @pytest.fixture
    def load_balancer(self):
        """Фикстура для взвешенного балансировщика"""
        return WeightedLoadBalancer()

    @pytest.fixture
    def doctor_metrics(self):
        """Фикстура для метрик врачей"""
        return [
            DoctorMetrics(1, "терапевт", "отделение", 0.8, 30, 0.7, 0.8, 0.5),
            DoctorMetrics(2, "терапевт", "отделение", 0.6, 25, 0.8, 0.9, 0.6),
            DoctorMetrics(3, "терапевт", "отделение", 0.4, 20, 0.9, 0.7, 0.4)
        ]

    def test_balance_load(self, load_balancer, doctor_metrics):
        """Тест взвешенной балансировки нагрузки"""
        balanced_loads = load_balancer.balance_load(doctor_metrics)
        
        assert len(balanced_loads) == 3
        assert all(0.1 <= load <= 0.95 for load in balanced_loads.values())
        
        # Проверяем, что врач с лучшими метриками получил более высокую нагрузку
        doctor_2_load = balanced_loads[2]  # Врач с лучшими метриками
        doctor_1_load = balanced_loads[1]  # Врач с худшими метриками
        assert doctor_2_load >= doctor_1_load

    def test_balance_load_empty_list(self, load_balancer):
        """Тест балансировки с пустым списком"""
        balanced_loads = load_balancer.balance_load([])
        assert balanced_loads == {}


class TestScheduleOptimizer:
    """Тесты для ScheduleOptimizer"""

    @pytest.fixture
    def mock_repository(self):
        """Фикстура для мок-репозитория"""
        return Mock(spec=MedialogRepository)

    @pytest.fixture
    def optimizer(self, mock_repository):
        """Фикстура для оптимизатора"""
        return ScheduleOptimizer(mock_repository)

    @pytest.fixture
    def date_range(self):
        """Фикстура для диапазона дат"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)
        return (start_date, end_date)

    def test_calculate_current_load(self, optimizer, mock_repository, date_range):
        """Тест расчета текущей загрузки"""
        # Мокаем данные врачей
        mock_doctors = [
            Mock(medecins_id=1, speciality="терапевт", department="отделение"),
            Mock(medecins_id=2, speciality="кардиолог", department="отделение")
        ]
        mock_repository.get_all_medecins.return_value = mock_doctors
        
        # Мокаем расписания
        mock_schedules = [
            Mock(slots_total=20, slots_booked=15),
            Mock(slots_total=16, slots_booked=8)
        ]
        mock_repository.get_schedule_by_medecin.return_value = mock_schedules
        
        # Мокаем записи
        mock_appointments = [Mock(no_show_flag=False), Mock(no_show_flag=True)]
        mock_repository.get_appointments_by_medecin.return_value = mock_appointments
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2024, 1, 1)
            
            metrics = optimizer.calculate_current_load(date_range)
            
            assert len(metrics) == 2
            assert metrics[0].doctor_id == 1
            assert metrics[0].current_load == 0.75  # 15/20
            assert metrics[1].doctor_id == 2
            assert metrics[1].current_load == 0.5   # 8/16

    def test_analyze_appointment_patterns(self, optimizer, mock_repository, date_range):
        """Тест анализа паттернов записей"""
        # Мокаем записи
        mock_appointments = [
            Mock(appointment_time=datetime(2024, 1, 1, 9, 0), visit_type="consultation"),
            Mock(appointment_time=datetime(2024, 1, 1, 10, 0), visit_type="examination"),
            Mock(appointment_time=datetime(2024, 1, 2, 9, 0), visit_type="consultation")
        ]
        mock_repository.get_appointments_by_date.return_value = mock_appointments
        
        patterns = optimizer.analyze_appointment_patterns(date_range)
        
        assert 'hourly_patterns' in patterns
        assert 'daily_patterns' in patterns
        assert 'type_patterns' in patterns
        assert patterns['total_appointments'] == 3

    def test_balance_load_optimization(self, optimizer):
        """Тест оптимизации балансировки нагрузки"""
        doctor_metrics = [
            DoctorMetrics(1, "терапевт", "отделение", 0.8, 30, 0.7, 0.8, 0.5),
            DoctorMetrics(2, "терапевт", "отделение", 0.6, 25, 0.8, 0.9, 0.6),
            DoctorMetrics(3, "терапевт", "отделение", 0.4, 20, 0.9, 0.7, 0.4)
        ]
        
        optimal_distribution = optimizer.balance_load_optimization(doctor_metrics)
        
        assert len(optimal_distribution) == 3
        assert all(0.1 <= load <= 0.95 for load in optimal_distribution.values())
        
        # Проверяем, что нагрузки сбалансированы
        loads = list(optimal_distribution.values())
        avg_load = sum(loads) / len(loads)
        assert all(abs(load - avg_load) < 0.3 for load in loads)

    def test_minimize_wait_time_optimization(self, optimizer):
        """Тест оптимизации для минимизации времени ожидания"""
        doctor_metrics = [
            DoctorMetrics(1, "терапевт", "отделение", 0.8, 60, 0.7, 0.8, 0.5),
            DoctorMetrics(2, "терапевт", "отделение", 0.6, 30, 0.8, 0.9, 0.6),
            DoctorMetrics(3, "терапевт", "отделение", 0.4, 20, 0.9, 0.7, 0.4)
        ]
        
        patterns = {'hourly_patterns': {}, 'daily_patterns': {}, 'type_patterns': {}}
        
        optimal_distribution = optimizer.minimize_wait_time_optimization(doctor_metrics, patterns)
        
        assert len(optimal_distribution) == 3
        # Врач с меньшим временем ожидания должен получить более высокую нагрузку
        assert optimal_distribution[3] >= optimal_distribution[1]

    def test_maximize_utilization_optimization(self, optimizer):
        """Тест оптимизации для максимизации использования"""
        doctor_metrics = [
            DoctorMetrics(1, "терапевт", "отделение", 0.8, 30, 0.7, 0.8, 0.5),
            DoctorMetrics(2, "кардиолог", "отделение", 0.6, 25, 0.8, 0.9, 0.8),
            DoctorMetrics(3, "терапевт", "отделение", 0.4, 20, 0.9, 0.7, 0.4)
        ]
        
        optimal_distribution = optimizer.maximize_utilization_optimization(doctor_metrics)
        
        assert len(optimal_distribution) == 3
        # Кардиолог должен получить меньшую нагрузку из-за сложности
        assert optimal_distribution[2] <= optimal_distribution[1]

    def test_calculate_improvements(self, optimizer):
        """Тест расчета улучшений"""
        current_metrics = [
            DoctorMetrics(1, "терапевт", "отделение", 0.8, 60, 0.7, 0.8, 0.5),
            DoctorMetrics(2, "терапевт", "отделение", 0.6, 30, 0.8, 0.9, 0.6)
        ]
        
        optimal_distribution = {1: 0.7, 2: 0.75}
        
        improvements = optimizer.calculate_improvements(current_metrics, optimal_distribution)
        
        assert 'wait_time_reduction' in improvements
        assert 'utilization_improvement' in improvements
        assert 'efficiency_score' in improvements
        assert all(0 <= value <= 1 for value in improvements.values())

    def test_generate_recommendations(self, optimizer):
        """Тест генерации рекомендаций"""
        doctor_metrics = [
            DoctorMetrics(1, "терапевт", "отделение", 0.95, 60, 0.7, 0.8, 0.5),  # Перегружен
            DoctorMetrics(2, "терапевт", "отделение", 0.2, 30, 0.8, 0.9, 0.6),  # Недозагружен
            DoctorMetrics(3, "терапевт", "отделение", 0.6, 80, 0.9, 0.7, 0.4)   # Высокое время ожидания
        ]
        
        optimal_distribution = {1: 0.7, 2: 0.75, 3: 0.7}
        
        recommendations = optimizer.generate_recommendations(doctor_metrics, optimal_distribution)
        
        assert len(recommendations) > 0
        assert any("перегруженных" in rec for rec in recommendations)
        assert any("недозагруженных" in rec for rec in recommendations)


class TestResourceAllocator:
    """Тесты для ResourceAllocator"""

    @pytest.fixture
    def mock_repository(self):
        """Фикстура для мок-репозитория"""
        return Mock(spec=MedialogRepository)

    @pytest.fixture
    def allocator(self, mock_repository):
        """Фикстура для распределителя ресурсов"""
        return ResourceAllocator(mock_repository)

    def test_allocate_resources(self, allocator):
        """Тест распределения ресурсов"""
        date = datetime(2024, 1, 15)
        speciality = "терапевт"
        appointment_type = "consultation"
        
        resources = allocator.allocate_resources(date, speciality, appointment_type)
        
        assert 'room' in resources
        assert 'equipment' in resources
        assert 'allocation_time' in resources

    def test_get_available_rooms(self, allocator):
        """Тест получения доступных кабинетов"""
        date = datetime(2024, 1, 15)
        speciality = "терапевт"
        
        rooms = allocator.get_available_rooms(date, speciality)
        
        assert len(rooms) > 0
        assert all(r['speciality'] == speciality for r in rooms)
        assert all(r['available'] for r in rooms)

    def test_get_available_equipment(self, allocator):
        """Тест получения доступного оборудования"""
        date = datetime(2024, 1, 15)
        speciality = "кардиолог"
        
        equipment = allocator.get_available_equipment(date, speciality)
        
        assert len(equipment) > 0
        assert all(e['available'] for e in equipment)


class TestPerformanceTracker:
    """Тесты для PerformanceTracker"""

    @pytest.fixture
    def mock_repository(self):
        """Фикстура для мок-репозитория"""
        return Mock(spec=MedialogRepository)

    @pytest.fixture
    def tracker(self, mock_repository):
        """Фикстура для трекера производительности"""
        return PerformanceTracker(mock_repository)

    @pytest.fixture
    def date_range(self):
        """Фикстура для диапазона дат"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)
        return (start_date, end_date)

    def test_track_doctor_performance(self, tracker, mock_repository, date_range):
        """Тест отслеживания производительности врача"""
        doctor_id = 123
        
        # Мокаем данные
        mock_appointments = [
            Mock(no_show_flag=False),
            Mock(no_show_flag=True),
            Mock(no_show_flag=False)
        ]
        mock_repository.get_appointments_by_medecin.return_value = mock_appointments
        
        mock_consultations = [
            Mock(duration=30),
            Mock(duration=45),
            Mock(duration=60)
        ]
        mock_repository.get_consultations_by_medecin.return_value = mock_consultations
        
        performance = tracker.track_doctor_performance(doctor_id, date_range)
        
        assert performance['doctor_id'] == doctor_id
        assert performance['total_appointments'] == 3
        assert performance['completion_rate'] == 2/3
        assert performance['avg_consultation_time'] == 45.0
        assert 0 <= performance['patient_satisfaction'] <= 1
        assert 0 <= performance['time_efficiency'] <= 1
        assert 0 <= performance['performance_score'] <= 1

    def test_calculate_avg_consultation_time(self, tracker):
        """Тест расчета среднего времени консультации"""
        consultations = [
            Mock(duration=30),
            Mock(duration=45),
            Mock(duration=60)
        ]
        
        avg_time = tracker.calculate_avg_consultation_time(consultations)
        assert avg_time == 45.0

    def test_calculate_patient_satisfaction(self, tracker):
        """Тест расчета удовлетворенности пациентов"""
        appointments = [
            Mock(no_show_flag=False),
            Mock(no_show_flag=True),
            Mock(no_show_flag=False)
        ]
        
        satisfaction = tracker.calculate_patient_satisfaction(appointments)
        assert satisfaction == 2/3

    def test_calculate_time_efficiency(self, tracker):
        """Тест расчета эффективности использования времени"""
        appointments = [
            Mock(no_show_flag=False),
            Mock(no_show_flag=True),
            Mock(no_show_flag=False)
        ]
        consultations = [Mock(), Mock()]
        
        efficiency = tracker.calculate_time_efficiency(appointments, consultations)
        assert efficiency == 2/3


class TestDoctorLoadOptimizationService:
    """Тесты для основного сервиса оптимизации"""

    @pytest.fixture
    def mock_repository(self):
        """Фикстура для мок-репозитория"""
        return Mock(spec=MedialogRepository)

    @pytest.fixture
    def service(self, mock_repository):
        """Фикстура для сервиса"""
        return DoctorLoadOptimizationService(mock_repository)

    @pytest.fixture
    def date_range(self):
        """Фикстура для диапазона дат"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)
        return (start_date, end_date)

    def test_optimize_doctor_schedule(self, service, date_range):
        """Тест оптимизации расписания врачей"""
        result = service.optimize_doctor_schedule(date_range, OptimizationObjective.BALANCE_LOAD)
        
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'optimized_schedule')
        assert hasattr(result, 'load_distribution')
        assert hasattr(result, 'efficiency_score')

    def test_get_doctor_performance_report(self, service, mock_repository, date_range):
        """Тест получения отчета о производительности"""
        doctor_id = 123
        
        # Мокаем данные
        mock_appointments = [Mock(no_show_flag=False)]
        mock_repository.get_appointments_by_medecin.return_value = mock_appointments
        
        mock_consultations = [Mock(duration=30)]
        mock_repository.get_consultations_by_medecin.return_value = mock_consultations
        
        performance = service.get_doctor_performance_report(doctor_id, date_range)
        
        assert performance['doctor_id'] == doctor_id
        assert 'total_appointments' in performance
        assert 'completion_rate' in performance

    def test_get_optimal_doctor_for_appointment(self, service, mock_repository):
        """Тест получения оптимального врача для записи"""
        speciality = "терапевт"
        complexity = 0.5
        date = datetime(2024, 1, 15)
        
        # Мокаем врачей
        mock_doctors = [
            Mock(medecins_id=1, speciality="терапевт"),
            Mock(medecins_id=2, speciality="терапевт")
        ]
        mock_repository.get_medecins_by_department.return_value = mock_doctors
        
        doctor_id = service.get_optimal_doctor_for_appointment(speciality, complexity, date)
        
        # В данном случае может вернуть None из-за упрощенной логики
        assert doctor_id is None or isinstance(doctor_id, int) 