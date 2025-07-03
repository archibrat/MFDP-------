"""
Тесты для модуля оптимизации нагрузки врачей
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from app.services.doctor_load_optimization.doctor_load_optimization_service import DoctorLoadOptimizationService
from app.services.doctor_load_optimization.schemas import (
    OptimizationRequest, OptimizationResult, OptimizationObjective,
    BatchOptimizationRequest, BatchOptimizationResponse,
    ResourceAllocationRequest, ResourceAllocationResult,
    DoctorProfile, PerformanceMetrics, ScheduleSlot
)
from app.services.doctor_load_optimization.database_connector import MedialogDatabaseConnector


class TestDoctorLoadOptimizationService:
    """Тесты для основного сервиса оптимизации нагрузки врачей"""

    @pytest.fixture
    def mock_db_connector(self):
        """Мок коннектора к базе данных"""
        return Mock(spec=MedialogDatabaseConnector)

    @pytest.fixture
    def service(self, mock_db_connector):
        """Экземпляр сервиса с мок-зависимостями"""
        return DoctorLoadOptimizationService(mock_db_connector)

    @pytest.fixture
    def sample_doctor_profiles(self):
        """Пример профилей врачей для тестирования"""
        return [
            DoctorProfile(
                doctor_id=1,
                speciality="Терапевт",
                department="Терапевтическое отделение",
                current_load=0.7,
                avg_wait_time=15.5,
                utilization_rate=0.8,
                patient_satisfaction=0.85,
                complexity_score=0.6,
                experience_years=10,
                max_patients_per_day=20
            ),
            DoctorProfile(
                doctor_id=2,
                speciality="Терапевт",
                department="Терапевтическое отделение",
                current_load=0.5,
                avg_wait_time=12.0,
                utilization_rate=0.75,
                patient_satisfaction=0.8,
                complexity_score=0.5,
                experience_years=5,
                max_patients_per_day=15
            )
        ]

    @pytest.fixture
    def sample_optimization_request(self):
        """Пример запроса на оптимизацию"""
        return OptimizationRequest(
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            objective=OptimizationObjective.BALANCE_LOAD,
            specialities=["Терапевт"]
        )

    def test_service_initialization(self, mock_db_connector):
        """Тест инициализации сервиса"""
        service = DoctorLoadOptimizationService(mock_db_connector)
        
        assert service.db_connector == mock_db_connector
        assert service.schedule_optimizer is not None
        assert service.resource_allocator is not None
        assert service.performance_tracker is not None
        assert service.logger is not None

    @patch('app.services.doctor_load_optimization.doctor_load_optimization_service.ScheduleOptimizer')
    def test_optimize_doctor_schedule_success(self, mock_optimizer_class, service, 
                                            sample_optimization_request, sample_doctor_profiles):
        """Тест успешной оптимизации расписания врачей"""
        # Настройка моков
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        service.db_connector.get_doctors_by_criteria.return_value = sample_doctor_profiles
        
        expected_result = OptimizationResult(
            optimized_schedule={1: [], 2: []},
            load_distribution={1: 0.75, 2: 0.67},
            wait_time_reduction=15.5,
            utilization_improvement=12.3,
            recommendations=["Оптимизировать расписание", "Перераспределить нагрузку"],
            efficiency_score=0.85,
            cost_savings=5000.0,
            processing_time=1.5
        )
        mock_optimizer.optimize_doctor_load.return_value = expected_result
        
        # Выполнение теста
        result = service.optimize_doctor_schedule(sample_optimization_request)
        
        # Проверки
        assert result == expected_result
        service.db_connector.get_doctors_by_criteria.assert_called_once()
        mock_optimizer.optimize_doctor_load.assert_called_once()

    def test_optimize_doctor_schedule_no_doctors_found(self, service, sample_optimization_request):
        """Тест оптимизации при отсутствии врачей"""
        service.db_connector.get_doctors_by_criteria.return_value = []
        
        with pytest.raises(ValueError, match="Не найдены врачи для оптимизации"):
            service.optimize_doctor_schedule(sample_optimization_request)

    def test_optimize_doctor_schedule_database_error(self, service, sample_optimization_request):
        """Тест обработки ошибки базы данных"""
        service.db_connector.get_doctors_by_criteria.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            service.optimize_doctor_schedule(sample_optimization_request)

    def test_batch_optimize_schedules_success(self, service, sample_optimization_request):
        """Тест успешной пакетной оптимизации"""
        # Настройка моков
        batch_request = BatchOptimizationRequest(
            optimization_requests=[sample_optimization_request, sample_optimization_request]
        )
        
        expected_result = OptimizationResult(
            optimized_schedule={1: []},
            load_distribution={1: 0.7},
            wait_time_reduction=10.0,
            utilization_improvement=8.0,
            recommendations=["Оптимизация завершена"],
            efficiency_score=0.8,
            cost_savings=3000.0,
            processing_time=1.0
        )
        
        with patch.object(service, 'optimize_doctor_schedule', return_value=expected_result):
            result = service.batch_optimize_schedules(batch_request)
        
        # Проверки
        assert result.total_processed == 2
        assert result.successful_optimizations == 2
        assert result.failed_optimizations == 0
        assert len(result.results) == 2
        assert result.processing_time > 0

    def test_batch_optimize_schedules_partial_failure(self, service, sample_optimization_request):
        """Тест пакетной оптимизации с частичными ошибками"""
        batch_request = BatchOptimizationRequest(
            optimization_requests=[sample_optimization_request, sample_optimization_request]
        )
        
        expected_result = OptimizationResult(
            optimized_schedule={1: []},
            load_distribution={1: 0.7},
            wait_time_reduction=10.0,
            utilization_improvement=8.0,
            recommendations=["Оптимизация завершена"],
            efficiency_score=0.8,
            cost_savings=3000.0,
            processing_time=1.0
        )
        
        # Первый вызов успешен, второй - ошибка
        with patch.object(service, 'optimize_doctor_schedule') as mock_optimize:
            mock_optimize.side_effect = [expected_result, Exception("Test error")]
            
            result = service.batch_optimize_schedules(batch_request)
        
        # Проверки
        assert result.total_processed == 2
        assert result.successful_optimizations == 1
        assert result.failed_optimizations == 1
        assert len(result.results) == 1

    def test_allocate_resources_success(self, service):
        """Тест успешного распределения ресурсов"""
        request = ResourceAllocationRequest(
            date=datetime.now(),
            speciality="Терапевт",
            appointment_type="Консультация",
            patient_count=20,
            complexity_level=0.7,
            duration_minutes=30
        )
        
        expected_result = ResourceAllocationResult(
            allocated_doctors=[1, 2, 3],
            allocated_rooms=[1, 2],
            schedule_slots=[],
            efficiency_score=0.85,
            cost_estimate=5000.0,
            recommendations=["Оптимальное распределение ресурсов"]
        )
        
        service.resource_allocator.allocate_resources.return_value = expected_result
        
        result = service.allocate_resources(request)
        
        assert result == expected_result
        service.resource_allocator.allocate_resources.assert_called_once_with(request)

    def test_get_doctor_performance_report_success(self, service):
        """Тест получения отчета о производительности врача"""
        doctor_id = 1
        period_start = datetime.now() - timedelta(days=30)
        period_end = datetime.now()
        
        expected_metrics = PerformanceMetrics(
            doctor_id=doctor_id,
            period_start=period_start,
            period_end=period_end,
            total_appointments=150,
            completed_appointments=145,
            avg_consultation_time=20.5,
            patient_satisfaction=0.85,
            utilization_rate=0.75,
            efficiency_score=0.85,
            recommendations=["Увеличить количество приемов"]
        )
        
        service.performance_tracker.track_doctor_performance.return_value = expected_metrics
        
        result = service.get_doctor_performance_report(doctor_id, period_start, period_end)
        
        assert result == expected_metrics
        service.performance_tracker.track_doctor_performance.assert_called_once_with(
            doctor_id, period_start, period_end
        )

    def test_get_optimal_doctor_for_appointment_success(self, service, sample_doctor_profiles):
        """Тест выбора оптимального врача для приема"""
        speciality = "Терапевт"
        complexity = 0.7
        date = datetime.now()
        
        service.db_connector.get_doctors_by_criteria.return_value = sample_doctor_profiles
        service.schedule_optimizer.get_optimal_doctor_for_appointment.return_value = 1
        
        result = service.get_optimal_doctor_for_appointment(speciality, complexity, date)
        
        assert result == 1
        service.db_connector.get_doctors_by_criteria.assert_called_once_with(
            specialities=[speciality], active_only=True
        )

    def test_get_optimal_doctor_for_appointment_no_doctors(self, service):
        """Тест выбора врача при отсутствии доступных врачей"""
        service.db_connector.get_doctors_by_criteria.return_value = []
        
        result = service.get_optimal_doctor_for_appointment("Терапевт", 0.7, datetime.now())
        
        assert result is None

    def test_get_load_balance_analysis_success(self, service):
        """Тест анализа балансировки нагрузки"""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)
        specialities = ["Терапевт", "Хирург"]
        
        expected_analysis = {
            "total_doctors": 10,
            "average_load": 0.75,
            "load_distribution": {"low": 2, "medium": 5, "high": 3},
            "recommendations": ["Перераспределить нагрузку", "Добавить врачей"]
        }
        
        service.db_connector.get_load_balance_data.return_value = expected_analysis
        
        result = service.get_load_balance_analysis(start_date, end_date, specialities)
        
        assert result == expected_analysis
        service.db_connector.get_load_balance_data.assert_called_once_with(
            start_date, end_date, specialities
        )

    def test_get_resource_utilization_report_success(self, service):
        """Тест получения отчета об использовании ресурсов"""
        date = datetime.now().date()
        specialities = ["Терапевт"]
        
        expected_report = {
            "date": date,
            "total_doctors": 15,
            "active_doctors": 12,
            "utilization_rate": 0.8,
            "resource_recommendations": ["Увеличить штат", "Оптимизировать расписание"]
        }
        
        service.db_connector.get_resource_utilization_data.return_value = expected_report
        
        result = service.get_resource_utilization_report(date, specialities)
        
        assert result == expected_report
        service.db_connector.get_resource_utilization_data.assert_called_once_with(
            date, specialities
        )

    def test_get_service_health_success(self, service):
        """Тест проверки состояния сервиса"""
        health_info = service.get_service_health()
        
        assert "status" in health_info
        assert "timestamp" in health_info
        assert "version" in health_info
        assert "dependencies" in health_info
        assert health_info["status"] == "healthy"

    def test_calculate_summary_metrics(self, service):
        """Тест расчета сводных метрик"""
        results = [
            OptimizationResult(
                optimized_schedule={1: []},
                load_distribution={1: 0.7},
                wait_time_reduction=10.0,
                utilization_improvement=8.0,
                recommendations=["Оптимизация 1"],
                efficiency_score=0.7,
                cost_savings=2000.0,
                processing_time=1.5
            ),
            OptimizationResult(
                optimized_schedule={2: []},
                load_distribution={2: 0.8},
                wait_time_reduction=12.0,
                utilization_improvement=10.0,
                recommendations=["Оптимизация 2"],
                efficiency_score=0.8,
                cost_savings=3000.0,
                processing_time=2.0
            )
        ]
        
        summary = service._calculate_summary_metrics(results)
        
        assert summary["total_cost_savings"] == 5000.0
        assert summary["average_efficiency_score"] == 0.75
        assert summary["average_processing_time"] == 1.75
        assert summary["total_optimizations"] == 2


class TestDoctorLoadOptimizationIntegration:
    """Интеграционные тесты для модуля оптимизации нагрузки врачей"""

    @pytest.fixture
    def real_db_connector(self):
        """Реальный коннектор к тестовой базе данных"""
        # В реальном проекте здесь была бы настройка тестовой БД
        return MedialogDatabaseConnector()

    def test_full_optimization_workflow(self, real_db_connector):
        """Тест полного рабочего процесса оптимизации"""
        service = DoctorLoadOptimizationService(real_db_connector)
        
        request = OptimizationRequest(
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            objective=OptimizationObjective.BALANCE_LOAD,
            specialities=["Терапевт"]
        )
        
        # Этот тест требует реальной базы данных
        # В тестовой среде можно использовать моки или тестовую БД
        pytest.skip("Требует настройки тестовой базы данных")

    def test_performance_under_load(self, real_db_connector):
        """Тест производительности под нагрузкой"""
        service = DoctorLoadOptimizationService(real_db_connector)
        
        # Создание множественных запросов для тестирования производительности
        requests = [
            OptimizationRequest(
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=i),
                objective=OptimizationObjective.BALANCE_LOAD,
                specialities=["Терапевт"]
            )
            for i in range(1, 6)
        ]
        
        batch_request = BatchOptimizationRequest(optimization_requests=requests)
        
        # Этот тест также требует реальной базы данных
        pytest.skip("Требует настройки тестовой базы данных")


if __name__ == "__main__":
    pytest.main([__file__]) 