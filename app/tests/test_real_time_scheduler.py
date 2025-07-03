"""
Тесты для модуля адаптивного планирования реального времени
"""
import pytest
from unittest.mock import Mock
from datetime import datetime
from services.real_time_scheduler_service import RealTimeScheduler, EventProcessor, AdaptiveOptimizer, DynamicAdjuster

class TestEventProcessor:
    def test_process_event(self):
        repo = Mock()
        processor = EventProcessor(repo)
        event = processor.process_event('cancel', {'appointment_id': 1})
        assert event['event_type'] == 'cancel'
        assert event['event_data']['appointment_id'] == 1
        assert 'timestamp' in event

class TestAdaptiveOptimizer:
    def test_analyze_and_optimize(self):
        repo = Mock()
        repo.get_schedules_by_date.return_value = []
        repo.get_appointments_by_date.return_value = [Mock(status='cancelled', appointment_id=1)]
        optimizer = AdaptiveOptimizer(repo)
        result = optimizer.analyze_and_optimize(datetime(2024, 1, 1))
        assert 'conflicts' in result
        assert 'alternatives' in result
        assert 'efficiency' in result
        assert result['conflicts'][0]['type'] == 'cancel'

class TestDynamicAdjuster:
    def test_apply_adjustments(self):
        repo = Mock()
        adjuster = DynamicAdjuster(repo)
        alternatives = [{'appointment_id': 1}]
        results = adjuster.apply_adjustments(alternatives)
        assert results[0]['status'] == 'rescheduled'

class TestRealTimeScheduler:
    def test_handle_event(self):
        repo = Mock()
        repo.get_schedules_by_date.return_value = []
        repo.get_appointments_by_date.return_value = [Mock(status='cancelled', appointment_id=1)]
        scheduler = RealTimeScheduler(repo)
        result = scheduler.handle_event('cancel', {'appointment_id': 1, 'date': datetime(2024, 1, 1)})
        assert 'event' in result
        assert 'analysis' in result
        assert 'adjustments' in result 