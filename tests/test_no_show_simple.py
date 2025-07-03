#!/usr/bin/env python3
"""
Простой тест модуля прогнозирования неявок
"""

import sys
import os
sys.path.append('.')

from app.services.no_show_prediction.schemas import (
    PatientProfile, AppointmentContext, RiskLevel
)
from app.services.no_show_prediction.feature_extractor import FeatureExtractor
from app.services.no_show_prediction.risk import RiskAssessment
from datetime import datetime, timedelta


def test_patient_profile():
    """Тест создания профиля пациента"""
    print("Тестирование создания профиля пациента...")
    
    profile = PatientProfile(
        patient_id=1,
        age=30,
        gender="M",
        visit_history_count=5,
        avg_interval_between_visits=30.5,
        no_show_history_rate=0.2,
        phone_confirmed=True
    )
    
    assert profile.patient_id == 1
    assert profile.age == 30
    assert profile.gender == "M"
    print("✓ Профиль пациента создан успешно")


def test_appointment_context():
    """Тест создания контекста записи"""
    print("Тестирование создания контекста записи...")
    
    appointment_time = datetime.utcnow() + timedelta(days=7)
    
    context = AppointmentContext(
        appointment_id=1,
        patient_id=1,
        doctor_id=1,
        appointment_time=appointment_time,
        visit_type="consultation",
        advance_booking_days=7,
        is_repeat_visit=True,
        reminder_sent=False
    )
    
    assert context.appointment_id == 1
    assert context.patient_id == 1
    assert context.doctor_id == 1
    print("✓ Контекст записи создан успешно")


def test_feature_extraction():
    """Тест извлечения признаков"""
    print("Тестирование извлечения признаков...")
    
    patient = PatientProfile(
        patient_id=1,
        age=30,
        gender="M",
        visit_history_count=5,
        avg_interval_between_visits=30.5,
        no_show_history_rate=0.2,
        phone_confirmed=True
    )
    
    appointment_time = datetime.utcnow() + timedelta(days=7)
    appointment = AppointmentContext(
        appointment_id=1,
        patient_id=1,
        doctor_id=1,
        appointment_time=appointment_time,
        visit_type="consultation",
        advance_booking_days=7,
        is_repeat_visit=True,
        reminder_sent=False
    )
    
    features = FeatureExtractor.extract_features(patient, appointment)
    
    assert 'age' in features
    assert 'gender_encoded' in features
    assert 'visit_history_count' in features
    assert 'avg_interval_days' in features
    assert 'historical_no_show_rate' in features
    assert 'phone_confirmed' in features
    assert 'advance_booking_days' in features
    assert 'is_repeat_visit' in features
    assert 'reminder_sent' in features
    
    print("✓ Признаки извлечены успешно")
    print(f"  Количество признаков: {len(features)}")


def test_risk_assessment():
    """Тест оценки риска"""
    print("Тестирование оценки риска...")
    
    assessor = RiskAssessment()
    
    # Тест низкого риска
    risk_low = assessor.assess_risk(0.1)
    assert risk_low == "low"
    
    # Тест среднего риска
    risk_medium = assessor.assess_risk(0.5)
    assert risk_medium == "medium"
    
    # Тест высокого риска
    risk_high = assessor.assess_risk(0.8)
    assert risk_high == "high"
    
    print("✓ Оценка риска работает корректно")


def test_recommendations():
    """Тест получения рекомендаций"""
    print("Тестирование получения рекомендаций...")
    
    assessor = RiskAssessment()
    
    low_rec = assessor.get_recommendation("low")
    medium_rec = assessor.get_recommendation("medium")
    high_rec = assessor.get_recommendation("high")
    
    assert len(low_rec) > 0
    assert len(medium_rec) > 0
    assert len(high_rec) > 0
    
    print("✓ Рекомендации генерируются корректно")
    print(f"  Низкий риск: {low_rec[:50]}...")
    print(f"  Средний риск: {medium_rec[:50]}...")
    print(f"  Высокий риск: {high_rec[:50]}...")


def test_batch_feature_extraction():
    """Тест batch-извлечения признаков"""
    print("Тестирование batch-извлечения признаков...")
    
    patients = [
        PatientProfile(
            patient_id=1, age=30, gender="M", visit_history_count=5,
            avg_interval_between_visits=30.5, no_show_history_rate=0.2,
            phone_confirmed=True
        ),
        PatientProfile(
            patient_id=2, age=45, gender="F", visit_history_count=3,
            avg_interval_between_visits=45.0, no_show_history_rate=0.1,
            phone_confirmed=False
        )
    ]
    
    appointment_time = datetime.utcnow() + timedelta(days=7)
    appointments = [
        AppointmentContext(
            appointment_id=1, patient_id=1, doctor_id=1,
            appointment_time=appointment_time, visit_type="consultation",
            advance_booking_days=7, is_repeat_visit=True, reminder_sent=False
        ),
        AppointmentContext(
            appointment_id=2, patient_id=2, doctor_id=1,
            appointment_time=appointment_time, visit_type="examination",
            advance_booking_days=14, is_repeat_visit=False, reminder_sent=True
        )
    ]
    
    features_matrix = FeatureExtractor.extract_batch_features(patients, appointments)
    
    assert features_matrix.shape[0] == 2  # 2 пациента
    assert features_matrix.shape[1] > 0   # Признаки
    
    print("✓ Batch-извлечение признаков работает корректно")
    print(f"  Размер матрицы: {features_matrix.shape}")


def main():
    """Основная функция тестирования"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ ПРОГНОЗИРОВАНИЯ НЕЯВОК")
    print("=" * 60)
    
    try:
        test_patient_profile()
        test_appointment_context()
        test_feature_extraction()
        test_risk_assessment()
        test_recommendations()
        test_batch_feature_extraction()
        
        print("\n" + "=" * 60)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА В ТЕСТЕ: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 