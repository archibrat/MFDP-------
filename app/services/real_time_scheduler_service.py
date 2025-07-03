"""
Модуль адаптивного планирования реального времени
Реализует обработку событий расписания, адаптивную оптимизацию и динамическую корректировку
Адаптирован под схему БД Медиалог
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from enum import Enum

from app.repositories.medialog_repository import MedialogRepository
from app.models.medialog import MedialogPatient, Medecin, Consultation, Appointment
from app.services.ml_production.no_show_predictor import NoShowPredictor
from app.services.doctor_load_optimization_service import DoctorLoadOptimizationService


class EventType(str, Enum):
    """Типы событий расписания"""
    CANCELLATION = "cancellation"
    DELAY = "delay"
    EMERGENCY = "emergency"
    NO_SHOW = "no_show"
    DOCTOR_UNAVAILABLE = "doctor_unavailable"
    PATIENT_ARRIVAL = "patient_arrival"


class EventProcessor:
    """Процессор событий расписания (отмена, задержка, экстренный случай)"""
    def __init__(self, repository: MedialogRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def process_event(self, event_type: EventType, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка события и формирование запроса на адаптацию"""
        self.logger.info(f"Обработка события: {event_type}, данные: {event_data}")
        
        # Обогащение данных события
        enriched_data = self._enrich_event_data(event_type, event_data)
        
        return {
            'event_type': event_type,
            'event_data': enriched_data,
            'timestamp': datetime.utcnow(),
            'priority': self._calculate_event_priority(event_type, enriched_data)
        }

    def _enrich_event_data(self, event_type: EventType, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обогащение данных события дополнительной информацией"""
        enriched = event_data.copy()
        
        if 'planning_id' in event_data:
            # Получаем данные записи
            planning = self.repository.get_planning_by_id(event_data['planning_id'])
            if planning:
                enriched['patient_id'] = planning.patients_id
                enriched['doctor_id'] = planning.medecins_creator_id
                enriched['appointment_time'] = planning.date_cons
                enriched['visit_type'] = planning.motif
                enriched['is_cito'] = planning.cito == 'Y'
        
        return enriched

    def _calculate_event_priority(self, event_type: EventType, event_data: Dict[str, Any]) -> int:
        """Расчет приоритета события"""
        base_priority = {
            EventType.EMERGENCY: 10,
            EventType.DOCTOR_UNAVAILABLE: 8,
            EventType.NO_SHOW: 6,
            EventType.CANCELLATION: 4,
            EventType.DELAY: 3,
            EventType.PATIENT_ARRIVAL: 2
        }
        
        priority = base_priority.get(event_type, 1)
        
        # Дополнительные факторы
        if event_data.get('is_cito'):
            priority += 5
        if event_data.get('is_vip'):
            priority += 3
            
        return priority


class AdaptiveOptimizer:
    """Адаптивный оптимизатор расписания"""
    def __init__(self, repository: MedialogRepository, 
                 no_show_predictor: Optional[NoShowPredictor] = None,
                 load_optimizer: Optional[DoctorLoadOptimizationService] = None):
        self.repository = repository
        self.no_show_predictor = no_show_predictor
        self.load_optimizer = load_optimizer
        self.logger = logging.getLogger(__name__)

    def analyze_and_optimize(self, date: datetime, event_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Анализ текущего состояния и оптимизация расписания на дату"""
        try:
            # Получение данных расписания
            schedules = self.repository.get_schedules_by_date(date)
            appointments = self.repository.get_appointments_by_date(date)
            
            # Анализ конфликтов и возможностей оптимизации
            conflicts = self._find_conflicts(schedules, appointments, event_context)
            alternatives = self._generate_alternatives(schedules, appointments, conflicts, event_context)
            efficiency = self._evaluate_efficiency(schedules, appointments)
            
            # Интеграция с модулем прогнозирования неявок
            no_show_insights = self._analyze_no_show_risks(appointments)
            
            # Интеграция с модулем оптимизации нагрузки
            load_optimization = self._analyze_load_optimization(date, appointments)
            
            return {
                'conflicts': conflicts,
                'alternatives': alternatives,
                'efficiency': efficiency,
                'no_show_insights': no_show_insights,
                'load_optimization': load_optimization,
                'recommendations': self._generate_recommendations(conflicts, no_show_insights, load_optimization)
            }
        except Exception as e:
            self.logger.error(f"Ошибка анализа и оптимизации: {e}")
            return {
                'conflicts': [],
                'alternatives': [],
                'efficiency': 0.0,
                'no_show_insights': {},
                'load_optimization': {},
                'recommendations': ['Ошибка анализа']
            }

    def _find_conflicts(self, schedules: List, appointments: List, 
                       event_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Выявление конфликтов в расписании"""
        conflicts = []
        
        # Анализ отмен
        for app in appointments:
            if hasattr(app, 'cancelled') and app.cancelled == 'Y':
                conflicts.append({
                    'type': 'cancellation',
                    'planning_id': app.planning_id,
                    'patient_id': app.patients_id,
                    'doctor_id': app.medecins_creator_id,
                    'severity': 'high'
                })
        
        # Анализ перегрузки врачей
        doctor_loads = {}
        for app in appointments:
            if hasattr(app, 'medecins_creator_id') and app.medecins_creator_id:
                doctor_id = app.medecins_creator_id
                if doctor_id not in doctor_loads:
                    doctor_loads[doctor_id] = []
                doctor_loads[doctor_id].append(app)
        
        for doctor_id, apps in doctor_loads.items():
            if len(apps) > 25:  # Предполагаем максимум 25 приемов в день
                conflicts.append({
                    'type': 'overload',
                    'doctor_id': doctor_id,
                    'appointment_count': len(apps),
                    'severity': 'medium'
                })
        
        # Анализ временных конфликтов
        time_conflicts = self._find_time_conflicts(appointments)
        conflicts.extend(time_conflicts)
        
        return conflicts

    def _find_time_conflicts(self, appointments: List) -> List[Dict[str, Any]]:
        """Поиск временных конфликтов"""
        conflicts = []
        
        # Группировка по врачам
        doctor_appointments = {}
        for app in appointments:
            if hasattr(app, 'medecins_creator_id') and app.medecins_creator_id:
                doctor_id = app.medecins_creator_id
                if doctor_id not in doctor_appointments:
                    doctor_appointments[doctor_id] = []
                doctor_appointments[doctor_id].append(app)
        
        # Проверка перекрытий времени
        for doctor_id, apps in doctor_appointments.items():
            sorted_apps = sorted(apps, key=lambda x: x.date_cons)
            for i in range(1, len(sorted_apps)):
                prev_app = sorted_apps[i-1]
                curr_app = sorted_apps[i]
                
                # Предполагаем 30 минут на прием
                prev_end = prev_app.date_cons + timedelta(minutes=30)
                if curr_app.date_cons < prev_end:
                    conflicts.append({
                        'type': 'time_overlap',
                        'doctor_id': doctor_id,
                        'planning_id_1': prev_app.planning_id,
                        'planning_id_2': curr_app.planning_id,
                        'overlap_minutes': (prev_end - curr_app.date_cons).total_seconds() / 60,
                        'severity': 'high'
                    })
        
        return conflicts

    def _generate_alternatives(self, schedules: List, appointments: List, 
                             conflicts: List[Dict], event_context: Optional[Dict]) -> List[Dict[str, Any]]:
        """Генерация альтернативных вариантов для конфликтных ситуаций"""
        alternatives = []
        
        for conflict in conflicts:
            if conflict['type'] == 'cancellation':
                # Предложить ближайший свободный слот
                alternative = self._find_alternative_slot(
                    conflict['doctor_id'], 
                    conflict.get('patient_id'),
                    event_context
                )
                if alternative:
                    alternatives.append({
                        'conflict_id': conflict['planning_id'],
                        'alternative_slot': alternative,
                        'type': 'reschedule'
                    })
            
            elif conflict['type'] == 'overload':
                # Предложить перераспределение нагрузки
                alternatives.append({
                    'conflict_id': conflict['doctor_id'],
                    'alternative_slot': 'redistribute_load',
                    'type': 'load_balancing'
                })
            
            elif conflict['type'] == 'time_overlap':
                # Предложить сдвиг времени
                alternatives.append({
                    'conflict_id': conflict['planning_id_2'],
                    'alternative_slot': 'delay_appointment',
                    'type': 'time_adjustment',
                    'delay_minutes': conflict['overlap_minutes']
                })
        
        return alternatives

    def _find_alternative_slot(self, doctor_id: int, patient_id: Optional[int], 
                             event_context: Optional[Dict]) -> Optional[Dict]:
        """Поиск альтернативного слота"""
        try:
            # Поиск свободных слотов у того же врача
            current_date = datetime.utcnow()
            end_date = current_date + timedelta(days=7)
            
            # Получаем все записи врача на ближайшую неделю
            doctor_appointments = self.repository.get_appointments_by_medecin(
                doctor_id, current_date, end_date
            )
            
            # Генерируем возможные слоты (каждый час с 9 до 17)
            possible_slots = []
            for day in range(7):
                date = current_date + timedelta(days=day)
                if date.weekday() < 5:  # Только рабочие дни
                    for hour in range(9, 17):
                        slot_time = date.replace(hour=hour, minute=0, second=0, microsecond=0)
                        possible_slots.append(slot_time)
            
            # Исключаем занятые слоты
            occupied_slots = set()
            for app in doctor_appointments:
                if hasattr(app, 'date_cons'):
                    occupied_slots.add(app.date_cons.replace(minute=0, second=0, microsecond=0))
            
            available_slots = [slot for slot in possible_slots if slot not in occupied_slots]
            
            if available_slots:
                return {
                    'doctor_id': doctor_id,
                    'appointment_time': available_slots[0],
                    'slot_type': 'alternative'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска альтернативного слота: {e}")
            return None

    def _evaluate_efficiency(self, schedules: List, appointments: List) -> float:
        """Оценка эффективности расписания"""
        if not appointments:
            return 1.0
        
        # Расчет различных метрик эффективности
        total_appointments = len(appointments)
        cancelled_appointments = len([a for a in appointments if hasattr(a, 'cancelled') and a.cancelled == 'Y'])
        completion_rate = (total_appointments - cancelled_appointments) / total_appointments if total_appointments > 0 else 0.0
        
        # Расчет времени ожидания (упрощенный)
        wait_times = [30 for _ in appointments]  # Заглушка
        avg_wait_time = float(np.mean(wait_times))
        
        # Нормализованная оценка эффективности
        efficiency = 1.0 - avg_wait_time / 120  # Чем меньше ожидание, тем выше эффективность
        efficiency = efficiency * 0.7 + completion_rate * 0.3  # Взвешенная оценка
        
        return max(0.0, min(1.0, efficiency))

    def _analyze_no_show_risks(self, appointments: List) -> Dict[str, Any]:
        """Анализ рисков неявок с использованием модуля прогнозирования"""
        if not self.no_show_predictor:
            return {}
        
        try:
            high_risk_appointments = []
            total_risk_score = 0.0
            
            for app in appointments:
                if hasattr(app, 'planning_id') and hasattr(app, 'patients_id'):
                    # Получаем прогноз неявки
                    features = self._extract_no_show_features(app)
                    if features:
                        risk_score = self.no_show_predictor.predict(features)
                        total_risk_score += risk_score
                        
                        if risk_score > 0.6:  # Высокий риск
                            high_risk_appointments.append({
                                'planning_id': app.planning_id,
                                'patient_id': app.patients_id,
                                'risk_score': risk_score
                            })
            
            avg_risk = total_risk_score / len(appointments) if appointments else 0.0
            
            return {
                'high_risk_count': len(high_risk_appointments),
                'average_risk': avg_risk,
                'high_risk_appointments': high_risk_appointments,
                'recommendations': self._generate_no_show_recommendations(high_risk_appointments)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа рисков неявок: {e}")
            return {}

    def _extract_no_show_features(self, appointment) -> Optional[Dict[str, Any]]:
        """Извлечение признаков для прогнозирования неявок"""
        try:
            if not hasattr(appointment, 'patients_id') or not hasattr(appointment, 'date_cons'):
                return None
            
            # Базовые признаки
            features = {
                'appointment_hour': appointment.date_cons.hour,
                'appointment_weekday': appointment.date_cons.weekday(),
                'is_weekend': int(appointment.date_cons.weekday() >= 5),
                'month': appointment.date_cons.month,
                'advance_booking_days': 0  # Заглушка
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения признаков: {e}")
            return None

    def _analyze_load_optimization(self, date: datetime, appointments: List) -> Dict[str, Any]:
        """Анализ оптимизации нагрузки с использованием модуля балансировки"""
        if not self.load_optimizer:
            return {}
        
        try:
            # Группировка по врачам
            doctor_loads = {}
            for app in appointments:
                if hasattr(app, 'medecins_creator_id') and app.medecins_creator_id:
                    doctor_id = app.medecins_creator_id
                    if doctor_id not in doctor_loads:
                        doctor_loads[doctor_id] = []
                    doctor_loads[doctor_id].append(app)
            
            # Анализ распределения нагрузки
            load_distribution = {}
            for doctor_id, apps in doctor_loads.items():
                load_distribution[doctor_id] = {
                    'appointment_count': len(apps),
                    'load_percentage': min(1.0, len(apps) / 20),  # Норма: 20 приемов в день
                    'overloaded': len(apps) > 25
                }
            
            # Определение перегруженных врачей
            overloaded_doctors = [
                doctor_id for doctor_id, data in load_distribution.items()
                if data['overloaded']
            ]
            
            return {
                'load_distribution': load_distribution,
                'overloaded_doctors': overloaded_doctors,
                'recommendations': self._generate_load_recommendations(load_distribution)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа оптимизации нагрузки: {e}")
            return {}

    def _generate_recommendations(self, conflicts: List, no_show_insights: Dict, 
                                load_optimization: Dict) -> List[str]:
        """Генерация общих рекомендаций"""
        recommendations = []
        
        # Рекомендации по конфликтам
        if conflicts:
            recommendations.append(f"Обнаружено {len(conflicts)} конфликтов в расписании")
        
        # Рекомендации по неявкам
        if no_show_insights.get('high_risk_count', 0) > 0:
            recommendations.append(f"Высокий риск неявок у {no_show_insights['high_risk_count']} пациентов")
        
        # Рекомендации по нагрузке
        if load_optimization.get('overloaded_doctors'):
            recommendations.append(f"Перегружено {len(load_optimization['overloaded_doctors'])} врачей")
        
        if not recommendations:
            recommendations.append("Расписание оптимизировано")
        
        return recommendations

    def _generate_no_show_recommendations(self, high_risk_appointments: List) -> List[str]:
        """Генерация рекомендаций по снижению риска неявок"""
        recommendations = []
        
        if high_risk_appointments:
            recommendations.append("Отправить SMS-напоминания высокорисковым пациентам")
            recommendations.append("Рассмотреть двойное бронирование для слотов с риском >0.8")
        
        return recommendations

    def _generate_load_recommendations(self, load_distribution: Dict) -> List[str]:
        """Генерация рекомендаций по оптимизации нагрузки"""
        recommendations = []
        
        overloaded_count = sum(1 for data in load_distribution.values() if data['overloaded'])
        if overloaded_count > 0:
            recommendations.append(f"Перераспределить нагрузку с {overloaded_count} перегруженных врачей")
        
        return recommendations


class DynamicAdjuster:
    """Динамическая корректировка расписания"""
    def __init__(self, repository: MedialogRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def apply_adjustments(self, alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Внесение изменений в расписание на основе альтернатив"""
        results = []
        
        for alt in alternatives:
            try:
                if alt['type'] == 'reschedule':
                    result = self._reschedule_appointment(alt)
                elif alt['type'] == 'load_balancing':
                    result = self._redistribute_load(alt)
                elif alt['type'] == 'time_adjustment':
                    result = self._adjust_appointment_time(alt)
                else:
                    result = {'status': 'unknown_type', 'planning_id': alt.get('conflict_id')}
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Ошибка применения корректировки: {e}")
                results.append({
                    'status': 'error',
                    'planning_id': alt.get('conflict_id'),
                    'error': str(e)
                })
        
        return results

    def _reschedule_appointment(self, alternative: Dict) -> Dict[str, Any]:
        """Перенос записи на альтернативное время"""
        try:
            planning_id = alternative['conflict_id']
            new_slot = alternative['alternative_slot']
            
            # Обновление записи в базе данных
            # В реальной реализации здесь был бы UPDATE запрос
            self.logger.info(f"Перенос записи {planning_id} на {new_slot['appointment_time']}")
            
            return {
                'status': 'rescheduled',
                'planning_id': planning_id,
                'new_time': new_slot['appointment_time'],
                'doctor_id': new_slot['doctor_id']
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка переноса записи: {e}")
            return {'status': 'error', 'planning_id': planning_id, 'error': str(e)}

    def _redistribute_load(self, alternative: Dict) -> Dict[str, Any]:
        """Перераспределение нагрузки между врачами"""
        try:
            doctor_id = alternative['conflict_id']
            
            # Логика перераспределения нагрузки
            self.logger.info(f"Перераспределение нагрузки для врача {doctor_id}")
            
            return {
                'status': 'load_redistributed',
                'doctor_id': doctor_id,
                'action': 'redistribute_load'
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка перераспределения нагрузки: {e}")
            return {'status': 'error', 'doctor_id': doctor_id, 'error': str(e)}

    def _adjust_appointment_time(self, alternative: Dict) -> Dict[str, Any]:
        """Корректировка времени записи"""
        try:
            planning_id = alternative['conflict_id']
            delay_minutes = alternative.get('delay_minutes', 30)
            
            # Логика корректировки времени
            self.logger.info(f"Корректировка времени записи {planning_id} на {delay_minutes} минут")
            
            return {
                'status': 'time_adjusted',
                'planning_id': planning_id,
                'delay_minutes': delay_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка корректировки времени: {e}")
            return {'status': 'error', 'planning_id': planning_id, 'error': str(e)}


class RealTimeScheduler:
    """Планировщик реального времени"""
    def __init__(self, repository: MedialogRepository,
                 no_show_predictor: Optional[NoShowPredictor] = None,
                 load_optimizer: Optional[DoctorLoadOptimizationService] = None):
        self.repository = repository
        self.event_processor = EventProcessor(repository)
        self.optimizer = AdaptiveOptimizer(repository, no_show_predictor, load_optimizer)
        self.adjuster = DynamicAdjuster(repository)
        self.logger = logging.getLogger(__name__)

    def handle_event(self, event_type: EventType, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Главный обработчик событий расписания"""
        try:
            # Обработка события
            event = self.event_processor.process_event(event_type, event_data)
            
            # Определение даты для анализа
            date = event_data.get('date', datetime.utcnow())
            
            # Анализ и оптимизация
            analysis = self.optimizer.analyze_and_optimize(date, event['event_data'])
            
            # Применение корректировок
            adjustments = self.adjuster.apply_adjustments(analysis['alternatives'])
            
            return {
                'event': event,
                'analysis': analysis,
                'adjustments': adjustments,
                'success': len([a for a in adjustments if a.get('status') == 'success']) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки события: {e}")
            return {
                'event': {'event_type': event_type, 'error': str(e)},
                'analysis': {},
                'adjustments': [],
                'success': False
            }

    def get_schedule_snapshot(self, date: datetime) -> Dict[str, Any]:
        """Получение снэпшота расписания на дату"""
        try:
            schedules = self.repository.get_schedules_by_date(date)
            appointments = self.repository.get_appointments_by_date(date)
            
            return {
                'date': date,
                'total_appointments': len(appointments),
                'cancelled_appointments': len([a for a in appointments if hasattr(a, 'cancelled') and a.cancelled == 'Y']),
                'doctor_count': len(set(a.medecins_creator_id for a in appointments if hasattr(a, 'medecins_creator_id'))),
                'schedule_efficiency': self.optimizer._evaluate_efficiency(schedules, appointments)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения снэпшота расписания: {e}")
            return {'error': str(e)} 