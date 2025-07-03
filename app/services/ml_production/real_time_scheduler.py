"""
Real-Time Scheduler для реагирования на события
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

from app.models.ml_production import EventType
from app.services.ml_production.dal import MLDataAccessLayer


logger = logging.getLogger(__name__)


@dataclass
class DoctorState:
    """Состояние врача"""
    doctor_id: int
    free_at: datetime
    current_patient: Optional[int] = None
    queue_length: int = 0


class RealTimeScheduler:
    """Планировщик реального времени"""
    
    def __init__(self, dal: MLDataAccessLayer):
        self.dal = dal

    def handle_event(self, event_type: EventType, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает события планирования в реальном времени"""
        try:
            planning_id = event_data.get('planning_id')
            if not planning_id:
                return {'error': 'planning_id не указан'}
            
            logger.info(f"Обработка события {event_type} для planning_id {planning_id}")
            
            # Получение текущего состояния
            doctor_states = self._get_doctor_states()
            waiting_patients = self.dal.get_waiting_patients()
            
            # Обработка события
            if event_type == EventType.ARRIVE_DATE:
                result = self._handle_patient_arrival(planning_id, event_data, doctor_states, waiting_patients)
            elif event_type == EventType.CANCELLED:
                result = self._handle_cancellation(planning_id, event_data, doctor_states, waiting_patients)
            elif event_type == EventType.CONS_DURATION:
                result = self._handle_duration_change(planning_id, event_data, doctor_states, waiting_patients)
            elif event_type == EventType.NO_SHOW:
                result = self._handle_no_show(planning_id, event_data, doctor_states, waiting_patients)
            else:
                result = {'error': f'Неизвестный тип события: {event_type}'}
            
            # Сохранение снимка изменений
            if 'new_time' in result or 'new_doctor' in result:
                self._save_snapshot(planning_id, event_type, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки события {event_type}: {e}")
            return {'error': str(e)}

    def _get_doctor_states(self) -> Dict[int, DoctorState]:
        """Получает текущее состояние врачей"""
        try:
            # Здесь должен быть реальный запрос к БД для получения состояния врачей
            # Пока используем заглушку
            current_time = datetime.utcnow()
            
            # Заглушка: создаем состояния для нескольких врачей
            states = {}
            for doctor_id in range(1, 11):  # Врачи с ID 1-10
                states[doctor_id] = DoctorState(
                    doctor_id=doctor_id,
                    free_at=current_time + timedelta(minutes=doctor_id * 5),  # Разное время освобождения
                    queue_length=doctor_id % 3  # Разная длина очереди
                )
            
            return states
            
        except Exception as e:
            logger.error(f"Ошибка получения состояния врачей: {e}")
            return {}

    def _handle_patient_arrival(self, planning_id: int, event_data: Dict[str, Any],
                              doctor_states: Dict[int, DoctorState], 
                              waiting_patients: List[Dict]) -> Dict[str, Any]:
        """Обрабатывает прибытие пациента"""
        try:
            current_time = datetime.utcnow()
            arrival_time = event_data.get('arrival_time', current_time)
            
            # Находим врача для пациента
            doctor_id = event_data.get('doctor_id')
            if not doctor_id or doctor_id not in doctor_states:
                return {'error': 'Врач не найден'}
            
            doctor_state = doctor_states[doctor_id]
            
            # Рассчитываем время приема
            if doctor_state.free_at <= arrival_time:
                # Врач свободен, можем принять сразу
                new_time = arrival_time
            else:
                # Врач занят, ставим в очередь
                new_time = doctor_state.free_at
            
            # Обновляем время в БД
            self.dal.update_planning_time(planning_id, new_time)
            
            # Отправляем уведомление через WebSocket (заглушка)
            self._send_websocket_notification(planning_id, {
                'type': 'patient_arrived',
                'new_time': new_time.isoformat(),
                'doctor_id': doctor_id
            })
            
            return {
                'planning_id': planning_id,
                'old_time': event_data.get('scheduled_time'),
                'new_time': new_time,
                'delay_minutes': (new_time - arrival_time).total_seconds() / 60,
                'notifications_sent': ['websocket']
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки прибытия пациента: {e}")
            return {'error': str(e)}

    def _handle_cancellation(self, planning_id: int, event_data: Dict[str, Any],
                           doctor_states: Dict[int, DoctorState], 
                           waiting_patients: List[Dict]) -> Dict[str, Any]:
        """Обрабатывает отмену записи"""
        try:
            # Освобождаем слот
            doctor_id = event_data.get('doctor_id')
            cancelled_time = event_data.get('scheduled_time')
            
            if not doctor_id or doctor_id not in doctor_states:
                return {'error': 'Врач не найден'}
            
            # Перепланируем ожидающих пациентов
            rescheduled = []
            for patient in waiting_patients:
                if patient['priority'] >= 5:  # Высокий приоритет
                    # Назначаем на освободившееся время
                    self.dal.update_planning_time(patient['planning_id'], cancelled_time)
                    rescheduled.append(patient['planning_id'])
                    break
            
            # Уведомления
            notifications = []
            if rescheduled:
                self._send_websocket_notification(rescheduled[0], {
                    'type': 'rescheduled_earlier',
                    'new_time': cancelled_time.isoformat() if cancelled_time else None
                })
                notifications.append('websocket')
            
            return {
                'planning_id': planning_id,
                'status': 'cancelled',
                'rescheduled_patients': rescheduled,
                'notifications_sent': notifications
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки отмены: {e}")
            return {'error': str(e)}

    def _handle_duration_change(self, planning_id: int, event_data: Dict[str, Any],
                              doctor_states: Dict[int, DoctorState], 
                              waiting_patients: List[Dict]) -> Dict[str, Any]:
        """Обрабатывает изменение длительности консультации"""
        try:
            actual_duration = event_data.get('actual_duration', 30)  # минуты
            expected_duration = event_data.get('expected_duration', 30)
            doctor_id = event_data.get('doctor_id')
            
            duration_diff = actual_duration - expected_duration
            
            if abs(duration_diff) <= 10:  # Отклонение в пределах нормы
                return {'planning_id': planning_id, 'status': 'normal_duration'}
            
            # Значительное отклонение - пересчитываем расписание
            adjustments = []
            
            if duration_diff > 10:  # Консультация затянулась
                # Сдвигаем следующих пациентов
                delay_minutes = duration_diff
                
                for patient in waiting_patients:
                    if patient['priority'] < 8:  # Не срочные
                        old_time = patient['heure']
                        new_time = old_time + timedelta(minutes=delay_minutes)
                        
                        self.dal.update_planning_time(patient['planning_id'], new_time)
                        
                        adjustments.append({
                            'planning_id': patient['planning_id'],
                            'old_time': old_time,
                            'new_time': new_time,
                            'delay_minutes': delay_minutes
                        })
                        
                        # Уведомление о задержке
                        self._send_websocket_notification(patient['planning_id'], {
                            'type': 'delayed',
                            'delay_minutes': delay_minutes,
                            'new_time': new_time.isoformat()
                        })
            
            return {
                'planning_id': planning_id,
                'duration_diff': duration_diff,
                'adjustments': adjustments,
                'notifications_sent': ['websocket'] if adjustments else []
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки изменения длительности: {e}")
            return {'error': str(e)}

    def _handle_no_show(self, planning_id: int, event_data: Dict[str, Any],
                       doctor_states: Dict[int, DoctorState], 
                       waiting_patients: List[Dict]) -> Dict[str, Any]:
        """Обрабатывает неявку пациента"""
        try:
            doctor_id = event_data.get('doctor_id')
            noshow_time = event_data.get('scheduled_time', datetime.utcnow())
            
            # Проверяем, есть ли резервный пациент (double booking)
            reserve_patients = [p for p in waiting_patients 
                              if p.get('is_reserve', False)]
            
            promoted = []
            if reserve_patients:
                # Продвигаем резервного пациента
                reserve_patient = reserve_patients[0]
                self.dal.update_planning_time(reserve_patient['planning_id'], noshow_time)
                promoted.append(reserve_patient['planning_id'])
                
                self._send_websocket_notification(reserve_patient['planning_id'], {
                    'type': 'promoted_from_reserve',
                    'new_time': noshow_time.isoformat()
                })
            
            # Или продвигаем следующего в очереди
            elif waiting_patients:
                next_patient = waiting_patients[0]
                old_time = next_patient['heure']
                
                self.dal.update_planning_time(next_patient['planning_id'], noshow_time)
                promoted.append(next_patient['planning_id'])
                
                advance_minutes = (old_time - noshow_time).total_seconds() / 60
                
                self._send_websocket_notification(next_patient['planning_id'], {
                    'type': 'moved_earlier',
                    'advance_minutes': advance_minutes,
                    'new_time': noshow_time.isoformat()
                })
            
            return {
                'planning_id': planning_id,
                'status': 'no_show',
                'promoted_patients': promoted,
                'notifications_sent': ['websocket'] if promoted else []
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки неявки: {e}")
            return {'error': str(e)}

    def _save_snapshot(self, planning_id: int, event_type: EventType, 
                      result: Dict[str, Any]) -> None:
        """Сохраняет снимок изменений"""
        try:
            snapshot_data = {
                'planning_id': planning_id,
                'old_heure': result.get('old_time'),
                'new_heure': result.get('new_time'),
                'old_medecins_id': result.get('old_doctor'),
                'new_medecins_id': result.get('new_doctor'),
                'event_type': event_type.value,
                'delay_minutes': result.get('delay_minutes')
            }
            
            self.dal.save_snapshot(snapshot_data)
            
        except Exception as e:
            logger.error(f"Ошибка сохранения снимка: {e}")

    def _send_websocket_notification(self, planning_id: int, data: Dict[str, Any]) -> None:
        """Отправляет WebSocket уведомление (заглушка)"""
        try:
            # В реальной реализации здесь была бы отправка через WebSocket
            logger.info(f"WebSocket уведомление для planning_id {planning_id}: {data}")
            
        except Exception as e:
            logger.error(f"Ошибка отправки WebSocket уведомления: {e}")

    def optimize_waiting_queue(self, doctor_id: int) -> Dict[str, Any]:
        """Оптимизирует очередь ожидания для врача"""
        try:
            waiting_patients = self.dal.get_waiting_patients()
            doctor_patients = [p for p in waiting_patients 
                             if p.get('doctor_id') == doctor_id]
            
            if not doctor_patients:
                return {'reordered': 0, 'queue': []}
            
            # Сортируем по приоритету и времени записи
            doctor_patients.sort(key=lambda p: (-p['priority'], p['heure']))
            
            # Перепланируем время
            current_time = datetime.utcnow()
            slot_duration = 30  # минуты
            
            reordered = 0
            new_queue = []
            
            for i, patient in enumerate(doctor_patients):
                new_time = current_time + timedelta(minutes=i * slot_duration)
                old_time = patient['heure']
                
                if new_time != old_time:
                    self.dal.update_planning_time(patient['planning_id'], new_time)
                    reordered += 1
                
                new_queue.append({
                    'planning_id': patient['planning_id'],
                    'priority': patient['priority'],
                    'old_time': old_time,
                    'new_time': new_time
                })
            
            return {
                'doctor_id': doctor_id,
                'reordered': reordered,
                'queue': new_queue
            }
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации очереди: {e}")
            return {'reordered': 0, 'queue': []} 