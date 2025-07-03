"""
Load Balancer с OR-Tools оптимизацией
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

try:
    from ortools.linear_solver import pywraplp
except ImportError:
    pywraplp = None

from app.services.ml_production.dal import MLDataAccessLayer


logger = logging.getLogger(__name__)


class DoctorLoadOptimizer:
    """Оптимизатор нагрузки врачей с использованием OR-Tools"""
    
    def __init__(self, dal: MLDataAccessLayer):
        self.dal = dal
        
        if pywraplp is None:
            logger.warning("OR-Tools не установлен, используется простая эвристика")

    def optimize_load_distribution(self, start_date: datetime, end_date: datetime, 
                                 department_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Оптимизирует распределение нагрузки врачей"""
        try:
            # Получение данных
            doctor_profiles = self.dal.get_doctor_profiles(department_ids)
            schedule_data = self.dal.get_schedule_data(start_date, end_date)
            
            if not doctor_profiles or not schedule_data:
                return {'optimized_assignments': {}, 'metrics': {}, 'recommendations': []}
            
            # Оптимизация
            if pywraplp is not None:
                assignments = self._optimize_with_ortools(doctor_profiles, schedule_data)
            else:
                assignments = self._optimize_with_heuristic(doctor_profiles, schedule_data)
            
            # Расчет метрик
            metrics = self._calculate_metrics(doctor_profiles, schedule_data, assignments)
            
            # Генерация рекомендаций
            recommendations = self._generate_recommendations(doctor_profiles, metrics)
            
            # Применение изменений
            if assignments:
                self.dal.call_sp_reschedule_batch(assignments)
            
            return {
                'optimized_assignments': assignments,
                'metrics': metrics,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации нагрузки: {e}")
            return {'optimized_assignments': {}, 'metrics': {}, 'recommendations': []}

    def _optimize_with_ortools(self, doctor_profiles: List[Dict], 
                             schedule_data: List[Dict]) -> Dict[int, int]:
        """Оптимизация с использованием OR-Tools"""
        try:
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                return self._optimize_with_heuristic(doctor_profiles, schedule_data)
            
            # Данные
            doctors = {d['medecins_id']: d for d in doctor_profiles}
            patients = [s for s in schedule_data if s['status'] == 'scheduled']
            
            if not patients:
                return {}
            
            # Переменные: x[p,d] = 1 если пациент p назначен врачу d
            x = {}
            for patient in patients:
                for doctor_id in doctors.keys():
                    # Проверка совместимости специализации
                    if self._is_compatible(patient, doctors[doctor_id]):
                        x[(patient['planning_id'], doctor_id)] = solver.IntVar(0, 1, 
                                                                              f'x_{patient["planning_id"]}_{doctor_id}')
            
            # Ограничения: каждый пациент назначен одному врачу
            for patient in patients:
                compatible_assignments = [x[(patient['planning_id'], d_id)] 
                                        for d_id in doctors.keys() 
                                        if (patient['planning_id'], d_id) in x]
                if compatible_assignments:
                    solver.Add(sum(compatible_assignments) == 1)
            
            # Ограничения: дневная норма врачей
            for doctor_id, doctor in doctors.items():
                daily_assignments = [x[(p['planning_id'], doctor_id)] 
                                   for p in patients 
                                   if (p['planning_id'], doctor_id) in x]
                if daily_assignments:
                    solver.Add(sum(daily_assignments) <= doctor['daily_norm'])
            
            # Целевая функция: минимизация отклонений от целевой загрузки 0.8
            objective = solver.Objective()
            
            for doctor_id, doctor in doctors.items():
                assignments = [x[(p['planning_id'], doctor_id)] 
                             for p in patients 
                             if (p['planning_id'], doctor_id) in x]
                
                if assignments:
                    # Загрузка врача
                    load = sum(assignments) / doctor['daily_norm']
                    
                    # Переменная для отклонения от 0.8
                    deviation = solver.NumVar(0, 1, f'dev_{doctor_id}')
                    solver.Add(deviation >= load - 0.8)
                    solver.Add(deviation >= 0.8 - load)
                    
                    objective.SetCoefficient(deviation, 1.0)
            
            objective.SetMinimization()
            
            # Решение
            status = solver.Solve()
            
            if status == pywraplp.Solver.OPTIMAL:
                assignments = {}
                for (planning_id, doctor_id), var in x.items():
                    if var.solution_value() > 0.5:
                        assignments[planning_id] = doctor_id
                        
                logger.info(f"OR-Tools оптимизация: {len(assignments)} переназначений")
                return assignments
            else:
                logger.warning("OR-Tools не нашел оптимального решения")
                return self._optimize_with_heuristic(doctor_profiles, schedule_data)
                
        except Exception as e:
            logger.error(f"Ошибка OR-Tools оптимизации: {e}")
            return self._optimize_with_heuristic(doctor_profiles, schedule_data)

    def _optimize_with_heuristic(self, doctor_profiles: List[Dict], 
                               schedule_data: List[Dict]) -> Dict[int, int]:
        """Простая эвристическая оптимизация"""
        try:
            doctors = {d['medecins_id']: d for d in doctor_profiles}
            patients = [s for s in schedule_data if s['status'] == 'scheduled']
            
            assignments = {}
            doctor_loads = {d_id: 0 for d_id in doctors.keys()}
            
            # Сортируем пациентов по приоритету (CITO, VIP)
            patients.sort(key=lambda p: (
                -int(p.get('cito') == 'Y'),
                -int(p.get('vip_groups_id') is not None),
                p['planning_id']
            ))
            
            for patient in patients:
                # Находим подходящих врачей
                compatible_doctors = [
                    (d_id, doctor) for d_id, doctor in doctors.items()
                    if self._is_compatible(patient, doctor) and 
                       doctor_loads[d_id] < doctor['daily_norm']
                ]
                
                if compatible_doctors:
                    # Выбираем врача с наименьшей загрузкой
                    best_doctor = min(compatible_doctors, 
                                    key=lambda x: doctor_loads[x[0]] / x[1]['daily_norm'])
                    
                    doctor_id = best_doctor[0]
                    
                    # Переназначаем только если текущий врач перегружен
                    current_doctor_id = patient.get('medecins_id')
                    if current_doctor_id in doctors:
                        current_load = doctor_loads.get(current_doctor_id, 0) / doctors[current_doctor_id]['daily_norm']
                        new_load = doctor_loads[doctor_id] / doctors[doctor_id]['daily_norm']
                        
                        if current_load > 0.9 and new_load < 0.8:
                            assignments[patient['planning_id']] = doctor_id
                    
                    doctor_loads[doctor_id] += 1
            
            logger.info(f"Эвристическая оптимизация: {len(assignments)} переназначений")
            return assignments
            
        except Exception as e:
            logger.error(f"Ошибка эвристической оптимизации: {e}")
            return {}


    def _calculate_metrics(self, doctor_profiles: List[Dict], 
                         schedule_data: List[Dict], 
                         assignments: Dict[int, int]) -> Dict[str, float]:
        """Рассчитывает метрики оптимизации"""
        try:
            doctors = {d['medecins_id']: d for d in doctor_profiles}
            
            # Подсчет загрузки врачей после оптимизации
            doctor_loads = {d_id: 0 for d_id in doctors.keys()}
            
            for schedule_item in schedule_data:
                if schedule_item['status'] == 'scheduled':
                    planning_id = schedule_item['planning_id']
                    doctor_id = assignments.get(planning_id, schedule_item['medecins_id'])
                    
                    if doctor_id in doctor_loads:
                        doctor_loads[doctor_id] += 1
            
            # Расчет метрик
            load_ratios = []
            for doctor_id, load in doctor_loads.items():
                if doctor_id in doctors:
                    ratio = load / doctors[doctor_id]['daily_norm']
                    load_ratios.append(ratio)
            
            if load_ratios:
                avg_load = np.mean(load_ratios)
                load_variance = np.var(load_ratios)
                overloaded_count = sum(1 for ratio in load_ratios if ratio > 0.9)
                underloaded_count = sum(1 for ratio in load_ratios if ratio < 0.5)
            else:
                avg_load = 0.0
                load_variance = 0.0
                overloaded_count = 0
                underloaded_count = 0
            
            return {
                'average_load': avg_load,
                'load_variance': load_variance,
                'overloaded_doctors': overloaded_count,
                'underloaded_doctors': underloaded_count,
                'total_reassignments': len(assignments)
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик: {e}")
            return {}

    def _generate_recommendations(self, doctor_profiles: List[Dict], 
                                metrics: Dict[str, float]) -> List[str]:
        """Генерирует рекомендации по оптимизации"""
        recommendations = []
        
        overloaded = metrics.get('overloaded_doctors', 0)
        underloaded = metrics.get('underloaded_doctors', 0)
        avg_load = metrics.get('average_load', 0)
        
        if overloaded > 0:
            recommendations.append(f"Выявлено {overloaded} перегруженных врачей - требуется перераспределение")
        
        if underloaded > 0:
            recommendations.append(f"Выявлено {underloaded} недозагруженных врачей - можно увеличить нагрузку")
        
        if avg_load < 0.6:
            recommendations.append("Средняя загрузка низкая - рассмотреть привлечение дополнительных пациентов")
        elif avg_load > 0.85:
            recommendations.append("Средняя загрузка высокая - рассмотреть расширение штата")
        
        if metrics.get('load_variance', 0) > 0.1:
            recommendations.append("Высокая неравномерность нагрузки - требуется балансировка")
        
        if not recommendations:
            recommendations.append("Оптимальное распределение нагрузки")
        
        return recommendations

    def generate_nightly_report(self, department_ids: Optional[List[int]] = None) -> None:
        """Генерирует ночной отчет для отделений"""
        try:
            start_date = datetime.utcnow().date()
            end_date = start_date + timedelta(days=14)
            
            # Оптимизация на 14 дней
            result = self.optimize_load_distribution(
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.min.time()),
                department_ids
            )
            
            # Сохранение статистики по отделениям
            for dep_id in (department_ids or [1]):  # Заглушка для отделений
                stats = {
                    'fm_dep_id': dep_id,
                    'date': datetime.utcnow(),
                    'avg_load': result['metrics'].get('average_load', 0),
                    'cancel_rate': 0.05,  # Заглушка
                    'no_slots_rate': result['metrics'].get('overloaded_doctors', 0) / 10.0
                }
                
                self.dal.save_load_stats(stats)
            
            logger.info(f"Ночной отчет сгенерирован: {result['metrics']}")
            
        except Exception as e:
            logger.error(f"Ошибка генерации ночного отчета: {e}") 