"""
Модуль оценки рисков неявки пациентов
Поддерживает настраиваемые пороги и динамические рекомендации
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from app.services.no_show_prediction.schemas import RiskLevel


class RiskThresholds:
    """Настраиваемые пороги для оценки рисков"""
    
    # Пороги по умолчанию
    DEFAULT_LOW_RISK = 0.3
    DEFAULT_HIGH_RISK = 0.7
    
    def __init__(self, low_risk: float = DEFAULT_LOW_RISK, 
                 high_risk: float = DEFAULT_HIGH_RISK):
        """
        Инициализация порогов риска
        
        Args:
            low_risk: Порог для низкого риска (0-1)
            high_risk: Порог для высокого риска (0-1)
        """
        if not (0 <= low_risk <= high_risk <= 1):
            raise ValueError("Пороги должны быть в диапазоне [0, 1] и low_risk <= high_risk")
            
        self.low_risk = low_risk
        self.high_risk = high_risk


class RecommendationType(str, Enum):
    """Типы рекомендаций"""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class Recommendation:
    """Рекомендация для снижения риска неявки"""
    type: RecommendationType
    title: str
    description: str
    priority: int  # 1 - высший приоритет
    cost_level: str  # "low", "medium", "high"
    expected_effectiveness: float  # 0-1, ожидаемая эффективность


class RiskAssessment:
    """
    Класс для оценки уровня риска неявки
    Поддерживает настраиваемые пороги и динамические рекомендации
    """

    def __init__(self, thresholds: Optional[RiskThresholds] = None):
        """
        Инициализация оценщика рисков
        
        Args:
            thresholds: Настраиваемые пороги риска
        """
        self.thresholds = thresholds or RiskThresholds()
        self.logger = logging.getLogger(__name__)
        self._recommendations = self._initialize_recommendations()

    def assess_risk(self, probability: float) -> RiskLevel:
        """
        Определение уровня риска на основе вероятности
        
        Args:
            probability: Вероятность неявки (0-1)
            
        Returns:
            Уровень риска
        """
        if probability < self.thresholds.low_risk:
            return RiskLevel.LOW
        elif probability < self.thresholds.high_risk:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def get_recommendation(self, risk_level: RiskLevel, 
                          patient_context: Optional[Dict] = None) -> str:
        """
        Получение рекомендации на основе уровня риска
        
        Args:
            risk_level: Уровень риска
            patient_context: Дополнительный контекст пациента
            
        Returns:
            Текст рекомендации
        """
        recommendations = self._get_recommendations_for_risk(risk_level, patient_context)
        
        if not recommendations:
            return "Стандартные действия"
            
        # Выбираем рекомендацию с наивысшим приоритетом
        best_recommendation = max(recommendations, key=lambda r: r.priority)
        
        return f"{best_recommendation.title}: {best_recommendation.description}"

    def get_detailed_recommendations(self, risk_level: RiskLevel,
                                   patient_context: Optional[Dict] = None) -> List[Recommendation]:
        """
        Получение детальных рекомендаций для уровня риска
        
        Args:
            risk_level: Уровень риска
            patient_context: Дополнительный контекст пациента
            
        Returns:
            Список рекомендаций, отсортированный по приоритету
        """
        recommendations = self._get_recommendations_for_risk(risk_level, patient_context)
        return sorted(recommendations, key=lambda r: r.priority, reverse=True)

    def calculate_risk_score(self, probability: float, 
                           patient_context: Optional[Dict] = None) -> Dict:
        """
        Расчет детального риска с дополнительными метриками
        
        Args:
            probability: Вероятность неявки
            patient_context: Контекст пациента
            
        Returns:
            Словарь с детальной информацией о риске
        """
        risk_level = self.assess_risk(probability)
        recommendations = self.get_detailed_recommendations(risk_level, patient_context)
        
        # Расчет дополнительных метрик
        risk_score = self._calculate_risk_score(probability, patient_context)
        
        return {
            'probability': probability,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'recommendations': recommendations,
            'urgency_level': self._calculate_urgency_level(probability, patient_context)
        }

    def _initialize_recommendations(self) -> Dict[RiskLevel, List[Recommendation]]:
        """Инициализация рекомендаций по уровням риска"""
        return {
            RiskLevel.LOW: [
                Recommendation(
                    type=RecommendationType.STANDARD,
                    title="Стандартное уведомление",
                    description="SMS-уведомление за день до приема",
                    priority=1,
                    cost_level="low",
                    expected_effectiveness=0.3
                )
            ],
            RiskLevel.MEDIUM: [
                Recommendation(
                    type=RecommendationType.ENHANCED,
                    title="Дополнительное уведомление",
                    description="SMS-уведомление за 2 часа до приема + звонок",
                    priority=2,
                    cost_level="medium",
                    expected_effectiveness=0.6
                ),
                Recommendation(
                    type=RecommendationType.STANDARD,
                    title="Стандартное уведомление",
                    description="SMS-уведомление за день до приема",
                    priority=1,
                    cost_level="low",
                    expected_effectiveness=0.3
                )
            ],
            RiskLevel.HIGH: [
                Recommendation(
                    type=RecommendationType.AGGRESSIVE,
                    title="Агрессивное напоминание",
                    description="Телефонный звонок для подтверждения + резервирование времени",
                    priority=3,
                    cost_level="high",
                    expected_effectiveness=0.8
                ),
                Recommendation(
                    type=RecommendationType.ENHANCED,
                    title="Дополнительное уведомление",
                    description="SMS-уведомление за 2 часа до приема + звонок",
                    priority=2,
                    cost_level="medium",
                    expected_effectiveness=0.6
                ),
                Recommendation(
                    type=RecommendationType.STANDARD,
                    title="Стандартное уведомление",
                    description="SMS-уведомление за день до приема",
                    priority=1,
                    cost_level="low",
                    expected_effectiveness=0.3
                )
            ]
        }

    def _get_recommendations_for_risk(self, risk_level: RiskLevel,
                                    patient_context: Optional[Dict] = None) -> List[Recommendation]:
        """Получение рекомендаций для уровня риска с учетом контекста"""
        base_recommendations = self._recommendations.get(risk_level, [])
        
        if not patient_context:
            return base_recommendations
            
        # Адаптация рекомендаций под контекст пациента
        adapted_recommendations = []
        
        for rec in base_recommendations:
            # Проверяем, подходит ли рекомендация для данного пациента
            if self._is_recommendation_suitable(rec, patient_context):
                adapted_recommendations.append(rec)
                
        return adapted_recommendations or base_recommendations

    def _is_recommendation_suitable(self, recommendation: Recommendation,
                                  patient_context: Dict) -> bool:
        """Проверка применимости рекомендации к пациенту"""
        # Проверяем наличие телефона для SMS/звонков
        if "phone_confirmed" in patient_context and not patient_context["phone_confirmed"]:
            if "SMS" in recommendation.description or "звонок" in recommendation.description:
                return False
                
        # Проверяем возраст для определенных рекомендаций
        if "age" in patient_context:
            age = patient_context["age"]
            if age < 18 and "агрессивное" in recommendation.description.lower():
                return False
                
        return True

    def _calculate_risk_score(self, probability: float,
                            patient_context: Optional[Dict] = None) -> float:
        """Расчет дополнительного риска на основе контекста"""
        base_score = probability
        
        if not patient_context:
            return base_score
            
        # Модификаторы риска
        modifiers = 0.0
        
        # История неявок
        if "historical_no_show_rate" in patient_context:
            history_rate = patient_context["historical_no_show_rate"]
            modifiers += history_rate * 0.2
            
        # Заблаговременность записи
        if "advance_booking_days" in patient_context:
            booking_days = patient_context["advance_booking_days"]
            if booking_days > 30:
                modifiers += 0.1
            elif booking_days < 7:
                modifiers -= 0.05
                
        # Время приема
        if "appointment_hour" in patient_context:
            hour = patient_context["appointment_hour"]
            if hour < 9 or hour > 17:
                modifiers += 0.1
                
        return min(1.0, max(0.0, base_score + modifiers))

    def _calculate_urgency_level(self, probability: float,
                               patient_context: Optional[Dict] = None) -> str:
        """Расчет уровня срочности"""
        if probability > 0.8:
            return "critical"
        elif probability > 0.6:
            return "high"
        elif probability > 0.4:
            return "medium"
        else:
            return "low" 