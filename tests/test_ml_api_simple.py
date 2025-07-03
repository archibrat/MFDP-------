"""
Простая проверка ML Production API
"""

import requests
import json
from datetime import datetime, date, timedelta


def test_ml_api_health():
    """Тест проверки здоровья ML API"""
    try:
        response = requests.get("http://localhost:8081/status")
        print(f"Статус API: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Ошибка: {response.text}")
    except Exception as e:
        print(f"Ошибка подключения: {e}")


def test_no_show_prediction():
    """Тест прогнозирования неявки"""
    try:
        planning_id = 12345
        response = requests.get(f"http://localhost:8081/api/ml/noshowscore", 
                               params={"planning_id": planning_id})
        
        print(f"\nПрогноз неявки для записи {planning_id}:")
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Ошибка: {response.text}")
    except Exception as e:
        print(f"Ошибка: {e}")


def test_load_balance():
    """Тест балансировки нагрузки"""
    try:
        data = {
            "start_date": date.today().isoformat(),
            "end_date": (date.today() + timedelta(days=7)).isoformat(),
            "department_ids": [1, 2, 3],
            "target_utilization": 0.8
        }
        
        response = requests.post("http://localhost:8081/api/ml/load-balance", 
                               json=data)
        
        print(f"\nБалансировка нагрузки:")
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Ответ: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"Ошибка: {response.text}")
    except Exception as e:
        print(f"Ошибка: {e}")


def test_schedule_event():
    """Тест обработки события планирования"""
    try:
        data = {
            "event_type": "ARRIVE_DATE",
            "planning_id": 12345,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"arrival_time": datetime.utcnow().isoformat()}
        }
        
        response = requests.post("http://localhost:8081/api/ml/schedule-event", 
                               json=data)
        
        print(f"\nОбработка события:")
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Ответ: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"Ошибка: {response.text}")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    print("🧪 Тестирование MFDP ML Production API")
    print("=" * 50)
    
    # Запуск тестов
    test_ml_api_health()
    test_no_show_prediction()
    test_load_balance()
    test_schedule_event()
    
    print("\n✅ Тестирование завершено") 