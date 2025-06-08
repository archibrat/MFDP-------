from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps
from typing import Callable, Any

# Создание реестра метрик
REGISTRY = CollectorRegistry()

# Метрики для ML-предсказаний
PREDICTION_REQUESTS = Counter(
    'ml_prediction_requests_total',
    'Общее количество запросов на предсказание',
    ['model_version', 'prediction_type', 'status'],
    registry=REGISTRY
)

PREDICTION_DURATION = Histogram(
    'ml_prediction_duration_seconds',
    'Время выполнения предсказания',
    ['model_version', 'prediction_type'],
    registry=REGISTRY
)

MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Текущая точность модели',
    ['model_version'],
    registry=REGISTRY
)

QUEUE_LENGTH = Gauge(
    'ml_queue_length',
    'Длина очереди ML-задач',
    ['queue_name'],
    registry=REGISTRY
)

ACTIVE_WORKERS = Gauge(
    'ml_active_workers',
    'Количество активных ML-воркеров',
    ['worker_type'],
    registry=REGISTRY
)

def track_prediction_metrics(model_version: str, prediction_type: str):
    """Декоратор для отслеживания метрик предсказаний"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                
                PREDICTION_REQUESTS.labels(
                    model_version=model_version,
                    prediction_type=prediction_type,
                    status=status
                ).inc()
                
                PREDICTION_DURATION.labels(
                    model_version=model_version,
                    prediction_type=prediction_type
                ).observe(duration)
        
        return wrapper
    return decorator

class MetricsCollector:
    """Сборщик метрик для системы"""
    
    def __init__(self):
        self.last_update = time.time()
    
    async def collect_queue_metrics(self):
        """Сбор метрик очередей"""
        # Интеграция с RabbitMQ API для получения длины очередей
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'http://rabbitmq:15672/api/queues',
                    auth=aiohttp.BasicAuth('rmuser', 'rmpassword')
                ) as response:
                    queues = await response.json()
                    
                    for queue in queues:
                        QUEUE_LENGTH.labels(
                            queue_name=queue['name']
                        ).set(queue.get('messages_ready', 0))
        except Exception as e:
            print(f"Ошибка сбора метрик очередей: {e}")
    
    async def collect_worker_metrics(self):
        """Сбор метрик воркеров"""
        try:
            import docker
            client = docker.from_env()
            
            # Подсчет активных воркеров по типам
            containers = client.containers.list(
                filters={'name': 'ml_worker'}
            )
            
            worker_counts = {}
            for container in containers:
                worker_type = container.attrs['Config']['Env']
                worker_type = next(
                    (env.split('=')[1] for env in worker_type 
                     if env.startswith('WORKER_TYPE=')), 
                    'unknown'
                )
                worker_counts[worker_type] = worker_counts.get(worker_type, 0) + 1
            
            for worker_type, count in worker_counts.items():
                ACTIVE_WORKERS.labels(worker_type=worker_type).set(count)
                
        except Exception as e:
            print(f"Ошибка сбора метрик воркеров: {e}")

# Инициализация сборщика метрик
metrics_collector = MetricsCollector()
