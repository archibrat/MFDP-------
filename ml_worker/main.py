import asyncio
import signal
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import typer
import logging
from contextlib import asynccontextmanager

from settings import MLWorkerSettings, setup_logging
from rmq_client import RMQClient, RMQSettings
from ml_processor import MLProcessor
from health_monitor import HealthMonitor

# Создаем Typer приложение
app = typer.Typer(
    name="ml-worker",
    help="ML Worker для обработки задач машинного обучения",
    add_completion=False
)

logger = logging.getLogger(__name__)

class MLWorker:
    """Основной класс ML Worker"""
    
    def __init__(self, settings: MLWorkerSettings):
        self.settings = settings
        self.rmq_client = RMQClient(RMQSettings(
            rabbitmq_url=settings.rabbitmq_url,
            ml_queue_name=settings.ml_queue_name,
            result_queue_name=settings.result_queue_name
        ))
        self.ml_processor = MLProcessor(settings)
        self.health_monitor = HealthMonitor(settings)
        self._shutdown_event = asyncio.Event()
        self._tasks = []
    
    async def start(self):
        """Запуск ML Worker"""
        logger.info(f"Starting ML Worker: {self.settings.worker_name}")
        
        try:
            # Подключаемся к RabbitMQ
            await self.rmq_client.connect()
            
            # Инициализируем ML процессор
            await self.ml_processor.initialize()
            
            # Регистрируем обработчик задач
            self.rmq_client.register_handler(
                self.settings.ml_queue_name, 
                self.ml_processor.process_task
            )
            
            # Запускаем потребление задач
            await self.rmq_client.consume_tasks(self.settings.ml_queue_name)
            
            # Запускаем мониторинг здоровья
            if self.settings.metrics_enabled:
                health_task = asyncio.create_task(self.health_monitor.start())
                self._tasks.append(health_task)
            
            logger.info("ML Worker started successfully")
            
            # Ждем сигнала завершения
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error starting ML Worker: {str(e)}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Завершение работы ML Worker"""
        logger.info("Shutting down ML Worker...")
        
        # Отменяем все задачи
        for task in self._tasks:
            task.cancel()
        
        # Ждем завершения задач
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Закрываем соединения
        await self.rmq_client.disconnect()
        await self.ml_processor.cleanup()
        
        logger.info("ML Worker shutdown complete")
    
    def handle_signal(self, signum, frame):
        """Обработчик сигналов завершения"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_event.set()

@app.command()
def run(
    config_file: Optional[Path] = typer.Option(
        None, 
        "--config", 
        "-c",
        help="Путь к файлу конфигурации"
    ),
    worker_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n", 
        help="Имя воркера"
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Уровень логирования (DEBUG, INFO, WARNING, ERROR)"
    ),
    rabbitmq_url: Optional[str] = typer.Option(
        None,
        "--rabbitmq-url",
        "-r",
        help="URL для подключения к RabbitMQ"
    ),
    max_concurrent: Optional[int] = typer.Option(
        None,
        "--max-concurrent",
        "-m",
        help="Максимальное количество одновременных задач"
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu/--no-gpu",
        help="Использовать GPU для обработки"
    )
):
    """Запуск ML Worker"""
    
    # Загружаем настройки
    if config_file and config_file.exists():
        # Можно добавить загрузку из YAML/JSON файла
        pass
    
    settings = MLWorkerSettings()
    
    # Переопределяем настройки из CLI аргументов
    if worker_name:
        settings.worker_name = worker_name
    if log_level:
        settings.logging.log_level = log_level
    if rabbitmq_url:
        settings.rabbitmq_url = rabbitmq_url
    if max_concurrent:
        settings.max_concurrent_tasks = max_concurrent
    if gpu:
        settings.gpu_enabled = gpu
    
    # Настраиваем логирование
    setup_logging(settings.logging)
    
    # Создаем и запускаем воркер
    worker = MLWorker(settings)
    
    # Настраиваем обработчики сигналов
    signal.signal(signal.SIGINT, worker.handle_signal)
    signal.signal(signal.SIGTERM, worker.handle_signal)
    
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}")
        sys.exit(1)

@app.command()
def health():
    """Проверка состояния ML Worker"""
    async def check_health():
        settings = MLWorkerSettings()
        rmq_client = RMQClient(RMQSettings(rabbitmq_url=settings.rabbitmq_url))
        
        try:
            await rmq_client.connect()
            health_info = await rmq_client.health_check()
            
            typer.echo("ML Worker Health Check:")
            typer.echo(f"Status: {health_info.get('status', 'unknown')}")
            
            if 'queues' in health_info:
                typer.echo("Queues:")
                for queue_name, queue_info in health_info['queues'].items():
                    typer.echo(f"  {queue_name}: {queue_info.get('message_count', 0)} messages")
            
            if health_info.get('status') == 'healthy':
                typer.echo("ML Worker is healthy")
            else:
                typer.echo("ML Worker has issues")
                sys.exit(1)
                
        except Exception as e:
            typer.echo(f"Health check failed: {str(e)}")
            sys.exit(1)
        finally:
            await rmq_client.disconnect()
    
    asyncio.run(check_health())

@app.command()
def queue_stats():
    """Статистика очередей"""
    async def get_stats():
        settings = MLWorkerSettings()
        rmq_client = RMQClient(RMQSettings(rabbitmq_url=settings.rabbitmq_url))
        
        try:
            await rmq_client.connect()
            
            ml_queue_info = await rmq_client.get_queue_info(settings.ml_queue_name)
            result_queue_info = await rmq_client.get_queue_info(settings.result_queue_name)
            
            typer.echo("Queue Statistics:")
            typer.echo(f"ML Tasks Queue: {ml_queue_info.get('message_count', 0)} messages, {ml_queue_info.get('consumer_count', 0)} consumers")
            typer.echo(f"Results Queue: {result_queue_info.get('message_count', 0)} messages, {result_queue_info.get('consumer_count', 0)} consumers")
            
        except Exception as e:
            typer.echo(f"Failed to get queue stats: {str(e)}")
            sys.exit(1)
        finally:
            await rmq_client.disconnect()
    
    asyncio.run(get_stats())

@app.command()
def test_task(
    task_type: str = typer.Argument(..., help="Тип задачи (prediction, training, evaluation)"),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Файл с входными данными (JSON)")
):
    """Отправка тестовой задачи"""
    import json
    import uuid
    from datetime import datetime
    
    async def send_test_task():
        settings = MLWorkerSettings()
        rmq_client = RMQClient(RMQSettings(rabbitmq_url=settings.rabbitmq_url))
        
        try:
            await rmq_client.connect()
            
            # Подготавливаем тестовые данные
            if input_file and input_file.exists():
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
            else:
                # Тестовые данные по умолчанию в зависимости от типа задачи
                if task_type == "prediction":
                    input_data = {
                        "patient_id": "test_patient_001",
                        "age": 45,
                        "gender": "male",
                        "symptoms": ["fever", "cough", "fatigue"],
                        "medical_history": ["diabetes", "hypertension"],
                        "appointment_date": "2024-01-15T10:00:00Z"
                    }
                elif task_type == "training":
                    input_data = {
                        "dataset_path": "/data/training_dataset.csv",
                        "model_type": "random_forest",
                        "hyperparameters": {
                            "n_estimators": 100,
                            "max_depth": 10
                        },
                        "validation_split": 0.2
                    }
                elif task_type == "evaluation":
                    input_data = {
                        "model_id": "model_001",
                        "test_dataset_path": "/data/test_dataset.csv",
                        "metrics": ["accuracy", "precision", "recall", "f1"]
                    }
                else:
                    typer.echo(f"Неизвестный тип задачи: {task_type}")
                    typer.echo("Доступные типы: prediction, training, evaluation")
                    return
            
            # Создаем сообщение задачи
            task_message = {
                "task_id": str(uuid.uuid4()),
                "task_type": task_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": input_data,
                "priority": "normal",
                "timeout": 300  # 5 минут
            }
            
            # Отправляем задачу в очередь
            await rmq_client.send_task(settings.ml_queue_name, task_message)
            
            typer.echo(f"Тестовая задача отправлена:")
            typer.echo(f"  Task ID: {task_message['task_id']}")
            typer.echo(f"  Type: {task_type}")
            typer.echo(f"  Queue: {settings.ml_queue_name}")
            typer.echo(f"  Data: {json.dumps(input_data, indent=2, ensure_ascii=False)}")
            
        except FileNotFoundError:
            typer.echo(f"Файл не найден: {input_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            typer.echo(f"Ошибка парсинга JSON: {str(e)}")
            sys.exit(1)
        except Exception as e:
            typer.echo(f"Ошибка отправки задачи: {str(e)}")
            sys.exit(1)
        finally:
            await rmq_client.disconnect()
    
    asyncio.run(send_test_task())