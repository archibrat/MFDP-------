import asyncio
import json
import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import aio_pika
from aio_pika import Message, DeliveryMode
from database.config import get_settings
from services.logging.logging import get_logger
from exceptions import MLServiceException

settings = get_settings()
logger = get_logger(__name__)

class RabbitMQClient:
    """RabbitMQ клиент для асинхронных ML задач"""
    
    def __init__(self):
        self.connection_url = getattr(settings, 'RABBITMQ_URL', 'amqp://guest:guest@rabbitmq:5672/')
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.callback_queue: Optional[aio_pika.Queue] = None
        self.futures: Dict[str, asyncio.Future] = {}
        
        # Настройки очередей
        self.ml_queue_name = getattr(settings, 'ML_QUEUE_NAME', 'ml_tasks')
        self.result_queue_name = getattr(settings, 'ML_RESULT_QUEUE_NAME', 'ml_results')
        self.rpc_timeout = getattr(settings, 'RPC_TIMEOUT', 300)  # 5 минут
    
    async def connect(self):
        """Установка соединения с RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(self.connection_url)
            self.channel = await self.connection.channel()
            
            # Объявляем очереди
            await self.channel.declare_queue(
                self.ml_queue_name, 
                durable=True
            )
            
            # Создаем callback очередь для RPC
            self.callback_queue = await self.channel.declare_queue(
                exclusive=True
            )
            
            # Начинаем слушать ответы
            await self.callback_queue.consume(self._on_response)
            
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise MLServiceException(f"RabbitMQ connection error: {str(e)}")
    
    async def disconnect(self):
        """Закрытие соединения с RabbitMQ"""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
    
    async def _on_response(self, message: aio_pika.IncomingMessage):
        """Обработчик ответов RPC"""
        async with message.process():
            correlation_id = message.correlation_id
            if correlation_id in self.futures:
                future = self.futures.pop(correlation_id)
                try:
                    result = json.loads(message.body.decode())
                    future.set_result(result)
                except json.JSONDecodeError as e:
                    future.set_exception(MLServiceException(f"Invalid JSON response: {str(e)}"))
                except Exception as e:
                    future.set_exception(MLServiceException(f"Response processing error: {str(e)}"))
    
    async def _ensure_connected(self):
        """Проверка и восстановление соединения"""
        if not self.connection or self.connection.is_closed:
            await self.connect()
    
    async def send_task(
        self, 
        task_type: str, 
        task_data: Dict[str, Any], 
        priority: int = 1,
        wait_for_result: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Отправка задачи в очередь
        
        Аргументы:
            task_type: Тип задачи
            task_data: Данные задачи
            priority: Приоритет задачи (1-10)
            wait_for_result: Ждать результат (RPC режим)
            
        Возвращает:
            Optional[Dict[str, Any]]: Результат задачи (если wait_for_result=True)
        """
        await self._ensure_connected()
        
        task_id = str(uuid.uuid4())
        
        message_body = {
            "task_id": task_id,
            "task_type": task_type,
            "task_data": task_data,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": priority
        }
        
        message = Message(
            json.dumps(message_body).encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            priority=priority,
            message_id=task_id,
            timestamp=datetime.utcnow()
        )
        
        if wait_for_result:
            # RPC режим
            correlation_id = str(uuid.uuid4())
            message.correlation_id = correlation_id
            message.reply_to = self.callback_queue.name
            
            # Создаем future для ожидания результата
            future = asyncio.Future()
            self.futures[correlation_id] = future
            
            try:
                # Отправляем сообщение
                await self.channel.default_exchange.publish(
                    message,
                    routing_key=self.ml_queue_name
                )
                
                logger.info(f"Sent RPC task {task_type} with ID: {task_id}")
                
                # Ждем результат с таймаутом
                result = await asyncio.wait_for(future, timeout=self.rpc_timeout)
                logger.info(f"Received result for task {task_id}")
                return result
                
            except asyncio.TimeoutError:
                self.futures.pop(correlation_id, None)
                raise MLServiceException(f"Task {task_id} timed out after {self.rpc_timeout} seconds")
            except Exception as e:
                self.futures.pop(correlation_id, None)
                raise MLServiceException(f"RPC task error: {str(e)}")
        else:
            # Fire-and-forget режим
            await self.channel.default_exchange.publish(
                message,
                routing_key=self.ml_queue_name
            )
            
            logger.info(f"Sent async task {task_type} with ID: {task_id}")
            return {"task_id": task_id, "status": "queued"}
    
    async def send_prediction_task(
        self, 
        input_data: Dict[str, Any], 
        model_version: str = "latest",
        include_explanation: bool = True,
        wait_for_result: bool = True
    ) -> Dict[str, Any]:
        """
        Отправка задачи предсказания
        
        Аргументы:
            input_data: Входные данные
            model_version: Версия модели
            include_explanation: Включить объяснение
            wait_for_result: Ждать результат
            
        Возвращает:
            Dict[str, Any]: Результат предсказания
        """
        task_data = {
            "input_data": input_data,
            "model_version": model_version,
            "include_explanation": include_explanation
        }
        
        return await self.send_task(
            task_type="prediction",
            task_data=task_data,
            priority=5,
            wait_for_result=wait_for_result
        )
    
    async def send_batch_prediction_task(
        self, 
        batch_data: list[Dict[str, Any]], 
        model_version: str = "latest",
        wait_for_result: bool = False
    ) -> Dict[str, Any]:
        """
        Отправка задачи пакетного предсказания
        
        Аргументы:
            batch_data: Список входных данных
            model_version: Версия модели
            wait_for_result: Ждать результат
            
        Возвращает:
            Dict[str, Any]: Информация о задаче
        """
        task_data = {
            "batch_data": batch_data,
            "model_version": model_version
        }
        
        return await self.send_task(
            task_type="batch_prediction",
            task_data=task_data,
            priority=3,
            wait_for_result=wait_for_result
        )
    
    async def send_training_task(
        self, 
        training_data: Dict[str, Any], 
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Отправка задачи обучения модели
        
        Аргументы:
            training_data: Данные для обучения
            model_config: Конфигурация модели
            
        Возвращает:
            Dict[str, Any]: Информация о задаче обучения
        """
        task_data = {
            "training_data": training_data,
            "model_config": model_config
        }
        
        return await self.send_task(
            task_type="training",
            task_data=task_data,
            priority=2,
            wait_for_result=False
        )
    
    async def send_evaluation_task(
        self, 
        test_data: Dict[str, Any], 
        model_version: str = "latest"
    ) -> Dict[str, Any]:
        """
        Отправка задачи оценки модели
        
        Аргументы:
            test_data: Тестовые данные
            model_version: Версия модели
            
        Возвращает:
            Dict[str, Any]: Информация о задаче оценки
        """
        task_data = {
            "test_data": test_data,
            "model_version": model_version
        }
        
        return await self.send_task(
            task_type="evaluation",
            task_data=task_data,
            priority=4,
            wait_for_result=False
        )
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Получение статистики очередей
        
        Возвращает:
            Dict[str, Any]: Статистика очередей
        """
        await self._ensure_connected()
        
        try:
            queue = await self.channel.declare_queue(
                self.ml_queue_name, 
                durable=True, 
                passive=True
            )
            
            return {
                "queue_name": self.ml_queue_name,
                "message_count": queue.declaration_result.message_count,
                "consumer_count": queue.declaration_result.consumer_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get queue stats: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Глобальный экземпляр клиента
rabbitmq_client = RabbitMQClient()