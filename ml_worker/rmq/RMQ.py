import asyncio
import json
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import aio_pika
from aio_pika import Message, DeliveryMode, Connection, Channel, Queue
from pydantic import BaseSettings, Field
import logging

logger = logging.getLogger(__name__)

class RMQSettings(BaseSettings):
    """Настройки RabbitMQ"""
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672/", env="RABBITMQ_URL")
    ml_queue_name: str = Field(default="ml_tasks", env="ML_QUEUE_NAME")
    result_queue_name: str = Field(default="ml_results", env="ML_RESULT_QUEUE_NAME")
    rpc_timeout: int = Field(default=300, env="RPC_TIMEOUT")
    max_retries: int = Field(default=3, env="RMQ_MAX_RETRIES")
    
    class Config:
        env_file = ".env"

class RMQClient:
    """Объединенный RabbitMQ клиент для ML Worker"""
    
    def __init__(self, settings: Optional[RMQSettings] = None):
        self.settings = settings or RMQSettings()
        self.connection: Optional[Connection] = None
        self.channel: Optional[Channel] = None
        self.callback_queue: Optional[Queue] = None
        self.futures: Dict[str, asyncio.Future] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self._consuming = False
    
    async def connect(self):
        """Установка соединения с RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(self.settings.rabbitmq_url)
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)
            
            # Объявляем основные очереди
            await self.channel.declare_queue(self.settings.ml_queue_name, durable=True)
            await self.channel.declare_queue(self.settings.result_queue_name, durable=True)
            
            # Создаем callback очередь для RPC
            self.callback_queue = await self.channel.declare_queue(exclusive=True)
            await self.callback_queue.consume(self._on_rpc_response)
            
            logger.info(f"Connected to RabbitMQ: {self.settings.rabbitmq_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise
    
    async def disconnect(self):
        """Закрытие соединения"""
        self._consuming = False
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
    
    async def _on_rpc_response(self, message: aio_pika.IncomingMessage):
        """Обработчик RPC ответов"""
        async with message.process():
            correlation_id = message.correlation_id
            if correlation_id in self.futures:
                future = self.futures.pop(correlation_id)
                try:
                    result = json.loads(message.body.decode())
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
    
    async def send_task(
        self, 
        task_data: Dict[str, Any], 
        queue_name: Optional[str] = None,
        priority: int = 1,
        wait_for_result: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Отправка задачи в очередь"""
        queue_name = queue_name or self.settings.ml_queue_name
        task_id = str(uuid.uuid4())
        
        message_body = {
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
            **task_data
        }
        
        message = Message(
            json.dumps(message_body).encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            priority=priority,
            message_id=task_id
        )
        
        if wait_for_result:
            # RPC режим
            correlation_id = str(uuid.uuid4())
            message.correlation_id = correlation_id
            message.reply_to = self.callback_queue.name
            
            future = asyncio.Future()
            self.futures[correlation_id] = future
            
            try:
                await self.channel.default_exchange.publish(message, routing_key=queue_name)
                result = await asyncio.wait_for(future, timeout=self.settings.rpc_timeout)
                return result
            except asyncio.TimeoutError:
                self.futures.pop(correlation_id, None)
                raise TimeoutError(f"Task {task_id} timed out")
        else:
            # Fire-and-forget режим
            await self.channel.default_exchange.publish(message, routing_key=queue_name)
            return {"task_id": task_id, "status": "queued"}
    
    async def consume_tasks(self, queue_name: Optional[str] = None, handler: Optional[Callable] = None):
        """Потребление задач из очереди"""
        queue_name = queue_name or self.settings.ml_queue_name
        queue = await self.channel.declare_queue(queue_name, durable=True)
        
        if handler:
            self.message_handlers[queue_name] = handler
        
        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    task_data = json.loads(message.body.decode())
                    handler_func = self.message_handlers.get(queue_name)
                    
                    if handler_func:
                        result = await handler_func(task_data)
                        
                        # Отправляем результат обратно, если есть reply_to
                        if message.reply_to:
                            response = Message(
                                json.dumps(result).encode(),
                                correlation_id=message.correlation_id
                            )
                            await self.channel.default_exchange.publish(
                                response, 
                                routing_key=message.reply_to
                            )
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    # Отправляем ошибку обратно
                    if message.reply_to:
                        error_response = Message(
                            json.dumps({"error": str(e)}).encode(),
                            correlation_id=message.correlation_id
                        )
                        await self.channel.default_exchange.publish(
                            error_response, 
                            routing_key=message.reply_to
                        )
        
        await queue.consume(process_message)
        self._consuming = True
        logger.info(f"Started consuming from queue: {queue_name}")
    
    async def publish_result(self, result_data: Dict[str, Any], queue_name: Optional[str] = None):
        """Публикация результата"""
        queue_name = queue_name or self.settings.result_queue_name
        
        message = Message(
            json.dumps(result_data).encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            timestamp=datetime.utcnow()
        )
        
        await self.channel.default_exchange.publish(message, routing_key=queue_name)
    
    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """Получение информации об очереди"""
        try:
            queue = await self.channel.declare_queue(queue_name, durable=True, passive=True)
            return {
                "name": queue_name,
                "message_count": queue.declaration_result.message_count,
                "consumer_count": queue.declaration_result.consumer_count
            }
        except Exception as e:
            return {"error": str(e)}
    
    def register_handler(self, queue_name: str, handler: Callable):
        """Регистрация обработчика для очереди"""
        self.message_handlers[queue_name] = handler
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния соединения"""
        try:
            if not self.connection or self.connection.is_closed:
                return {"status": "disconnected"}
            
            # Проверяем основные очереди
            ml_queue_info = await self.get_queue_info(self.settings.ml_queue_name)
            result_queue_info = await self.get_queue_info(self.settings.result_queue_name)
            
            return {
                "status": "healthy",
                "queues": {
                    "ml_tasks": ml_queue_info,
                    "ml_results": result_queue_info
                },
                "consuming": self._consuming
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Глобальный экземпляр клиента
rmq_client = RMQClient()