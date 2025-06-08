import json
import pika
import logging
from typing import Optional
from models.mltask import MLTask, TaskStatus

# Устанавливаем уровень WARNING для логов pika
logging.getLogger('pika').setLevel(logging.INFO)

class RabbitMQClient:
    """
    Клиент для взаимодействия с RabbitMQ.
    
    Attributes:
        connection_params: Параметры подключения к RabbitMQ серверу
        queue_name: Имя очереди для ML задач
    """
    
    def __init__(
        self,
        host: str = 'rabbitmq',
        port: int = 5672,
        username: str = 'rmuser',
        password: str = 'rmpassword',
        queue_name: str = 'ml_task_queue'
    ):
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            virtual_host='/',
            credentials=pika.PlainCredentials(username=username, password=password),
            heartbeat=30,
            blocked_connection_timeout=2
        )
        self.queue_name = queue_name

    def send_task(self, task: MLTask) -> bool:
        """
        Отправляет ML задачу в очередь RabbitMQ.
        
        Args:
            task: Объект MLTask для обработки
            
        Returns:
            bool: True если отправка прошла успешно, False в случае ошибки
            
        Raises:
            pika.exceptions.AMQPError: При проблемах с подключением к RabbitMQ
        """
        try:
            connection = pika.BlockingConnection(self.connection_params)
            channel = connection.channel()
            
            # Создаем очередь если её нет
            channel.queue_declare(queue=self.queue_name)
            
            # Подготавливаем сообщение
            message = json.dumps(task.to_queue_message())
            
            # Отправляем сообщение
            channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=message
            )
            
            connection.close()
            return True
            
        except pika.exceptions.AMQPError as e:
            print(f"RabbitMQ error: {str(e)}")
            return False

# Создаем глобальный экземпляр клиента
rabbit_client = RabbitMQClient()

# def send_ml_task(task: MLTask) -> bool:
#     """
#     Отправляет ML задачу на обработку.
    
#     Args:
#         task: Объект MLTask для обработки
        
#     Returns:
#         bool: True если задача успешно отправлена, False в случае ошибки
#     """
#     success = rabbit_client.send_task(task)
#     if success:
#         task.status = TaskStatus.QUEUED
#     return success