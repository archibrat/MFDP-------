from rmq.rmqconf import RabbitMQConfig
from llm import do_task
import pika
import time
import requests
import logging
import json

# Настраиваем общий уровень логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Устанавливаем уровень WARNING для логов pika
logging.getLogger('pika').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Определяем основной класс для обработки ML задач
class MLWorker:
    """
    Рабочий класс для обработки ML задач из очереди RabbitMQ.
    Обеспечивает подключение к очереди и обработку поступающих сообщений.
    """
    # Константы класса
    MAX_RETRIES = 3
    RETRY_DELAY = 0.5
    RESULT_ENDPOINT = 'http://app:8080/api/ml/send_task_result'
    
    def __init__(self, config: RabbitMQConfig):
        """
        Инициализация обработчика с заданной конфигурацией.
        
        Args:
            config: Объект конфигурации RabbitMQ
        """
        # Сохраняем конфигурацию
        self.config = config
        # Инициализируем соединение как None
        self.connection = None
        # Инициализируем канал как None
        self.channel = None
        self.retry_count = 0
        
    def connect(self) -> None:
        """
        Установка соединения с сервером RabbitMQ с повторными попытками.
        """
        while True:
            try:
                connection_params = self.config.get_connection_params()
                self.connection = pika.BlockingConnection(connection_params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.config.queue_name)
                logger.info("Successfully connected to RabbitMQ")
                break
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {e}")
                time.sleep(self.RETRY_DELAY)

    def cleanup(self):
        """Корректное закрытие соединений с RabbitMQ"""
        try:
            if self.channel:
                self.channel.close()
            if self.connection:
                self.connection.close()
            logger.info("Соединения успешно закрыты")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединений: {e}")

    def send_result(self, task_id: str, result: str) -> bool:
        """
        Отправка результатов обработки задачи на сервер.
        
        Returns:
            bool: Признак успешности отправки результата
        """
        try:
            response = requests.post(
                self.RESULT_ENDPOINT,
                params={'task_id': task_id, 'result': result}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send result: {e}")
            return False

    def process_message(self, ch, method, properties, body):
        """
        Обработка полученного сообщения из очереди.
        
        Args:
            ch: Объект канала RabbitMQ
            method: Метод доставки сообщения
            properties: Свойства сообщения
            body: Тело сообщения
            
        Note:
            Симулирует обработку задачи с задержкой в 3 секунды
        """
        try:
            # Логируем информацию о полученном сообщении
            logger.info(f"Processing message: {body}")
            
            # Декодируем bytes в строку и затем парсим JSON
            data = json.loads(body.decode('utf-8'))
            
            result = do_task(data['question'])
            
            logger.info(f"Result: {result}")
            
            if self.send_result(data['task_id'], result):
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.retry_count = 0
                logger.info("Task completed successfully")
            else:
                raise Exception("Failed to send result")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.retry_count += 1
            
            if self.retry_count >= self.MAX_RETRIES:
                logger.error("Max retries reached, rejecting message")
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
                self.retry_count = 0
            else:
                time.sleep(self.RETRY_DELAY)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
    def start_consuming(self) -> None:
        """
        Запуск процесса получения сообщений из очереди.
        
        Note:
            Блокирующая операция, прерывается по Ctrl+C
        """
        try:
            # Настраиваем потребление сообщений из очереди
            self.channel.basic_consume(
                queue=self.config.queue_name,  # Имя очереди
                on_message_callback=self.process_message,  # Callback для обработки сообщений
                auto_ack=False  # Отключаем автоматическое подтверждение
            )
            # Логируем информацию о старте потребления сообщений
            logger.info('Started consuming messages. Press Ctrl+C to exit.')
            # Запускаем потребление сообщений
            self.channel.start_consuming()
        except KeyboardInterrupt:
            # Логируем информацию о завершении работы
            logger.info("Shutting down...")
        finally:
            # Закрываем соединение при завершении работы
            self.cleanup()
