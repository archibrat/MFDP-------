import uuid
from typing import Optional, Any
import pika
import time
import logging
from functools import wraps
from services.rm.rmqconf import RabbitMQConfig

# Устанавливаем уровень WARNING для логов pika
logging.getLogger('pika').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def retry_connection(retries: int = 3, delay: float = 1.0):
    """Декоратор для повторных попыток подключения."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(self, *args, **kwargs)
                except (Exception, pika.exceptions.AMQPError) as e:
                    last_exception = e
                    if attempt < retries - 1:
                        time.sleep(delay)
                        self._setup_connection()
            raise last_exception
        return wrapper
    return decorator

class RpcClient:
    """RPC клиент для удаленного вызова процедур через RabbitMQ."""

    def __init__(self, config: Optional[RabbitMQConfig] = None) -> None:
        """
        Инициализация RPC клиента.
        
        Args:
            config: Конфигурация подключения к RabbitMQ
        """
        self.config = config or RabbitMQConfig()
        self.rpc_queue_name = self.config.rpc_queue_name
        self.connection_params = self.config.get_connection_params()
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        self.callback_queue: Optional[str] = None
        self.response: Optional[str] = None
        self.corr_id: Optional[str] = None
        self._connect_attempt: int = 0
        self._max_connect_attempts: int = 3
        
        self._setup_connection()

    def _setup_connection(self) -> None:
        """Установка соединения с RabbitMQ и настройка канала."""
        try:
            if self.connection and not self.connection.is_closed:
                return

            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()
            
            self._declare_queues()
            self._setup_callback_queue()
            self._connect_attempt = 0
            
        except pika.exceptions.AMQPConnectionError as e:
            self._connect_attempt += 1
            if self._connect_attempt >= self._max_connect_attempts:
                logger.error(f"Failed to connect after {self._max_connect_attempts} attempts: {e}")
                raise Exception(f"Failed to connect to RabbitMQ: {e}")
            raise

    def _declare_queues(self) -> None:
        """Объявление необходимых очередей."""
        try:
            self.channel.queue_declare(queue=self.rpc_queue_name, passive=True)
        except pika.exceptions.ChannelClosedByBroker:
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.rpc_queue_name)

    def _setup_callback_queue(self) -> None:
        """Настройка очереди обратного вызова."""
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

    def on_response(self, ch, method, props, body) -> None:
        """Обработка ответа от RPC сервера."""
        if self.corr_id == props.correlation_id:
            self.response = body.decode()

    @retry_connection()
    def call(self, text: str, timeout: float = 10.0) -> str:
        """Выполнить RPC вызов с переданным текстом.
        
        Args:
            text: Текст для отправки
            timeout: Максимальное время ожидания ответа в секундах
        
        Raises:
            Exception: Если ответ не получен в течение timeout секунд
        """
        try:
            self._setup_connection()
            self.response = None
            self.corr_id = str(uuid.uuid4())

            self._publish_message(text)
            return self._wait_response(timeout)

        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise Exception(f"RPC call failed: {str(e)}")

    def _publish_message(self, text: str) -> None:
        """Публикация сообщения в очередь."""
        self.channel.basic_publish(
            exchange='',
            routing_key=self.rpc_queue_name,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=text.encode()
        )

    def _wait_response(self, timeout: float) -> str:
        """Ожидание ответа от сервера."""
        start_time = time.time()
        while self.response is None:
            if time.time() - start_time > timeout:
                raise Exception(f"Request timed out after {timeout} seconds")
            try:
                self.connection.process_data_events(time_limit=0.5)
            except Exception as e:
                raise Exception(f"Error processing events: {str(e)}")
        return self.response

    def close(self) -> None:
        """Закрыть соединение с RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()

    def __enter__(self):
        """Вход в контекстный менеджер."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера."""
        self.close()

# Экземпляр-синглтон с настройками по умолчанию
rpc_client = RpcClient(RabbitMQConfig())