import asyncio
import uuid
import pika
from functools import wraps
from typing import Optional
import logging
import time

from .rmqconf import RabbitMQConfig

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
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        time.sleep(delay)
                        self._setup_connection()
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

class RpcClient:
    """RPC клиент для взаимодействия с RabbitMQ"""
    
    def __init__(self, config: Optional[RabbitMQConfig] = None) -> None:
        self.config = config or RabbitMQConfig()
        self.rpc_queue_name = "rpc_queue"
        self._max_connect_attempts = 3
        self._connect_attempt = 0
        
        # Параметры подключения
        self.connection_params = self.config.get_connection_params()
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel = None  # type: ignore
        self.callback_queue: Optional[str] = None
        self.response: Optional[str] = None
        self.corr_id: Optional[str] = None
        
        # Инициализация подключения
        self._setup_connection()
        self._declare_queues()
        self._setup_callback_queue()
    
    def _setup_connection(self) -> None:
        """Настройка подключения к RabbitMQ"""
        try:
            if self.connection and not self.connection.is_closed:
                if self.channel is None or self.channel.is_closed:
                    self.channel = self.connection.channel()
                return
                
            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()
            self._connect_attempt = 0
            
        except Exception as e:
            self._connect_attempt += 1
            if self._connect_attempt >= self._max_connect_attempts:
                logger.error(f"Failed to connect after {self._max_connect_attempts} attempts: {e}")
                raise
            logger.warning(f"Connection attempt {self._connect_attempt} failed: {e}")
            time.sleep(1)
            self._setup_connection()

    def _declare_queues(self) -> None:
        """Объявление необходимых очередей."""
        try:
            if self.channel:
                try:
                    self.channel.queue_declare(queue=self.rpc_queue_name, passive=True)
                except Exception:
                    self.channel.queue_declare(queue=self.rpc_queue_name)
        except Exception as e:
            logger.error(f"Ошибка объявления очереди: {e}")

    def _setup_callback_queue(self) -> None:
        """Настройка очереди обратного вызова."""
        if self.channel:
            result = self.channel.queue_declare(queue='', exclusive=True)
            self.callback_queue = result.method.queue
            self.channel.basic_consume(
                queue=self.callback_queue,
                on_message_callback=self.on_response,
                auto_ack=True
            )

    def on_response(self, ch, method, props, body) -> None:
        """Обработчик ответа от RPC сервера"""
        if self.corr_id == props.correlation_id:
            self.response = body.decode()

    @retry_connection()
    def call(self, text: str, timeout: float = 10.0) -> str:
        """Отправка RPC запроса"""
        self.response = None
        self.corr_id = str(uuid.uuid4())
        
        try:
            self._publish_message(text)
            return self._wait_response(timeout)
        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise

    def _publish_message(self, text: str) -> None:
        """Публикация сообщения в очередь."""
        if self.channel:
            self.channel.basic_publish(
                exchange='',
                routing_key=self.rpc_queue_name,
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    correlation_id=self.corr_id,
                ),
                body=text
            )

    def _wait_response(self, timeout: float) -> str:
        """Ожидание ответа с таймаутом"""
        start_time = time.time()
        while self.response is None:
            if time.time() - start_time > timeout:
                raise Exception(f"Request timed out after {timeout} seconds")
            try:
                if self.connection:
                    self.connection.process_data_events(time_limit=1)
            except Exception as e:
                raise Exception(f"Error processing events: {str(e)}")
        return self.response

    def close(self) -> None:
        """Закрытие соединения"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Глобальный экземпляр RPC клиента
_rpc_client: Optional[RpcClient] = None

def get_rpc_client() -> RpcClient:
    """Получение глобального экземпляра RPC клиента"""
    global _rpc_client
    if _rpc_client is None:
        try:
            _rpc_client = RpcClient()
        except Exception as e:
            logger.warning(f"Не удалось создать RPC клиент: {e}")
            raise
    return _rpc_client