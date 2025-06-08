from rmq.rmqconf import RabbitMQConfig
from rmq.rmqworker import MLWorker
from rmq.rpcworker import RPCWorker
import sys
import pika
import time
import logging

# Настраиваем базовую конфигурацию логирования
logging.basicConfig(
    level=logging.DEBUG,  # Устанавливаем уровень логирования DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Задаем формат сообщений лога
)

logger = logging.getLogger(__name__)

def create_worker(mode: str, config: RabbitMQConfig):
    """Create appropriate worker instance based on mode."""
    return MLWorker(config) if mode == 'ml' else RPCWorker(config)

def run_worker(worker):
    """Run worker with reconnection logic."""
    while True:
        try:
            if not worker.connection or not worker.connection.is_open:
                logger.info("Connecting to RabbitMQ...")
                worker.connect()
            
            logger.info("Starting message consumption...")
            worker.start_consuming()
            
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Connection error: {e}")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        
        time.sleep(1)

def main():
    mode = 'rpc' # Можно использовать rpc
    logger.info(f"Starting worker in {mode} mode")
    
    worker = None
    try:
        config = RabbitMQConfig()
        worker = create_worker(mode, config)
        run_worker(worker)
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
