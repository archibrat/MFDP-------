import logging
import os
from pathlib import Path


def get_logger(level=logging.DEBUG, logger_name='default logger') -> logging.Logger:
    """
    Создает и настраивает логгер с указанным уровнем логирования и именем.

    Args:
        level (int): Уровень логирования (по умолчанию logging.DEBUG)
        logger_name (str): Имя логгера (по умолчанию 'default logger')

    Returns:
        logging.Logger: Настроенный объект логгера

    Raises:
        OSError: Если невозможно создать директорию для логов
    """
    # Создаем директорию для логов, если она не существует
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)

    # Устанавливаем базовую конфигурацию логирования
    logging.basicConfig(level=level)

    # Создаем обработчик для записи логов в файл
    handler = logging.FileHandler('./logs/myapp.log')
    
    # Определяем формат сообщений лога:
    # %(asctime)s - временная метка
    # %(name)s - имя логгера
    # %(levelname)s - уровень важности сообщения
    # %(message)s - текст сообщения
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Применяем форматирование к обработчику
    handler.setFormatter(formatter)

    # Получаем логгер с указанным именем
    logger = logging.getLogger(logger_name)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(handler)
    
    # Устанавливаем уровень логирования
    logger.setLevel(level)

    return logger
