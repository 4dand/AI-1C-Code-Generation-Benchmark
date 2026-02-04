"""
logging utilities - логирование
"""

import logging
from datetime import datetime
from typing import Optional

# глобальный логгер
_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "benchmark",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Настроить логгер
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу логов (опционально)
        
    Returns:
        Настроенный логгер
    """
    global _logger
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Получить текущий логгер или создать новый"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def log(message: str, level: str = "INFO") -> None:
    """
    Простое логирование с автоматическим timestamp
    
    Args:
        message: Сообщение
        level: Уровень (INFO, WARNING, ERROR, DEBUG)
    """
    logger = get_logger()
    
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    logger.log(log_level, message)


def log_section(title: str, char: str = "=", width: int = 60) -> None:
    """Вывести заголовок секции"""
    logger = get_logger()
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)
