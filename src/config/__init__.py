"""
Config module — централизованная конфигурация проекта

Экспортирует:
- Settings: класс настроек
- get_settings(): получить singleton настроек
- reload_settings(): перезагрузить настройки
- setup_logging(): настроить логирование согласно конфигу
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .settings import Settings, get_settings, reload_settings


def setup_logging(settings: Optional[Settings] = None) -> None:
    """
    Настроить логирование согласно settings.yaml
    
    Настраивает:
    - Консольный вывод (если enabled)
    - Файловый вывод с ротацией (если enabled)
    
    Args:
        settings: Объект настроек (если None — загружается автоматически)
    """
    if settings is None:
        settings = get_settings()
    
    log_config = settings.logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.level, logging.INFO))
    root_logger.handlers.clear()
    formatter = logging.Formatter(
        fmt=log_config.format,
        datefmt=log_config.date_format,
    )
    if log_config.console.enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_config.console.level, logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
   
    if log_config.file.enabled:
        log_file_path = settings.get_log_file_path()
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if log_config.file.rotation.enabled:
            file_handler = RotatingFileHandler(
                filename=str(log_file_path),
                maxBytes=log_config.file.rotation.max_size_mb * 1024 * 1024,
                backupCount=log_config.file.rotation.backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(
                filename=str(log_file_path),
                encoding="utf-8",
            )
        
        file_handler.setLevel(getattr(logging, log_config.file.level, logging.DEBUG))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.debug(f"Логирование настроено: level={log_config.level}, file={log_config.file.enabled}")


__all__ = ["Settings", "get_settings", "reload_settings", "setup_logging"]
