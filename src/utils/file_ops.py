"""
file operations - работа с файлами
"""

import json
import yaml
from pathlib import Path
from typing import Any, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Создать директорию если не существует
    
    Args:
        path: Путь к директории
        
    Returns:
        Path объект
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Union[str, Path]) -> dict:
    """
    Загрузить YAML файл
    
    Args:
        path: Путь к файлу
        
    Returns:
        Словарь с данными
        
    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: Если файл невалидный
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Union[str, Path]) -> None:
    """
    Сохранить данные в YAML файл
    
    Args:
        data: Данные для сохранения
        path: Путь к файлу
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Сохранить данные в JSON файл
    
    Args:
        data: Данные для сохранения
        path: Путь к файлу
        indent: Отступ для форматирования
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    """
    Загрузить JSON файл
    
    Args:
        path: Путь к файлу
        
    Returns:
        Данные из файла
    """
    path = Path(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
