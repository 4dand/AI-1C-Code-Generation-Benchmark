"""
hashing utilities - функции хеширования для сравнения ответов
"""

import re
import hashlib
from typing import Literal


def normalize_code(text: str) -> str:
    """
    Нормализация кода для хеширования:
    - Извлечь код из markdown блоков
    - Убрать лишние пробелы и переносы
    - Унифицировать отступы
    
    Args:
        text: Исходный текст ответа модели
        
    Returns:
        Нормализованный код
    """
    if not text:
        return ""
    
    code_match = re.search(
        r'```(?:1c|1С|bsl|)?\s*\n(.*?)```', 
        text, 
        re.DOTALL | re.IGNORECASE
    )
    if code_match:
        text = code_match.group(1)
    
    lines = [line.rstrip() for line in text.strip().split('\n')]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)


def compute_hash(
    text: str, 
    normalize: bool = True,
    algorithm: Literal["md5", "sha256"] = "md5"
) -> str:
    """
    Вычислить хеш текста
    
    Args:
        text: Текст для хеширования
        normalize: Нормализовать код перед хешированием
        algorithm: Алгоритм хеширования (md5 или sha256)
        
    Returns:
        Хеш строка
    """
    if normalize:
        text = normalize_code(text)
    
    encoded = text.encode('utf-8')
    if algorithm == "sha256":
        return hashlib.sha256(encoded).hexdigest()
    else:
        return hashlib.md5(encoded).hexdigest()


def compare_hashes(hashes: list[str]) -> dict:
    """
    Сравнить список хешей для анализа детерминизма
    
    Args:
        hashes: Список хешей (минимум 3: seed1, seed1, seed2)
        
    Returns:
        {
            "same_seed_match": bool,  # Совпадают ли первые два
            "different_seed_differs": bool,  # Отличается ли третий
            "all_same": bool,  # Все одинаковые
            "unique_count": int  # Количество уникальных
        }
    """
    if len(hashes) < 2:
        return {
            "same_seed_match": True,
            "different_seed_differs": False,
            "all_same": True,
            "unique_count": len(set(hashes))
        }
    
    same_seed_match = hashes[0] == hashes[1] if len(hashes) >= 2 else True
    different_seed_differs = hashes[0] != hashes[2] if len(hashes) >= 3 else False
    unique = set(hashes)
    
    return {
        "same_seed_match": same_seed_match,
        "different_seed_differs": different_seed_differs,
        "all_same": len(unique) == 1,
        "unique_count": len(unique)
    }
