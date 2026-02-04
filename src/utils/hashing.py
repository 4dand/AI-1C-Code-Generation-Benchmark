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
        hashes: Список хешей от всех прогонов
        
    Returns:
        {
            "total_runs": int,  # Всего прогонов
            "unique_count": int,  # Количество уникальных ответов
            "match_rate": float,  # Процент совпадений (0.0 - 1.0)
            "most_common_hash": str,  # Самый частый хеш
            "most_common_count": int,  # Сколько раз встретился
        }
    """
    if not hashes:
        return {
            "total_runs": 0,
            "unique_count": 0,
            "match_rate": 0.0,
            "most_common_hash": "",
            "most_common_count": 0
        }
    
    from collections import Counter
    counter = Counter(hashes)
    most_common_hash, most_common_count = counter.most_common(1)[0]
    
    # match_rate = сколько ответов совпадают с самым частым / всего
    # Например: [A, A, B] -> 2/3 = 0.667 (67%)
    # Например: [A, A, A] -> 3/3 = 1.0 (100%)
    # Например: [A, B, C] -> 1/3 = 0.333 (33%)
    match_rate = most_common_count / len(hashes)
    
    return {
        "total_runs": len(hashes),
        "unique_count": len(counter),
        "match_rate": match_rate,
        "most_common_hash": most_common_hash,
        "most_common_count": most_common_count
    }
