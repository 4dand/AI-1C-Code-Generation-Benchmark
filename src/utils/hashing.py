"""
Hashing Utilities — функции хеширования для анализа детерминизма

Модуль предоставляет инструменты для:
- Нормализации кода 1С перед сравнением
- Вычисления хешей ответов моделей  
- Анализа детерминизма (совпадения ответов между прогонами)

Использование:
    from src.utils.hashing import compute_hash, compare_hashes
    
    # Хеширование ответа модели
    hash1 = compute_hash(response1)
    hash2 = compute_hash(response2)
    
    # Анализ детерминизма
    stats = compare_hashes([hash1, hash2, hash3])
    print(f"Совпадений: {stats['match_rate']:.1%}")
"""

import re
import hashlib
from collections import Counter
from typing import Literal, Dict, List, Optional

from ..config.settings import get_settings


# =============================================================================
# Нормализация кода
# =============================================================================

# Паттерн для извлечения кода из markdown блоков
CODE_BLOCK_PATTERN = re.compile(
    r'```(?:1c|1С|bsl|)?\s*\n(.*?)```',
    re.DOTALL | re.IGNORECASE
)


def normalize_code(text: str) -> str:
    """
    Нормализация кода 1С для корректного сравнения
    
    Выполняет:
    - Извлечение кода из markdown блоков (```1c ... ```)
    - Удаление пустых строк в начале и конце
    - Удаление trailing whitespace
    
    Args:
        text: Исходный текст ответа модели
        
    Returns:
        Нормализованный код готовый к хешированию
        
    Example:
        >>> text = "```1c\\nПроцедура Тест()\\nКонецПроцедуры\\n```"
        >>> normalize_code(text)
        'Процедура Тест()\\nКонецПроцедуры'
    """
    if not text:
        return ""
    
    # Извлекаем код из markdown блока если есть
    code_match = CODE_BLOCK_PATTERN.search(text)
    if code_match:
        text = code_match.group(1)
    
    # Убираем trailing whitespace и нормализуем
    lines = [line.rstrip() for line in text.strip().split('\n')]
    
    # Убираем пустые строки в начале
    while lines and not lines[0].strip():
        lines.pop(0)
    
    # Убираем пустые строки в конце
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)


# =============================================================================
# Хеширование
# =============================================================================

def compute_hash(
    text: str, 
    normalize: bool = True,
    algorithm: Optional[Literal["md5", "sha256"]] = None
) -> str:
    """
    Вычислить хеш текста для сравнения ответов
    
    Args:
        text: Текст для хеширования
        normalize: Нормализовать код перед хешированием (рекомендуется)
        algorithm: Алгоритм хеширования (если None — из Settings)
        
    Returns:
        Хеш-строка (hex)
        
    Example:
        >>> compute_hash("Процедура Тест()\\nКонецПроцедуры")
        'd41d8cd98f00b204e9800998ecf8427e'
    """
    # Получаем настройки если algorithm не указан
    if algorithm is None:
        settings = get_settings()
        algorithm = settings.hashing.algorithm
    
    if normalize:
        text = normalize_code(text)
    
    encoded = text.encode('utf-8')
    
    if algorithm == "sha256":
        return hashlib.sha256(encoded).hexdigest()
    else:
        return hashlib.md5(encoded).hexdigest()


def compute_hash_with_settings(text: str) -> str:
    """
    Вычислить хеш используя настройки из Settings
    
    Удобная функция для использования в BenchmarkRunner.
    Параметры берутся из settings.hashing.
    
    Args:
        text: Текст для хеширования
        
    Returns:
        Хеш-строка
    """
    settings = get_settings()
    
    return compute_hash(
        text=text,
        normalize=settings.hashing.normalize,
        algorithm=settings.hashing.algorithm
    )


# =============================================================================
# Анализ детерминизма
# =============================================================================

def compare_hashes(hashes: List[str]) -> Dict[str, any]:
    """
    Анализ детерминизма по списку хешей
    
    Вычисляет статистику совпадения ответов между прогонами.
    match_rate показывает какая доля ответов совпадает с самым частым.
    
    Args:
        hashes: Список хешей от всех прогонов
        
    Returns:
        Словарь со статистикой:
        - total_runs: Всего прогонов
        - unique_count: Количество уникальных ответов
        - match_rate: Доля совпадений (0.0 - 1.0)
        - most_common_hash: Самый частый хеш
        - most_common_count: Сколько раз встретился самый частый
        
    Example:
        >>> compare_hashes(["abc", "abc", "def"])
        {'total_runs': 3, 'unique_count': 2, 'match_rate': 0.667, ...}
        
        >>> compare_hashes(["abc", "abc", "abc"])  # 100% детерминизм
        {'total_runs': 3, 'unique_count': 1, 'match_rate': 1.0, ...}
    """
    if not hashes:
        return {
            "total_runs": 0,
            "unique_count": 0,
            "match_rate": 0.0,
            "most_common_hash": "",
            "most_common_count": 0
        }
    
    counter = Counter(hashes)
    most_common_hash, most_common_count = counter.most_common(1)[0]
    
    # match_rate = доля ответов совпадающих с самым частым
    # [A, A, B] -> 2/3 = 0.667 (67% детерминизм)
    # [A, A, A] -> 3/3 = 1.0   (100% детерминизм)
    # [A, B, C] -> 1/3 = 0.333 (33% детерминизм)
    match_rate = most_common_count / len(hashes)
    
    return {
        "total_runs": len(hashes),
        "unique_count": len(counter),
        "match_rate": match_rate,
        "most_common_hash": most_common_hash,
        "most_common_count": most_common_count
    }


def is_deterministic(hashes: List[str]) -> bool:
    """
    Проверить являются ли все ответы идентичными
    
    Args:
        hashes: Список хешей
        
    Returns:
        True если все хеши одинаковые
    """
    if not hashes:
        return False
    return len(set(hashes)) == 1
