"""
Core — ядро бенчмарка

Основные компоненты:
- BenchmarkRunner: Оркестратор запуска экспериментов
- AgenticContextLoader: Агентный загрузчик контекста метаданных
"""

from .benchmark import BenchmarkRunner
from .context_loader import AgenticContextLoader, SmartContextLoader

__all__ = [
    # Основные классы
    "BenchmarkRunner",
    "AgenticContextLoader",
    # Алиасы для совместимости
    "SmartContextLoader",
]
