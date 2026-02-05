"""
Utils — утилиты для бенчмарка

Модули:
- hashing: Хеширование и анализ детерминизма
- file_ops: Работа с файлами (YAML, JSON)
- logging: Настройка логирования
- code_export: Экспорт кода в .bsl файлы
- cli: Форматирование CLI вывода
"""

from .hashing import (
    compute_hash,
    compute_hash_with_settings,
    normalize_code,
    compare_hashes,
    is_deterministic,
)
from .file_ops import load_yaml, load_json, save_json, save_yaml, ensure_dir
from .code_export import (
    export_experiment_code,
    export_from_json_file,
    export_code_to_bsl,
)

__all__ = [
    # Hashing
    "compute_hash",
    "compute_hash_with_settings",
    "normalize_code",
    "compare_hashes",
    "is_deterministic",
    # File operations
    "load_yaml",
    "load_json",
    "save_json",
    "save_yaml",
    "ensure_dir",
    # Code export
    "export_experiment_code",
    "export_from_json_file",
    "export_code_to_bsl",
]
