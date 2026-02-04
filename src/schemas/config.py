"""
Configuration Schemas - структуры для конфигурации
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Конфигурация модели из models.yaml"""
    id: str
    name: str
    context_window: int = 0
    price_input: float = 0.0
    price_output: float = 0.0
    supports_seed: bool = False
    supports_tools: bool = False
    determinism_param: str = "temperature"  # "seed" или "temperature"


@dataclass
class TaskConfig:
    """Конфигурация задачи"""
    id: str
    name: str
    difficulty: str
    prompt: str
    expected_objects: List[str] = field(default_factory=list)  # Для категории B


@dataclass
class GenerationParams:
    """Параметры генерации"""
    temperature: float = 0.0
    max_tokens: int = 4096
    seeds: List[int] = field(default_factory=lambda: [42, 42, 999])
    runs: int = 3  # Для моделей без seed


@dataclass
class MCPConfig:
    """Конфигурация MCP сервера"""
    url: str = "http://localhost:3000"
    timeout: int = 30
    use_mock: bool = True


@dataclass
class ExperimentConfig:
    """Полная конфигурация эксперимента"""
    name: str
    version: str
    category: str  # "A" или "B"
    
    # Пути
    results_dir: str = "results"
    
    # Параметры хеширования
    hash_algorithm: str = "md5"
    normalize_code: bool = True
    
    # Генерация
    generation: GenerationParams = field(default_factory=GenerationParams)
    
    # MCP (для категории B)
    mcp: MCPConfig = field(default_factory=MCPConfig)
