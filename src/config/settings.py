"""
Settings - централизованная конфигурация проекта

Единая точка доступа ко всем настройкам:
- Загружает configs/settings.yaml
- Загружает секреты из .env
- Валидирует значения через Pydantic
- Предоставляет типизированный доступ

Использование:
    from src.config import get_settings
    
    settings = get_settings()
    print(settings.openrouter.api_key)
    print(settings.paths.raw_results_dir)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Вложенные модели - точно соответствуют settings.yaml
# =============================================================================

class PathsConfig(BaseModel):
    """Пути проекта"""
    configs_dir: str = "configs"
    raw_results_dir: str = "raw_results"
    code_outputs_dir: str = "code_outputs"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    evaluations_dir: str = "evaluations"
    reports_dir: str = "reports"
    
    # Файлы конфигов (относительно configs_dir)
    experiment_file: str = "experiment.yaml"
    models_file: str = "models.yaml"
    tasks_category_a: str = "tasks_category_A.yaml"
    tasks_category_b: str = "tasks_category_B.yaml"
    smop_criteria_file: str = "smop_criteria.yaml"
    
    def get_models_path(self) -> Path:
        """Полный путь к models.yaml"""
        return Path(self.configs_dir) / self.models_file
    
    def get_tasks_path(self, category: str) -> Path:
        """Полный путь к файлу задач категории"""
        filename = self.tasks_category_a if category.upper() == "A" else self.tasks_category_b
        return Path(self.configs_dir) / filename
    
    def get_experiment_path(self) -> Path:
        """Полный путь к experiment.yaml"""
        return Path(self.configs_dir) / self.experiment_file
    
    def get_smop_criteria_path(self) -> Path:
        """Полный путь к smop_criteria.yaml"""
        return Path(self.configs_dir) / self.smop_criteria_file


class OpenRouterConfig(BaseModel):
    """Настройки OpenRouter API"""
    api_key: Optional[str] = None  # Загружается из .env
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str = "https://1c-benchmark.local"
    app_title: str = "AI-1C-Code-Generation-Benchmark"
    timeout: int = 120


class MCPConfig(BaseModel):
    """Настройки MCP сервера"""
    url: str = "http://localhost:8000"
    timeout: int = 30


class AgentCacheConfig(BaseModel):
    """Настройки кэширования агента"""
    enabled: bool = True
    dir: str = "context"  # Подпапка в cache_dir
    ttl_hours: int = 24


class AgentPromptsConfig(BaseModel):
    """Промпты агента контекста"""
    system: str = ""
    user_template: str = ""


class AgentConfig(BaseModel):
    """Настройки агента контекста"""
    model: str = "google/gemini-2.0-flash-001"
    max_iterations: int = 5
    max_context_lines: int = 50
    max_total_context_chars: int = 15000  # Лимит общего контекста в символах
    max_objects: int = 3  # Максимум объектов метаданных
    cache: AgentCacheConfig = Field(default_factory=AgentCacheConfig)
    prompts: AgentPromptsConfig = Field(default_factory=AgentPromptsConfig)


class EvaluatorConfig(BaseModel):
    """Настройки модуля оценки SMOP"""
    autosave_interval: int = 30  # секунд
    default_evaluator_id: str = "expert_01"
    quality_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"high": 8.0, "acceptable": 5.0, "low": 0.0}
    )


class HashingConfig(BaseModel):
    """Настройки хеширования"""
    algorithm: Literal["md5", "sha256"] = "md5"
    normalize: bool = True
    extract_code: bool = True


class LoggingConsoleConfig(BaseModel):
    """Настройки консольного логирования"""
    enabled: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    colorize: bool = True


class LoggingRotationConfig(BaseModel):
    """Настройки ротации логов"""
    enabled: bool = True
    max_size_mb: int = 10
    backup_count: int = 5


class LoggingFileConfig(BaseModel):
    """Настройки файлового логирования"""
    enabled: bool = True
    path: str = "benchmark.log"
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    rotation: LoggingRotationConfig = Field(default_factory=LoggingRotationConfig)


class LoggingConfig(BaseModel):
    """Настройки логирования"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    console: LoggingConsoleConfig = Field(default_factory=LoggingConsoleConfig)
    file: LoggingFileConfig = Field(default_factory=LoggingFileConfig)


class ExportConfig(BaseModel):
    """Настройки экспорта"""
    code_to_bsl: bool = True


# =============================================================================
# Главный класс настроек
# =============================================================================

class Settings(BaseSettings):
    """
    Централизованные настройки проекта
    
    Загружает:
    1. Дефолтные значения из класса
    2. Значения из configs/settings.yaml
    3. Переменные окружения (секреты)
    
    Приоритет: env > yaml > defaults
    """
    
    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        extra="ignore"
    )
    
    # Вложенные конфиги - соответствуют секциям settings.yaml
    paths: PathsConfig = Field(default_factory=PathsConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    hashing: HashingConfig = Field(default_factory=HashingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    
    def __init__(self, **kwargs):
        # Загружаем YAML используя file_ops
        yaml_data = self._load_yaml_config()
        
        # Мержим: yaml < kwargs (kwargs имеет приоритет)
        merged = self._deep_merge(yaml_data, kwargs)
        
        # Загружаем API ключ из окружения
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            if "openrouter" not in merged:
                merged["openrouter"] = {}
            merged["openrouter"]["api_key"] = api_key
        
        super().__init__(**merged)
    
    @staticmethod
    def _load_yaml_config() -> dict:
        """Загрузить settings.yaml используя file_ops"""
        # Импортируем здесь чтобы избежать циклических импортов
        from src.utils.file_ops import load_yaml
        
        # Ищем settings.yaml относительно корня проекта
        possible_paths = [
            Path("configs/settings.yaml"),
            Path(__file__).parent.parent.parent / "configs" / "settings.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    return load_yaml(path)
                except Exception:
                    return {}
        
        return {}
    
    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Глубокое слияние словарей"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Settings._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_api_key(self) -> None:
        """Проверить наличие API ключа"""
        if not self.openrouter.api_key:
            raise ValueError(
                "API ключ не найден. Установите переменную окружения OPENROUTER_API_KEY "
                "или добавьте её в файл .env"
            )
    
    def get_cache_context_path(self) -> Path:
        """Полный путь к папке кэша контекста"""
        return Path(self.paths.cache_dir) / self.agent.cache.dir
    
    def get_log_file_path(self) -> Path:
        """Полный путь к файлу логов"""
        return Path(self.paths.logs_dir) / self.logging.file.path


# =============================================================================
# Singleton и доступ к настройкам
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Получить экземпляр настроек (singleton)
    
    Использование:
        settings = get_settings()
        print(settings.paths.raw_results_dir)
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Перезагрузить настройки (очистить кеш)
    
    Использовать при изменении конфигов в runtime
    """
    get_settings.cache_clear()
    return get_settings()
