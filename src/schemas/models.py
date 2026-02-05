"""
Model Schemas — схемы для конфигурации моделей из models.yaml

Отвечает за:
- Валидацию конфигурации моделей
- Типизацию параметров генерации для конкретных моделей
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class ModelGenerationParams(BaseModel):
    """Параметры генерации для конкретной модели"""
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seeds: Optional[List[int]] = None  # Для моделей с поддержкой seed
    runs: Optional[int] = None  # Для моделей без seed (claude)
    
    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is not None and len(v) == 0:
            raise ValueError("seeds не может быть пустым списком")
        return v


class ModelMeta(BaseModel):
    """Метаданные модели (неизменные характеристики)"""
    context_window: int = Field(default=0, ge=0)
    price_input: float = Field(default=0.0, ge=0.0, description="Цена за 1M входных токенов")
    price_output: float = Field(default=0.0, ge=0.0, description="Цена за 1M выходных токенов")
    supports_seed: bool = False
    supports_tools: bool = True
    determinism_param: Literal["seed", "temperature"] = "temperature"


class ModelConfig(BaseModel):
    """
    Конфигурация модели из models.yaml
    
    Пример YAML:
        claude:
          id: "anthropic/claude-sonnet-4"
          name: "Claude Sonnet 4"
          meta:
            context_window: 200000
            supports_seed: false
          generation:
            temperature: 0.0
            runs: 3
    """
    id: str = Field(..., description="ID модели для OpenRouter API")
    name: str = Field(..., description="Человекочитаемое название")
    meta: ModelMeta = Field(default_factory=ModelMeta)
    generation: ModelGenerationParams = Field(default_factory=ModelGenerationParams)
    
    @property
    def supports_seed(self) -> bool:
        """Поддерживает ли модель seed для детерминизма"""
        return self.meta.supports_seed
    
    @property
    def runs_count(self) -> int:
        """Количество прогонов для этой модели"""
        if self.generation.seeds:
            return len(self.generation.seeds)
        return self.generation.runs or 3
    
    def get_seed_for_run(self, run_index: int) -> Optional[int]:
        """Получить seed для конкретного прогона"""
        if self.generation.seeds and run_index < len(self.generation.seeds):
            return self.generation.seeds[run_index]
        return None


class ModelsRegistry(BaseModel):
    """
    Реестр всех моделей из models.yaml
    
    Пример YAML:
        models:
          claude:
            id: "anthropic/claude-sonnet-4"
            ...
          gpt:
            id: "openai/gpt-4.1"
            ...
    """
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    
    def get(self, key: str) -> Optional[ModelConfig]:
        """Получить модель по ключу"""
        return self.models.get(key)
    
    def keys(self) -> list[str]:
        """Список ключей моделей"""
        return list(self.models.keys())
    
    def __iter__(self):
        return iter(self.models.items())
    
    def __len__(self):
        return len(self.models)
