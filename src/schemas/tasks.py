"""
Task Schemas — схемы для задач из tasks_category_*.yaml

Отвечает за:
- Валидацию структуры задач
- Конфигурацию категорий (A/B)
- Параметры генерации специфичные для категории
"""

from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TaskConfig(BaseModel):
    """
    Конфигурация одной задачи
    
    Пример YAML:
        - id: "A1"
          name: "Сортировка пузырьком"
          difficulty: "easy"
          prompt: |
            Напишите функцию сортировки...
    """
    id: str = Field(..., pattern=r"^[A-Z]\d+$", description="ID задачи (например A1, B2)")
    name: str = Field(..., min_length=1)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    prompt: str = Field(..., min_length=10, description="Текст задания для модели")
    
    # Дополнительные поля для категории B
    expected_objects: List[str] = Field(
        default_factory=list,
        description="Ожидаемые объекты метаданных (для валидации)"
    )


class CategoryGenerationOverrides(BaseModel):
    """
    Переопределение параметров генерации для категории
    
    Позволяет задать специфичные настройки для категории,
    которые переопределяют глобальные дефолты из settings.yaml
    """
    max_tokens: Optional[int] = Field(default=None, ge=256, le=32768)
    
    # Переопределения для конкретных моделей
    model_params: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Параметры для конкретных моделей: {claude: {temperature: 0.0}}"
    )
    
    def get_model_params(self, model_key: str) -> Dict[str, Any]:
        """Получить параметры для конкретной модели"""
        return self.model_params.get(model_key, {})


class CategoryConfig(BaseModel):
    """
    Конфигурация категории задач
    
    Пример YAML:
        category:
          id: "A"
          name: "Алгоритмические задачи"
          requires_mcp: false
    """
    id: Literal["A", "B"] = Field(..., description="Идентификатор категории")
    name: str = Field(..., min_length=1)
    description: str = ""
    requires_mcp: bool = Field(
        default=False,
        description="Требуется ли MCP сервер для сбора контекста"
    )


class TasksFile(BaseModel):
    """
    Схема файла tasks_category_*.yaml
    
    Объединяет:
    - Метаданные категории
    - Параметры генерации
    - Системный промпт
    - Список задач
    """
    category: CategoryConfig
    generation: CategoryGenerationOverrides = Field(default_factory=CategoryGenerationOverrides)
    system_prompt: str = Field(..., min_length=10, description="Системный промпт для LLM")
    tasks: List[TaskConfig] = Field(..., min_length=1)
    
    @field_validator("tasks")
    @classmethod
    def validate_unique_ids(cls, v: List[TaskConfig]) -> List[TaskConfig]:
        """Проверить уникальность ID задач"""
        ids = [task.id for task in v]
        if len(ids) != len(set(ids)):
            raise ValueError("ID задач должны быть уникальными")
        return v
    
    def get_task(self, task_id: str) -> Optional[TaskConfig]:
        """Получить задачу по ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[TaskConfig]:
        """Получить задачи по сложности"""
        return [t for t in self.tasks if t.difficulty == difficulty]
    
    @property
    def task_ids(self) -> List[str]:
        """Список ID всех задач"""
        return [t.id for t in self.tasks]
