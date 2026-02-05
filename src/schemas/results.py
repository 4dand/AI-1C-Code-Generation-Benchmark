"""
Result Schemas - схемы для результатов бенчмарка

Отвечает за:
- Результаты отдельных прогонов (RunResult)
- Результаты задач (TaskResult)
- Результаты экспериментов (ExperimentResult)
- Анализ детерминизма (DeterminismResult)
- Результаты загрузки контекста (ContextLoadResult)
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, computed_field


class RunResult(BaseModel):
    """
    Результат одного прогона генерации
    
    Один прогон = один вызов LLM с конкретным seed/temperature
    """
    run_index: int = Field(..., ge=0)
    seed: Optional[int] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    
    # Ответ модели
    response: str = ""
    response_hash: str = ""
    
    # Метрики токенов
    tokens_input: int = Field(default=0, ge=0)
    tokens_output: int = Field(default=0, ge=0)
    tokens_total: int = Field(default=0, ge=0)
    
    # Время
    elapsed_time: float = Field(default=0.0, ge=0.0)
    
    # Стоимость
    cost_input: float = Field(default=0.0, ge=0.0)
    cost_output: float = Field(default=0.0, ge=0.0)
    cost_total: float = Field(default=0.0, ge=0.0)
    
    # Статус
    success: bool = True
    error: Optional[str] = None


class DeterminismResult(BaseModel):
    """
    Результат анализа детерминизма модели
    
    Сравнивает хеши ответов между прогонами
    """
    total_runs: int = Field(..., ge=1, description="Всего прогонов")
    unique_responses: int = Field(..., ge=1, description="Количество уникальных ответов")
    match_rate: float = Field(..., ge=0.0, le=1.0, description="Доля совпадений (0.0-1.0)")
    most_common_hash: str = Field(..., description="Самый частый хеш")
    most_common_count: int = Field(..., ge=1, description="Сколько раз встретился")
    hashes: List[str] = Field(default_factory=list, description="Все хеши по порядку")
    note: Optional[str] = None
    
    @computed_field
    @property
    def match_percent(self) -> float:
        """Процент совпадений (0-100)"""
        return self.match_rate * 100
    
    @computed_field
    @property
    def is_deterministic(self) -> bool:
        """Все ответы одинаковые?"""
        return self.unique_responses == 1


class ContextLoadResult(BaseModel):
    """
    Результат загрузки контекста из MCP (для категории B)
    
    Содержит собранный контекст и метрики агента
    """
    success: bool = True
    context_text: str = ""
    objects_loaded: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Загруженные объекты: [{type: 'Справочник', name: 'Номенклатура'}]"
    )
    
    # Метрики агента
    analysis_tokens: int = Field(default=0, ge=0)
    analysis_cost: float = Field(default=0.0, ge=0.0)
    iterations_count: int = Field(default=0, ge=0)
    
    # Ошибка
    error: Optional[str] = None
    
    @property
    def objects_count(self) -> int:
        """Количество загруженных объектов"""
        return len(self.objects_loaded)


class TaskResult(BaseModel):
    """
    Результат выполнения одной задачи
    
    Содержит все прогоны и агрегированные метрики
    """
    task_id: str
    task_name: str
    model_id: str
    model_name: str
    
    # Контекст (для категории B)
    context_loaded: bool = False
    context_objects: List[Dict[str, str]] = Field(default_factory=list)
    context_analysis_cost: float = Field(default=0.0, ge=0.0)
    
    # Прогоны
    runs: List[RunResult] = Field(default_factory=list)
    
    # Анализ детерминизма
    determinism: Optional[DeterminismResult] = None
    
    # Агрегированные метрики
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)
    avg_time: float = Field(default=0.0, ge=0.0)
    
    def calculate_aggregates(self) -> None:
        """Пересчитать агрегированные метрики из runs"""
        if not self.runs:
            return
        
        self.total_tokens = sum(r.tokens_total for r in self.runs)
        self.total_cost = sum(r.cost_total for r in self.runs) + self.context_analysis_cost
        self.avg_time = sum(r.elapsed_time for r in self.runs) / len(self.runs)
    
    @property
    def runs_count(self) -> int:
        """Количество прогонов"""
        return len(self.runs)
    
    @property
    def successful_runs(self) -> int:
        """Количество успешных прогонов"""
        return sum(1 for r in self.runs if r.success)


class ExperimentResult(BaseModel):
    """
    Результат всего эксперимента
    
    Содержит все результаты задач и общие метрики
    """
    experiment_name: str
    category: str = Field(..., pattern=r"^[AB]$")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Конфигурация
    models_used: List[str] = Field(default_factory=list)
    tasks_count: int = Field(default=0, ge=0)
    runs_per_task: int = Field(default=0, ge=0)
    
    # Результаты по задачам
    task_results: List[TaskResult] = Field(default_factory=list)
    
    # Итоги
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)
    total_time: float = Field(default=0.0, ge=0.0)
    
    def calculate_totals(self) -> None:
        """Пересчитать итоговые метрики"""
        self.total_tokens = sum(t.total_tokens for t in self.task_results)
        self.total_cost = sum(t.total_cost for t in self.task_results)
        self.total_time = sum(t.avg_time * t.runs_count for t in self.task_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь для сериализации"""
        return self.model_dump()
    
    @property
    def avg_determinism(self) -> float:
        """Средний процент детерминизма по всем задачам"""
        results_with_det = [t for t in self.task_results if t.determinism]
        if not results_with_det:
            return 0.0
        return sum(t.determinism.match_percent for t in results_with_det) / len(results_with_det)
