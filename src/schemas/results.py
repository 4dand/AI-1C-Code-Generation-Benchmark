"""
Result Schemas - структуры для результатов
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ChatMessage:
    """Сообщение чата для API"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None  # Для assistant с tool calls
    tool_call_id: Optional[str] = None  # Для tool response


@dataclass
class GenerationResult:
    """Результат генерации от LLM API"""
    success: bool
    content: str = ""
    
    # Метрики
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    elapsed_time: float = 0.0
    
    # Информация о модели
    model_used: str = ""
    
    # Tool calls (для agentic режима)
    tool_calls: Optional[List[Dict]] = None
    
    # Ошибка
    error: Optional[str] = None
    
    # Сырой ответ (для отладки)
    raw_response: Optional[Dict] = None


@dataclass
class ContextLoadResult:
    """Результат загрузки контекста из MCP"""
    success: bool
    context_text: str = ""
    objects_loaded: List[Dict[str, str]] = field(default_factory=list)
    analysis_tokens: int = 0
    analysis_cost: float = 0.0
    error: Optional[str] = None


@dataclass 
class RunResult:
    """Результат одного прогона генерации"""
    run_index: int
    seed: Optional[int]
    temperature: float
    
    # Ответ модели
    response: str
    response_hash: str
    
    # Метрики
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    elapsed_time: float = 0.0
    
    # Стоимость
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_total: float = 0.0
    
    # Статус
    success: bool = True
    error: Optional[str] = None


@dataclass
class DeterminismResult:
    """Результат проверки детерминизма"""
    total_runs: int  # Всего прогонов
    unique_responses: int  # Количество уникальных ответов
    match_rate: float  # Процент совпадений (0.0 - 1.0)
    most_common_hash: str  # Самый частый хеш
    most_common_count: int  # Сколько раз встретился самый частый
    hashes: List[str] = field(default_factory=list)
    note: Optional[str] = None
    
    @property
    def match_percent(self) -> float:
        """Процент совпадений в удобном формате"""
        return self.match_rate * 100
    
    @property
    def is_deterministic(self) -> bool:
        """Считается детерминированным если все ответы одинаковые"""
        return self.unique_responses == 1


@dataclass
class TaskResult:
    """Результат выполнения задачи"""
    task_id: str
    task_name: str
    model_id: str
    model_name: str
    
    # Контекст (для категории B)
    context_loaded: bool = False
    context_objects: List[Dict[str, str]] = field(default_factory=list)
    context_analysis_cost: float = 0.0
    
    # Прогоны
    runs: List[RunResult] = field(default_factory=list)
    
    # Анализ детерминизма
    determinism: Optional[DeterminismResult] = None
    
    # Агрегированные метрики
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_time: float = 0.0


@dataclass
class ExperimentResult:
    """Результат всего эксперимента"""
    experiment_name: str
    category: str
    timestamp: str
    
    # Конфигурация
    models_used: List[str] = field(default_factory=list)
    tasks_count: int = 0
    runs_per_task: int = 0
    
    # Результаты по задачам
    task_results: List[TaskResult] = field(default_factory=list)
    
    # Итоги
    total_tokens: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь для сериализации"""
        from dataclasses import asdict
        return asdict(self)
