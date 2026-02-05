"""
Evaluator Schemas — модели данных для модуля оценки SMOP

Содержит:
- SMOPScores: оценки S, M, O, P с расчётом Q
- RunEvaluation: оценка одного прогона
- TaskEvaluation: оценка задачи (все прогоны)
- ExperimentEvaluation: оценка эксперимента (все задачи)
- ReportSummary: итоговый отчёт с метриками
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, computed_field


# Допустимые значения оценок
VALID_SCORES = frozenset([0, 2, 4, 6, 8, 10])

# Пороги качества
QUALITY_HIGH = 8.0
QUALITY_ACCEPTABLE = 5.0


class SMOPScores(BaseModel):
    """
    Оценки SMOP для одного прогона
    
    S — Синтаксическая корректность
    M — Семантическая корректность (Meaning)
    O — Оптимальность
    P — Платформенная интеграция
    Q — Интегральный показатель (вычисляется автоматически)
    """
    S: Optional[int] = Field(default=None, description="Синтаксис (0-10)")
    M: Optional[int] = Field(default=None, description="Семантика (0-10)")
    O: Optional[int] = Field(default=None, description="Оптимальность (0-10)")
    P: Optional[int] = Field(default=None, description="Платформа (0-10)")
    
    @field_validator('S', 'M', 'O', 'P', mode='before')
    @classmethod
    def validate_score(cls, v: Any) -> Optional[int]:
        """Проверка допустимых значений оценки"""
        if v is None:
            return None
        v = int(v)
        if v not in VALID_SCORES:
            raise ValueError(f"Оценка должна быть одной из: {sorted(VALID_SCORES)}")
        return v
    
    @computed_field
    @property
    def Q(self) -> Optional[float]:
        """Интегральный показатель качества Q = (S + M + O + P) / 4"""
        scores = [self.S, self.M, self.O, self.P]
        filled = [s for s in scores if s is not None]
        if not filled:
            return None
        return sum(filled) / len(filled)
    
    @property
    def is_complete(self) -> bool:
        """Все ли оценки проставлены"""
        return all(s is not None for s in [self.S, self.M, self.O, self.P])
    
    @property
    def quality_level(self) -> Optional[Literal["high", "acceptable", "low"]]:
        """Уровень качества на основе Q"""
        if self.Q is None:
            return None
        if self.Q >= QUALITY_HIGH:
            return "high"
        if self.Q >= QUALITY_ACCEPTABLE:
            return "acceptable"
        return "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "S": self.S,
            "M": self.M,
            "O": self.O,
            "P": self.P,
            "Q": self.Q
        }


class RunEvaluation(BaseModel):
    """
    Оценка одного прогона генерации
    
    Привязывается к run_index из raw_results
    """
    run_index: int = Field(..., ge=0, description="Индекс прогона")
    response_hash: str = Field(default="", description="Хеш ответа для сверки")
    
    # Оценки SMOP
    scores: SMOPScores = Field(default_factory=SMOPScores)
    
    # Комментарий эксперта
    comment: str = Field(default="", description="Комментарий к оценке")
    
    # Метаданные оценки
    evaluated_at: Optional[str] = Field(default=None, description="Время оценки ISO")
    
    @property
    def is_evaluated(self) -> bool:
        """Прогон оценён (хотя бы одна оценка)"""
        return self.scores.Q is not None
    
    def mark_evaluated(self) -> None:
        """Отметить время оценки"""
        self.evaluated_at = datetime.now().isoformat()


class TaskEvaluation(BaseModel):
    """
    Оценка одной задачи (все прогоны одной модели)
    """
    task_id: str = Field(..., description="ID задачи")
    model_id: str = Field(..., description="ID модели")
    model_name: str = Field(default="", description="Название модели")
    
    # Прогоны
    runs: List[RunEvaluation] = Field(default_factory=list)
    
    @property
    def total_runs(self) -> int:
        """Всего прогонов"""
        return len(self.runs)
    
    @property
    def evaluated_runs(self) -> int:
        """Количество оценённых прогонов"""
        return sum(1 for r in self.runs if r.is_evaluated)
    
    @property
    def is_complete(self) -> bool:
        """Все прогоны оценены полностью"""
        return all(r.scores.is_complete for r in self.runs)
    
    @property
    def avg_Q(self) -> Optional[float]:
        """Средний Q по всем оценённым прогонам"""
        qs = [r.scores.Q for r in self.runs if r.scores.Q is not None]
        if not qs:
            return None
        return sum(qs) / len(qs)
    
    def get_run(self, run_index: int) -> Optional[RunEvaluation]:
        """Получить прогон по индексу"""
        for run in self.runs:
            if run.run_index == run_index:
                return run
        return None


class ExperimentEvaluation(BaseModel):
    """
    Оценка всего эксперимента
    
    Сохраняется в evaluations/{experiment_id}_evaluation.json
    """
    experiment_id: str = Field(..., description="ID эксперимента")
    evaluator_id: str = Field(default="expert_01", description="ID эксперта")
    framework_version: str = Field(default="1.0.0", description="Версия фреймворка")
    
    # Время работы
    started_at: Optional[str] = Field(default=None, description="Начало оценки")
    completed_at: Optional[str] = Field(default=None, description="Завершение оценки")
    last_modified_at: Optional[str] = Field(default=None, description="Последнее изменение")
    
    # Задачи
    tasks: List[TaskEvaluation] = Field(default_factory=list)
    
    # Статус
    status: Literal["not_started", "in_progress", "completed"] = Field(
        default="not_started",
        description="Статус оценки"
    )
    
    @property
    def total_runs(self) -> int:
        """Всего прогонов"""
        return sum(t.total_runs for t in self.tasks)
    
    @property
    def evaluated_runs(self) -> int:
        """Количество оценённых прогонов"""
        return sum(t.evaluated_runs for t in self.tasks)
    
    @property
    def progress_percent(self) -> float:
        """Процент прогресса оценки"""
        if self.total_runs == 0:
            return 0.0
        return (self.evaluated_runs / self.total_runs) * 100
    
    @property
    def is_complete(self) -> bool:
        """Все прогоны оценены"""
        return self.evaluated_runs == self.total_runs and self.total_runs > 0
    
    def get_task(self, task_id: str, model_id: str) -> Optional[TaskEvaluation]:
        """Найти задачу по ID и модели"""
        for task in self.tasks:
            if task.task_id == task_id and task.model_id == model_id:
                return task
        return None
    
    def update_status(self) -> None:
        """Обновить статус на основе прогресса"""
        if self.is_complete:
            self.status = "completed"
            if not self.completed_at:
                self.completed_at = datetime.now().isoformat()
        elif self.evaluated_runs > 0:
            self.status = "in_progress"
            if not self.started_at:
                self.started_at = datetime.now().isoformat()
        self.last_modified_at = datetime.now().isoformat()
    
    def start(self) -> None:
        """Начать оценку"""
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
        self.status = "in_progress"
        self.last_modified_at = datetime.now().isoformat()


class MetricStats(BaseModel):
    """Статистика по одной метрике"""
    mean: float = Field(default=0.0, description="Среднее значение")
    std: float = Field(default=0.0, description="Стандартное отклонение")
    median: float = Field(default=0.0, description="Медиана")
    min: float = Field(default=0.0, description="Минимум")
    max: float = Field(default=0.0, description="Максимум")
    count: int = Field(default=0, ge=0, description="Количество значений")


class QualityStats(BaseModel):
    """Статистика по интегральному показателю Q"""
    mean: float = Field(default=0.0)
    std: float = Field(default=0.0)
    median: float = Field(default=0.0)
    min: float = Field(default=0.0, description="Минимум")
    max: float = Field(default=0.0, description="Максимум")
    ci_lower: float = Field(default=0.0, description="Нижняя граница 95% ДИ")
    ci_upper: float = Field(default=0.0, description="Верхняя граница 95% ДИ")
    count: int = Field(default=0, ge=0)


class ModelSummary(BaseModel):
    """Сводка по модели"""
    model_id: str
    model_name: str
    tasks_count: int = Field(default=0, ge=0)
    runs_count: int = Field(default=0, ge=0)
    
    # Метрики SMOP
    S: MetricStats = Field(default_factory=MetricStats)
    M: MetricStats = Field(default_factory=MetricStats)
    O: MetricStats = Field(default_factory=MetricStats)
    P: MetricStats = Field(default_factory=MetricStats)
    Q: QualityStats = Field(default_factory=QualityStats)
    
    # Детерминизм
    determinism_mean: float = Field(default=0.0, description="Средний % детерминизма")


class TaskSummary(BaseModel):
    """Сводка по задаче"""
    task_id: str
    task_name: str
    models_count: int = Field(default=0, ge=0)
    runs_count: int = Field(default=0, ge=0)
    
    # Метрики
    Q: QualityStats = Field(default_factory=QualityStats)
    
    # По моделям
    by_model: Dict[str, QualityStats] = Field(default_factory=dict)


class ReportSummary(BaseModel):
    """
    Итоговый отчёт эксперимента
    
    Сохраняется в reports/{experiment_id}_report.json
    """
    experiment_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Метаданные
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Общие метрики
    summary: Dict[str, Any] = Field(default_factory=dict)
    
    # По моделям
    by_model: List[ModelSummary] = Field(default_factory=list)
    
    # По задачам
    by_task: List[TaskSummary] = Field(default_factory=list)
    
    # По категориям (если применимо)
    by_category: Dict[str, QualityStats] = Field(default_factory=dict)
    
    # Межэкспертная надёжность (если несколько экспертов)
    inter_rater_reliability: Optional[Dict[str, float]] = None
