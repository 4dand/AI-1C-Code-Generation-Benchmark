"""
Statistics — статистический анализ результатов оценки

Отвечает за:
- FR-301: Расчёт среднего, медианы, стандартного отклонения
- FR-302: Расчёт доверительного интервала (95%) для Q
- FR-303: Агрегация по моделям
- FR-304: Агрегация по категориям задач
- FR-305: Агрегация по условиям (baseline vs MCP)
- FR-306: Корреляция между детерминизмом и качеством
- FR-307: Межэкспертная надёжность (Cohen's Kappa)
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .schemas import (
    ExperimentEvaluation,
    TaskEvaluation,
    RunEvaluation,
    SMOPScores,
    MetricStats,
    QualityStats,
    ModelSummary,
    TaskSummary,
)
from ..schemas.results import ExperimentResult


logger = logging.getLogger(__name__)


def calculate_mean(values: List[float]) -> float:
    """Среднее арифметическое"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_median(values: List[float]) -> float:
    """Медиана"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def calculate_std(values: List[float], mean: Optional[float] = None) -> float:
    """Стандартное отклонение (несмещённая оценка)"""
    if len(values) < 2:
        return 0.0
    if mean is None:
        mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def calculate_ci_95(values: List[float]) -> Tuple[float, float]:
    """
    Расчёт 95% доверительного интервала
    
    Использует t-распределение для малых выборок
    
    FR-302: Расчёт доверительного интервала
    """
    n = len(values)
    if n < 2:
        mean = calculate_mean(values)
        return (mean, mean)
    
    mean = calculate_mean(values)
    std = calculate_std(values, mean)
    
    # t-критерий для 95% ДИ (приближение для n >= 2)
    # Точные значения: n=2: 12.71, n=3: 4.30, n=5: 2.78, n=10: 2.26, n=30: 2.04
    t_critical = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145,
        20: 2.093, 25: 2.064, 30: 2.045, 50: 2.009, 100: 1.984
    }
    
    # Выбираем ближайшее значение
    t = 1.96  # По умолчанию для больших выборок
    for k in sorted(t_critical.keys()):
        if n <= k:
            t = t_critical[k]
            break
    
    margin = t * (std / math.sqrt(n))
    
    return (mean - margin, mean + margin)


def calculate_metric_stats(values: List[float]) -> MetricStats:
    """
    Расчёт статистики для одной метрики
    
    FR-301: Расчёт среднего, медианы, стандартного отклонения
    """
    if not values:
        return MetricStats()
    
    mean = calculate_mean(values)
    
    return MetricStats(
        mean=round(mean, 3),
        std=round(calculate_std(values, mean), 3),
        median=round(calculate_median(values), 3),
        min=min(values),
        max=max(values),
        count=len(values)
    )


def calculate_quality_stats(values: List[float]) -> QualityStats:
    """
    Расчёт статистики для Q с доверительным интервалом
    """
    if not values:
        return QualityStats()
    
    mean = calculate_mean(values)
    ci_lower, ci_upper = calculate_ci_95(values)
    
    return QualityStats(
        mean=round(mean, 3),
        std=round(calculate_std(values, mean), 3),
        median=round(calculate_median(values), 3),
        min=round(min(values), 3),
        max=round(max(values), 3),
        ci_lower=round(ci_lower, 3),
        ci_upper=round(ci_upper, 3),
        count=len(values)
    )


class StatisticsCalculator:
    """
    Калькулятор статистики для результатов оценки
    
    Использует данные оценок и сырые результаты эксперимента
    для расчёта агрегированных метрик.
    
    Example:
        calc = StatisticsCalculator(evaluation, experiment)
        
        summary = calc.calculate_summary()
        by_model = calc.aggregate_by_model()
        by_task = calc.aggregate_by_task()
    """
    
    def __init__(
        self,
        evaluation: ExperimentEvaluation,
        experiment: Optional[ExperimentResult] = None
    ):
        """
        Инициализация калькулятора
        
        Args:
            evaluation: Оценка эксперимента с проставленными SMOP
            experiment: Сырые результаты (для детерминизма)
        """
        self.evaluation = evaluation
        self.experiment = experiment
        
        # Кэш для детерминизма
        self._determinism_cache: Dict[Tuple[str, str], float] = {}
        if experiment:
            self._build_determinism_cache()
    
    def _build_determinism_cache(self) -> None:
        """Построить кэш детерминизма из результатов эксперимента"""
        if not self.experiment:
            return
        
        for task_result in self.experiment.task_results:
            key = (task_result.task_id, task_result.model_id)
            if task_result.determinism:
                self._determinism_cache[key] = task_result.determinism.match_rate
    
    def get_all_scores(self, metric: str = "Q") -> List[float]:
        """Получить все оценки по метрике"""
        values = []
        
        for task in self.evaluation.tasks:
            for run in task.runs:
                if metric == "Q":
                    if run.scores.Q is not None:
                        values.append(run.scores.Q)
                else:
                    score = getattr(run.scores, metric, None)
                    if score is not None:
                        values.append(float(score))
        
        return values
    
    def calculate_summary(self) -> Dict[str, Any]:
        """
        Рассчитать общую сводку по эксперименту
        
        Returns:
            Словарь с агрегированными метриками
            
        FR-301: Расчёт статистики
        """
        summary = {
            "overall_Q": calculate_quality_stats(self.get_all_scores("Q")).model_dump(),
            "by_metric": {
                "S": calculate_metric_stats(self.get_all_scores("S")).model_dump(),
                "M": calculate_metric_stats(self.get_all_scores("M")).model_dump(),
                "O": calculate_metric_stats(self.get_all_scores("O")).model_dump(),
                "P": calculate_metric_stats(self.get_all_scores("P")).model_dump(),
            },
            "total_evaluated": self.evaluation.evaluated_runs,
            "total_runs": self.evaluation.total_runs,
        }
        
        # Добавляем статистику детерминизма если есть
        if self._determinism_cache:
            det_values = list(self._determinism_cache.values())
            summary["determinism"] = {
                "mean": round(calculate_mean(det_values), 3),
                "std": round(calculate_std(det_values), 3),
            }
        
        return summary
    
    def aggregate_by_model(self) -> List[ModelSummary]:
        """
        Агрегация метрик по моделям
        
        Returns:
            Список ModelSummary для каждой модели
            
        FR-303: Агрегация по моделям
        """
        # Группируем по модели
        by_model: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"S": [], "M": [], "O": [], "P": [], "Q": [], "det": []}
        )
        model_names: Dict[str, str] = {}
        model_tasks: Dict[str, set] = defaultdict(set)
        
        for task in self.evaluation.tasks:
            model_id = task.model_id
            model_names[model_id] = task.model_name
            model_tasks[model_id].add(task.task_id)
            
            det_key = (task.task_id, model_id)
            if det_key in self._determinism_cache:
                by_model[model_id]["det"].append(self._determinism_cache[det_key])
            
            for run in task.runs:
                if run.scores.S is not None:
                    by_model[model_id]["S"].append(float(run.scores.S))
                if run.scores.M is not None:
                    by_model[model_id]["M"].append(float(run.scores.M))
                if run.scores.O is not None:
                    by_model[model_id]["O"].append(float(run.scores.O))
                if run.scores.P is not None:
                    by_model[model_id]["P"].append(float(run.scores.P))
                if run.scores.Q is not None:
                    by_model[model_id]["Q"].append(run.scores.Q)
        
        # Формируем результат
        summaries = []
        
        for model_id, scores in by_model.items():
            summary = ModelSummary(
                model_id=model_id,
                model_name=model_names.get(model_id, model_id),
                tasks_count=len(model_tasks[model_id]),
                runs_count=len(scores["Q"]),
                S=calculate_metric_stats(scores["S"]),
                M=calculate_metric_stats(scores["M"]),
                O=calculate_metric_stats(scores["O"]),
                P=calculate_metric_stats(scores["P"]),
                Q=calculate_quality_stats(scores["Q"]),
                determinism_mean=round(calculate_mean(scores["det"]) * 100, 1) if scores["det"] else 0.0,
            )
            summaries.append(summary)
        
        return summaries
    
    def aggregate_by_task(self) -> List[TaskSummary]:
        """
        Агрегация метрик по задачам
        
        Returns:
            Список TaskSummary для каждой задачи
            
        FR-304: Агрегация по задачам
        """
        # Группируем по задаче
        by_task: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"Q": [], "models": defaultdict(list)}
        )
        task_names: Dict[str, str] = {}
        
        for task in self.evaluation.tasks:
            task_id = task.task_id
            # Берём имя из первого вхождения
            if task_id not in task_names:
                task_names[task_id] = task_id  # Имя возьмём из experiment если есть
            
            for run in task.runs:
                if run.scores.Q is not None:
                    by_task[task_id]["Q"].append(run.scores.Q)
                    by_task[task_id]["models"][task.model_id].append(run.scores.Q)
        
        # Имена задач из эксперимента
        if self.experiment:
            for tr in self.experiment.task_results:
                task_names[tr.task_id] = tr.task_name
        
        # Формируем результат
        summaries = []
        
        for task_id, data in by_task.items():
            by_model = {}
            for model_id, qs in data["models"].items():
                by_model[model_id] = calculate_quality_stats(qs)
            
            summary = TaskSummary(
                task_id=task_id,
                task_name=task_names.get(task_id, task_id),
                models_count=len(data["models"]),
                runs_count=len(data["Q"]),
                Q=calculate_quality_stats(data["Q"]),
                by_model=by_model,
            )
            summaries.append(summary)
        
        return summaries
    
    def calculate_correlation_det_quality(self) -> Optional[float]:
        """
        Рассчитать корреляцию между детерминизмом и качеством
        
        Returns:
            Коэффициент корреляции Пирсона или None
            
        FR-306: Корреляция детерминизм-качество
        """
        if not self._determinism_cache:
            return None
        
        pairs = []
        
        for task in self.evaluation.tasks:
            det_key = (task.task_id, task.model_id)
            if det_key not in self._determinism_cache:
                continue
            
            det = self._determinism_cache[det_key]
            
            if task.avg_Q is not None:
                pairs.append((det, task.avg_Q))
        
        if len(pairs) < 3:
            return None
        
        # Коэффициент корреляции Пирсона
        n = len(pairs)
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        
        if denom_x == 0 or denom_y == 0:
            return None
        
        return round(numerator / (denom_x * denom_y), 3)
    
    def calculate_inter_rater_reliability(
        self,
        other_evaluation: ExperimentEvaluation
    ) -> Optional[Dict[str, float]]:
        """
        Рассчитать межэкспертную надёжность (Cohen's Kappa)
        
        Args:
            other_evaluation: Оценка другого эксперта
            
        Returns:
            Словарь с коэффициентами или None
            
        FR-307: Inter-rater reliability
        """
        # Собираем пары оценок
        pairs_by_metric = {"S": [], "M": [], "O": [], "P": []}
        
        for task in self.evaluation.tasks:
            other_task = other_evaluation.get_task(task.task_id, task.model_id)
            if not other_task:
                continue
            
            for run in task.runs:
                other_run = other_task.get_run(run.run_index)
                if not other_run:
                    continue
                
                for metric in ["S", "M", "O", "P"]:
                    score1 = getattr(run.scores, metric)
                    score2 = getattr(other_run.scores, metric)
                    if score1 is not None and score2 is not None:
                        pairs_by_metric[metric].append((score1, score2))
        
        # Расчёт Cohen's Kappa для каждой метрики
        result = {}
        
        for metric, pairs in pairs_by_metric.items():
            if len(pairs) < 5:  # Минимум для надёжного расчёта
                continue
            
            kappa = self._cohens_kappa(pairs)
            if kappa is not None:
                result[metric] = round(kappa, 3)
        
        return result if result else None
    
    def _cohens_kappa(self, pairs: List[Tuple[int, int]]) -> Optional[float]:
        """Расчёт Cohen's Kappa для пар оценок"""
        if len(pairs) < 2:
            return None
        
        n = len(pairs)
        categories = sorted(set(p[0] for p in pairs) | set(p[1] for p in pairs))
        k = len(categories)
        
        if k < 2:
            return 1.0  # Полное согласие
        
        # Матрица совпадений
        cat_idx = {c: i for i, c in enumerate(categories)}
        matrix = [[0] * k for _ in range(k)]
        
        for r1, r2 in pairs:
            i, j = cat_idx[r1], cat_idx[r2]
            matrix[i][j] += 1
        
        # Наблюдаемое согласие
        p_o = sum(matrix[i][i] for i in range(k)) / n
        
        # Ожидаемое случайное согласие
        row_sums = [sum(matrix[i]) for i in range(k)]
        col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
        p_e = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)
        
        if p_e == 1:
            return 1.0
        
        kappa = (p_o - p_e) / (1 - p_e)
        return kappa


def calculate_experiment_statistics(
    evaluation: ExperimentEvaluation,
    experiment: Optional[ExperimentResult] = None
) -> Dict[str, Any]:
    """
    Удобная функция для расчёта всей статистики
    
    Args:
        evaluation: Оценка эксперимента
        experiment: Сырые результаты (опционально)
        
    Returns:
        Полный набор статистики
    """
    calc = StatisticsCalculator(evaluation, experiment)
    
    return {
        "summary": calc.calculate_summary(),
        "by_model": [m.model_dump() for m in calc.aggregate_by_model()],
        "by_task": [t.model_dump() for t in calc.aggregate_by_task()],
        "correlation_det_quality": calc.calculate_correlation_det_quality(),
    }
