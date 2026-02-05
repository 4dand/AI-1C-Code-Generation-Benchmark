"""
Charts — визуализация результатов оценки SMOP

Отвечает за:
- FR-405: Графики распределения оценок, boxplot по метрикам
- Экспорт графиков в PNG, SVG, PDF форматы
- Сравнительные диаграммы моделей
- Тепловые карты корреляций
- Графики детерминизма

Все графики оптимизированы для публикации в научных статьях.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')  # Бэкенд без GUI для серверов
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from .schemas import (
    ExperimentEvaluation,
    ReportSummary,
    ModelSummary,
    TaskSummary,
    QualityStats,
)
from .statistics import StatisticsCalculator
from ..schemas.results import ExperimentResult
from ..utils.file_ops import ensure_dir


logger = logging.getLogger(__name__)


# =============================================================================
# Стили для научных публикаций
# =============================================================================

# Цветовая палитра (подходит для печати и цветового восприятия)
COLORS = {
    'primary': '#2E86AB',      # Синий
    'secondary': '#A23B72',    # Пурпурный
    'success': '#28A745',      # Зелёный
    'warning': '#FFC107',      # Жёлтый
    'danger': '#DC3545',       # Красный
    'info': '#17A2B8',         # Голубой
    'dark': '#343A40',         # Тёмно-серый
    'light': '#F8F9FA',        # Светло-серый
}

# Палитра для моделей (до 10 моделей)
MODEL_COLORS = [
    '#2E86AB', '#A23B72', '#28A745', '#FFC107', '#DC3545',
    '#17A2B8', '#6610F2', '#FD7E14', '#20C997', '#E83E8C',
]

# Палитра для метрик SMOP
SMOP_COLORS = {
    'S': '#2E86AB',  # Синий
    'M': '#28A745',  # Зелёный
    'O': '#FFC107',  # Жёлтый
    'P': '#A23B72',  # Пурпурный
    'Q': '#343A40',  # Тёмно-серый
}

# Настройки для научных публикаций
PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}


def _check_matplotlib():
    """Проверить доступность matplotlib"""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib не установлен. Установите: pip install matplotlib"
        )


def _apply_style():
    """Применить стиль для публикаций"""
    plt.rcParams.update(PUBLICATION_STYLE)


def _get_quality_color(q_value: float) -> str:
    """Получить цвет в зависимости от уровня качества"""
    if q_value >= 8:
        return COLORS['success']
    elif q_value >= 5:
        return COLORS['warning']
    return COLORS['danger']


# =============================================================================
# Класс генератора графиков
# =============================================================================

class ChartGenerator:
    """
    Генератор графиков для результатов оценки SMOP
    
    Создаёт визуализации с экспортом в PNG, SVG, PDF.
    Все графики оптимизированы для научных публикаций.
    
    Example:
        generator = ChartGenerator(evaluation, experiment, "reports/charts")
        
        # Отдельные графики
        generator.plot_smop_radar()
        generator.plot_models_comparison()
        generator.plot_scores_distribution()
        
        # Все графики сразу
        generator.generate_all()
    """
    
    def __init__(
        self,
        evaluation: ExperimentEvaluation,
        experiment: Optional[ExperimentResult] = None,
        output_dir: str = "reports/charts",
        formats: List[str] = None
    ):
        """
        Инициализация генератора
        
        Args:
            evaluation: Оценка эксперимента
            experiment: Сырые результаты (для детерминизма)
            output_dir: Директория для сохранения графиков
            formats: Форматы экспорта ["png", "svg", "pdf"]
        """
        _check_matplotlib()
        _apply_style()
        
        self.evaluation = evaluation
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.formats = formats or ["png", "svg"]
        
        # Создаём директорию
        ensure_dir(self.output_dir)
        
        # Вычисляем статистику
        self.calc = StatisticsCalculator(evaluation, experiment)
        self.summary = self.calc.calculate_summary()
        self.by_model = self.calc.aggregate_by_model()
        self.by_task = self.calc.aggregate_by_task()
        
        logger.info(f"ChartGenerator инициализирован: {self.output_dir}")
    
    def _save_figure(self, fig: plt.Figure, name: str) -> List[Path]:
        """Сохранить фигуру во всех форматах"""
        paths = []
        
        for fmt in self.formats:
            path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(path, format=fmt, dpi=300 if fmt == 'png' else None)
            paths.append(path)
            logger.debug(f"Сохранён график: {path}")
        
        plt.close(fig)
        return paths
    
    # =========================================================================
    # Основные графики
    # =========================================================================
    
    def plot_smop_radar(self) -> List[Path]:
        """
        Радарная диаграмма SMOP по моделям
        
        Показывает профиль каждой модели по метрикам S, M, O, P.
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        metrics = ['S', 'M', 'O', 'P']
        num_metrics = len(metrics)
        
        # Углы для каждой метрики
        angles = [n / float(num_metrics) * 2 * 3.14159 for n in range(num_metrics)]
        angles += angles[:1]  # Замыкаем
        
        # Рисуем для каждой модели
        for idx, model in enumerate(self.by_model):
            values = [
                model.S.mean,
                model.M.mean,
                model.O.mean,
                model.P.mean,
            ]
            values += values[:1]  # Замыкаем
            
            color = MODEL_COLORS[idx % len(MODEL_COLORS)]
            ax.plot(angles, values, 'o-', linewidth=2, label=model.model_name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        # Настройки осей
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=12, fontweight='bold')
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
        
        # Сетка и легенда
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title('Профиль SMOP по моделям', size=14, fontweight='bold', pad=20)
        
        return self._save_figure(fig, f"{self.evaluation.experiment_id}_radar")
    
    def plot_models_comparison(self) -> List[Path]:
        """
        Столбчатая диаграмма сравнения моделей
        
        Горизонтальные столбцы с метриками S, M, O, P для каждой модели.
        """
        if not self.by_model:
            return []
        
        fig, ax = plt.subplots(figsize=(10, max(4, len(self.by_model) * 1.5)))
        
        models = [m.model_name for m in self.by_model]
        metrics = ['S', 'M', 'O', 'P']
        
        y_pos = range(len(models))
        bar_height = 0.2
        
        for i, metric in enumerate(metrics):
            values = [getattr(m, metric).mean for m in self.by_model]
            offset = (i - 1.5) * bar_height
            
            bars = ax.barh(
                [y + offset for y in y_pos],
                values,
                bar_height,
                label=metric,
                color=SMOP_COLORS[metric],
                alpha=0.85
            )
            
            # Значения на столбцах
            for bar, val in zip(bars, values):
                ax.text(
                    val + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}',
                    va='center', ha='left', fontsize=8
                )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Оценка (0-10)')
        ax.set_xlim(0, 11)
        ax.legend(loc='lower right')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.title('Сравнение моделей по метрикам SMOP', fontweight='bold')
        plt.tight_layout()
        
        return self._save_figure(fig, f"{self.evaluation.experiment_id}_models_comparison")
    
    def plot_q_by_model(self) -> List[Path]:
        """
        Диаграмма интегрального Q по моделям с доверительными интервалами
        """
        if not self.by_model:
            return []
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = [m.model_name for m in self.by_model]
        q_means = [m.Q.mean for m in self.by_model]
        q_stds = [m.Q.std for m in self.by_model]
        
        x_pos = range(len(models))
        colors = [_get_quality_color(q) for q in q_means]
        
        bars = ax.bar(x_pos, q_means, yerr=q_stds, capsize=5, color=colors, alpha=0.85)
        
        # Горизонтальные линии порогов
        ax.axhline(y=8, color=COLORS['success'], linestyle='--', linewidth=1, alpha=0.7, label='Высокое (≥8)')
        ax.axhline(y=5, color=COLORS['warning'], linestyle='--', linewidth=1, alpha=0.7, label='Приемлемое (≥5)')
        
        # Значения над столбцами
        for bar, val, std in zip(bars, q_means, q_stds):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.2,
                f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10
            )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel('Интегральный показатель Q')
        ax.set_ylim(0, 11)
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.title('Интегральный показатель качества Q по моделям', fontweight='bold')
        plt.tight_layout()
        
        return self._save_figure(fig, f"{self.evaluation.experiment_id}_q_by_model")
    
    def plot_scores_distribution(self) -> List[Path]:
        """
        Гистограмма распределения оценок по каждой метрике
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = ['S', 'M', 'O', 'P']
        valid_scores = [0, 2, 4, 6, 8, 10]
        
        for ax, metric in zip(axes, metrics):
            values = self.calc.get_all_scores(metric)
            
            if not values:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric}', fontweight='bold')
                continue
            
            # Подсчёт частот
            counts = [values.count(s) for s in valid_scores]
            
            bars = ax.bar(valid_scores, counts, width=1.5, color=SMOP_COLORS[metric], alpha=0.85, edgecolor='black')
            
            # Значения над столбцами
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count),
                        ha='center', va='bottom', fontsize=9
                    )
            
            ax.set_xlabel('Оценка')
            ax.set_ylabel('Количество')
            ax.set_xticks(valid_scores)
            ax.set_title(f'{metric} — распределение оценок', fontweight='bold')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.suptitle('Распределение оценок SMOP', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return self._save_figure(fig, f"{self.evaluation.experiment_id}_distribution")
    
    def plot_boxplot_by_model(self) -> List[Path]:
        """
        Boxplot Q по моделям — показывает разброс оценок
        """
        if not self.by_model:
            return []
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Собираем данные по моделям
        data = []
        labels = []
        
        for task in self.evaluation.tasks:
            model_name = task.model_name
            for run in task.runs:
                if run.scores.Q is not None:
                    # Находим или создаём группу
                    if model_name not in labels:
                        labels.append(model_name)
                        data.append([])
                    idx = labels.index(model_name)
                    data[idx].append(run.scores.Q)
        
        if not data:
            plt.close(fig)
            return []
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Раскрашиваем боксы
        for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            box.set_facecolor(color)
            box.set_alpha(0.7)
            median.set_color('black')
            median.set_linewidth(2)
        
        # Пороговые линии
        ax.axhline(y=8, color=COLORS['success'], linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=5, color=COLORS['warning'], linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_ylabel('Интегральный показатель Q')
        ax.set_ylim(0, 11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.title('Разброс Q по моделям (boxplot)', fontweight='bold')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        return self._save_figure(fig, f"{self.evaluation.experiment_id}_boxplot")
    
    def plot_heatmap_tasks_models(self) -> List[Path]:
        """
        Тепловая карта: задачи × модели → Q
        """
        # Собираем матрицу
        task_ids = sorted(set(t.task_id for t in self.evaluation.tasks))
        model_ids = sorted(set(t.model_id for t in self.evaluation.tasks))
        
        if len(task_ids) < 2 or len(model_ids) < 2:
            # Недостаточно данных для тепловой карты
            return []
        
        matrix = [[None for _ in model_ids] for _ in task_ids]
        model_names = {}
        
        for task in self.evaluation.tasks:
            if task.avg_Q is not None:
                row = task_ids.index(task.task_id)
                col = model_ids.index(task.model_id)
                matrix[row][col] = task.avg_Q
                model_names[task.model_id] = task.model_name
        
        fig, ax = plt.subplots(figsize=(max(8, len(model_ids) * 2), max(6, len(task_ids) * 0.8)))
        
        # Создаём массив для imshow
        import numpy as np
        data = np.array([[v if v is not None else np.nan for v in row] for row in matrix])
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
        
        # Подписи осей
        ax.set_xticks(range(len(model_ids)))
        ax.set_xticklabels([model_names.get(m, m) for m in model_ids], rotation=45, ha='right')
        ax.set_yticks(range(len(task_ids)))
        ax.set_yticklabels(task_ids)
        
        # Значения в ячейках
        for i in range(len(task_ids)):
            for j in range(len(model_ids)):
                val = matrix[i][j]
                if val is not None:
                    text_color = 'white' if val < 4 or val > 7 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Q (интегральный показатель)')
        
        plt.title('Тепловая карта: Q по задачам и моделям', fontweight='bold')
        plt.xlabel('Модель')
        plt.ylabel('Задача')
        plt.tight_layout()
        
        return self._save_figure(fig, f"{self.evaluation.experiment_id}_heatmap")
    
    def plot_determinism_vs_quality(self) -> List[Path]:
        """
        Scatter plot: детерминизм vs качество
        """
        if not self.experiment:
            return []
        
        # Собираем пары (детерминизм, Q)
        points = []
        labels = []
        
        for task in self.evaluation.tasks:
            # Находим детерминизм в эксперименте
            for tr in self.experiment.task_results:
                if tr.task_id == task.task_id and tr.model_id == task.model_id:
                    if tr.determinism and task.avg_Q is not None:
                        det = tr.determinism.match_rate * 100
                        points.append((det, task.avg_Q))
                        labels.append(f"{task.task_id}\n{task.model_name[:10]}")
                    break
        
        if len(points) < 3:
            return []
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        colors = [_get_quality_color(q) for q in y]
        
        scatter = ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Подписи точек
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), fontsize=7)
        
        # Корреляция
        corr = self.calc.calculate_correlation_det_quality()
        if corr is not None:
            ax.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Детерминизм (%)')
        ax.set_ylabel('Интегральный показатель Q')
        ax.set_xlim(-5, 105)
        ax.set_ylim(0, 11)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.title('Связь детерминизма и качества генерации', fontweight='bold')
        plt.tight_layout()
        
        return self._save_figure(fig, f"{self.evaluation.experiment_id}_det_vs_quality")
    
    def plot_summary_dashboard(self) -> List[Path]:
        """
        Сводный дашборд — все ключевые метрики на одном рисунке
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Создаём сетку
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Общий Q с CI (большая ячейка)
        ax1 = fig.add_subplot(gs[0, 0])
        overall_q = self.summary.get("overall_Q", {})
        q_mean = overall_q.get("mean", 0)
        q_ci_lower = overall_q.get("ci_lower", 0)
        q_ci_upper = overall_q.get("ci_upper", 0)
        
        color = _get_quality_color(q_mean)
        ax1.bar([0], [q_mean], color=color, alpha=0.85, width=0.5)
        ax1.errorbar([0], [q_mean], yerr=[[q_mean - q_ci_lower], [q_ci_upper - q_mean]], 
                     fmt='none', color='black', capsize=10, capthick=2)
        ax1.set_xlim(-0.5, 0.5)
        ax1.set_ylim(0, 11)
        ax1.set_xticks([])
        ax1.axhline(y=8, color=COLORS['success'], linestyle='--', alpha=0.5)
        ax1.axhline(y=5, color=COLORS['warning'], linestyle='--', alpha=0.5)
        ax1.set_title(f'Q = {q_mean:.2f}\n95% CI: [{q_ci_lower:.2f}, {q_ci_upper:.2f}]', fontweight='bold')
        ax1.set_ylabel('Q')
        
        # 2. Метрики SMOP (столбцы)
        ax2 = fig.add_subplot(gs[0, 1])
        by_metric = self.summary.get("by_metric", {})
        metrics = ['S', 'M', 'O', 'P']
        means = [by_metric.get(m, {}).get("mean", 0) for m in metrics]
        stds = [by_metric.get(m, {}).get("std", 0) for m in metrics]
        colors = [SMOP_COLORS[m] for m in metrics]
        
        bars = ax2.bar(metrics, means, yerr=stds, capsize=3, color=colors, alpha=0.85)
        ax2.set_ylim(0, 11)
        ax2.set_ylabel('Оценка')
        ax2.set_title('Средние оценки SMOP', fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 3. Прогресс оценки (пирог)
        ax3 = fig.add_subplot(gs[0, 2])
        evaluated = self.summary.get("total_evaluated", 0)
        total = self.summary.get("total_runs", 1)
        remaining = total - evaluated
        
        ax3.pie([evaluated, remaining], labels=['Оценено', 'Осталось'],
                colors=[COLORS['success'], COLORS['light']], autopct='%1.0f%%',
                startangle=90, explode=(0.05, 0))
        ax3.set_title(f'Прогресс: {evaluated}/{total}', fontweight='bold')
        
        # 4. Q по моделям (горизонтальные столбцы)
        ax4 = fig.add_subplot(gs[1, :2])
        if self.by_model:
            models = [m.model_name for m in self.by_model]
            q_vals = [m.Q.mean for m in self.by_model]
            q_stds = [m.Q.std for m in self.by_model]
            colors = [_get_quality_color(q) for q in q_vals]
            
            y_pos = range(len(models))
            ax4.barh(y_pos, q_vals, xerr=q_stds, capsize=3, color=colors, alpha=0.85)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(models)
            ax4.set_xlim(0, 11)
            ax4.set_xlabel('Q')
            ax4.axvline(x=8, color=COLORS['success'], linestyle='--', alpha=0.5)
            ax4.axvline(x=5, color=COLORS['warning'], linestyle='--', alpha=0.5)
        ax4.set_title('Интегральный Q по моделям', fontweight='bold')
        ax4.grid(axis='x', linestyle='--', alpha=0.5)
        
        # 5. Детерминизм
        ax5 = fig.add_subplot(gs[1, 2])
        det = self.summary.get("determinism", {})
        det_mean = det.get("mean", 0) * 100 if det else 0
        
        ax5.pie([det_mean, 100 - det_mean], labels=['Совпадения', 'Различия'],
                colors=[COLORS['primary'], COLORS['light']], autopct='%1.0f%%',
                startangle=90)
        ax5.set_title(f'Детерминизм: {det_mean:.0f}%', fontweight='bold')
        
        # 6. Распределение Q (гистограмма)
        ax6 = fig.add_subplot(gs[2, :])
        q_values = self.calc.get_all_scores("Q")
        if q_values:
            ax6.hist(q_values, bins=10, range=(0, 10), color=COLORS['primary'], 
                     alpha=0.7, edgecolor='black')
            ax6.axvline(x=q_mean, color=COLORS['danger'], linestyle='-', linewidth=2, label=f'Среднее: {q_mean:.2f}')
            ax6.axvline(x=8, color=COLORS['success'], linestyle='--', alpha=0.7, label='Порог высокого')
            ax6.axvline(x=5, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Порог приемлемого')
            ax6.legend(loc='upper left')
        ax6.set_xlabel('Q')
        ax6.set_ylabel('Количество')
        ax6.set_title('Распределение интегрального показателя Q', fontweight='bold')
        ax6.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Сводка эксперимента: {self.evaluation.experiment_id}', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        return self._save_figure(fig, f"{self.evaluation.experiment_id}_dashboard")
    
    # =========================================================================
    # Генерация всех графиков
    # =========================================================================
    
    def generate_all(self) -> Dict[str, List[Path]]:
        """
        Сгенерировать все графики
        
        Returns:
            Словарь {название: [пути к файлам]}
        """
        results = {}
        
        charts = [
            ("dashboard", self.plot_summary_dashboard),
            ("radar", self.plot_smop_radar),
            ("models_comparison", self.plot_models_comparison),
            ("q_by_model", self.plot_q_by_model),
            ("distribution", self.plot_scores_distribution),
            ("boxplot", self.plot_boxplot_by_model),
            ("heatmap", self.plot_heatmap_tasks_models),
            ("det_vs_quality", self.plot_determinism_vs_quality),
        ]
        
        for name, func in charts:
            try:
                paths = func()
                if paths:
                    results[name] = paths
                    logger.info(f"График '{name}' сохранён")
            except Exception as e:
                logger.warning(f"Ошибка при создании графика '{name}': {e}")
        
        return results


# =============================================================================
# Удобные функции
# =============================================================================

def generate_charts(
    evaluation: ExperimentEvaluation,
    experiment: Optional[ExperimentResult] = None,
    output_dir: str = "reports/charts",
    formats: List[str] = None
) -> Dict[str, List[Path]]:
    """
    Сгенерировать все графики для эксперимента
    
    Args:
        evaluation: Оценка эксперимента
        experiment: Сырые результаты
        output_dir: Директория для сохранения
        formats: Форматы ["png", "svg", "pdf"]
        
    Returns:
        Словарь с путями к файлам
    """
    generator = ChartGenerator(evaluation, experiment, output_dir, formats)
    return generator.generate_all()


def check_matplotlib_available() -> bool:
    """Проверить доступность matplotlib"""
    return MATPLOTLIB_AVAILABLE
