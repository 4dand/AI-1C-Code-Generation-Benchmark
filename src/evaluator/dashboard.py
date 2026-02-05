"""
Dashboard — интерактивный интерфейс для оценки SMOP

Отвечает за:
- FR-501: Интерактивный интерфейс для работы с оценками
- FR-502: Отображение списка экспериментов
- FR-503: Отображение списка задач и прогонов
- FR-504: Панель просмотра кода с подсветкой
- FR-505: Панель контекста метаданных
- FR-506: Панель оценки с валидацией
- FR-507: Отображение критериев оценки
- FR-508: Навигация между прогонами
- FR-509: Индикация прогресса
- FR-510: Сводная панель метрик
- FR-511: CLI-режим для автоматизации

NFR-04: Автосохранение каждые 30 секунд
NFR-05: Восстановление сессии
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import Progress, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.live import Live
from rich.markdown import Markdown

from ..schemas.results import ExperimentResult
from .schemas import ExperimentEvaluation, TaskEvaluation, RunEvaluation, SMOPScores, VALID_SCORES
from .parser import ExperimentParser
from .smop import SMOPEvaluator, SMOPCriteria, get_smop_criteria
from .statistics import StatisticsCalculator
from .report import ReportGenerator, generate_report


logger = logging.getLogger(__name__)
console = Console()


class EvaluatorDashboard:
    """
    Интерактивный TUI-интерфейс для оценки экспериментов
    
    Реализует пошаговую навигацию по прогонам с возможностью
    просмотра кода, контекста и проставления оценок SMOP.
    
    Example:
        dashboard = EvaluatorDashboard()
        dashboard.run("experiment_B_20260205_221310")
    """
    
    def __init__(
        self,
        results_dir: str = "raw_results",
        evaluations_dir: str = "evaluations",
        reports_dir: str = "reports"
    ):
        """
        Инициализация Dashboard
        
        Args:
            results_dir: Директория с результатами экспериментов
            evaluations_dir: Директория для сохранения оценок
            reports_dir: Директория для отчётов
        """
        self.parser = ExperimentParser(results_dir)
        self.evaluator = SMOPEvaluator(evaluations_dir)
        self.criteria = get_smop_criteria()
        self.reports_dir = reports_dir
        
        # Текущее состояние
        self.experiment: Optional[ExperimentResult] = None
        self.evaluation: Optional[ExperimentEvaluation] = None
        
        # Навигация
        self.current_task_idx: int = 0
        self.current_run_idx: int = 0
        
        # Автосохранение
        self.last_save_time: float = 0
        self.autosave_interval: int = 30  # секунд
    
    def list_experiments(self) -> None:
        """
        Показать список доступных экспериментов
        
        FR-502: Отображение списка экспериментов
        """
        experiments = self.parser.list_experiments()
        
        if not experiments:
            console.print("[yellow]Нет доступных экспериментов в raw_results/[/yellow]")
            return
        
        table = Table(title="📂 Доступные эксперименты", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="bright_white")
        table.add_column("Категория", justify="center")
        table.add_column("Модели", style="dim")
        table.add_column("Задач", justify="right")
        table.add_column("Прогонов", justify="right")
        table.add_column("Стоимость", justify="right", style="green")
        
        for exp in experiments:
            models = ", ".join(exp.get("models", [])[:2])
            if len(exp.get("models", [])) > 2:
                models += "..."
            
            table.add_row(
                exp["id"],
                exp["category"],
                models,
                str(exp.get("tasks_count", 0)),
                str(exp.get("runs_per_task", 0)),
                f"${exp.get('total_cost', 0):.4f}"
            )
        
        console.print()
        console.print(table)
        console.print()
    
    def show_status(self, experiment_id: str) -> None:
        """
        Показать статус оценки эксперимента
        
        Args:
            experiment_id: ID эксперимента
        """
        evaluations = self.evaluator.list_evaluations(experiment_id)
        
        if not evaluations:
            console.print(f"[yellow]Оценка эксперимента {experiment_id} ещё не начата[/yellow]")
            return
        
        table = Table(title=f"📊 Статус оценки: {experiment_id}", show_header=True)
        table.add_column("Эксперт", style="cyan")
        table.add_column("Статус", justify="center")
        table.add_column("Прогресс", justify="center")
        table.add_column("Оценено", justify="right")
        table.add_column("Последнее изменение", style="dim")
        
        for ev in evaluations:
            status_style = {
                "not_started": "dim",
                "in_progress": "yellow",
                "completed": "green"
            }.get(ev["status"], "white")
            
            progress_bar = self._make_progress_bar(ev["progress_percent"])
            
            table.add_row(
                ev["evaluator_id"],
                f"[{status_style}]{ev['status']}[/{status_style}]",
                progress_bar,
                f"{ev['evaluated_runs']}/{ev['total_runs']}",
                ev.get("last_modified_at", "-")[:16] if ev.get("last_modified_at") else "-"
            )
        
        console.print()
        console.print(table)
        console.print()
    
    def _make_progress_bar(self, percent: float) -> str:
        """Создать текстовую полоску прогресса"""
        filled = int(percent / 5)  # 20 символов = 100%
        empty = 20 - filled
        bar = "█" * filled + "░" * empty
        color = "green" if percent >= 80 else ("yellow" if percent >= 40 else "red")
        return f"[{color}]{bar}[/{color}] {percent:.0f}%"
    
    def run(self, experiment_id: str, evaluator_id: str = "expert_01") -> None:
        """
        Запустить интерактивную сессию оценки
        
        Args:
            experiment_id: ID эксперимента
            evaluator_id: ID эксперта
            
        FR-501: Интерактивный интерфейс
        """
        # Загружаем эксперимент
        self.experiment = self.parser.load_experiment(experiment_id)
        if not self.experiment:
            console.print(f"[red]Эксперимент не найден: {experiment_id}[/red]")
            return
        
        # Загружаем или создаём оценку
        self.evaluation = self.evaluator.load(experiment_id, evaluator_id)
        
        if self.evaluation:
            console.print(f"[green]Восстановлена сессия оценки ({self.evaluation.progress_percent:.0f}%)[/green]")
        else:
            self.evaluation = self.parser.create_evaluation(self.experiment, evaluator_id)
            self.evaluation.start()
            console.print("[cyan]Создана новая сессия оценки[/cyan]")
        
        # Находим первый неоценённый прогон
        self._find_first_unevaluated()
        
        # Основной цикл
        self._main_loop()
        
        # Сохраняем при выходе
        self._save()
        console.print("[green]✓ Оценки сохранены[/green]")
    
    def _find_first_unevaluated(self) -> None:
        """Найти первый неоценённый прогон"""
        for task_idx, task in enumerate(self.evaluation.tasks):
            for run_idx, run in enumerate(task.runs):
                if not run.scores.is_complete:
                    self.current_task_idx = task_idx
                    self.current_run_idx = run_idx
                    return
    
    def _main_loop(self) -> None:
        """Основной цикл интерфейса"""
        while True:
            self._autosave_check()
            
            # Очищаем экран
            console.clear()
            
            # Отображаем интерфейс
            self._render_header()
            self._render_current_run()
            
            # Обработка команды
            action = self._prompt_action()
            
            if action == "quit":
                break
            elif action == "next":
                self._navigate_next()
            elif action == "prev":
                self._navigate_prev()
            elif action == "save":
                self._save()
            elif action == "report":
                self._generate_report()
            elif action == "score":
                self._input_scores()
            elif action == "jump":
                self._jump_to_run()
    
    def _render_header(self) -> None:
        """Отрисовка заголовка с прогрессом"""
        task = self._get_current_task()
        run = self._get_current_run()
        
        if not task or not run:
            return
        
        # Прогресс
        progress_text = self._make_progress_bar(self.evaluation.progress_percent)
        
        header = Table.grid(expand=True)
        header.add_column(ratio=3)
        header.add_column(ratio=2, justify="right")
        
        header.add_row(
            f"[bold cyan]SMOP Evaluator[/bold cyan] │ {self.experiment.experiment_name}",
            f"Прогресс: {progress_text}"
        )
        
        console.print(Panel(header, style="blue"))
        console.print()
    
    def _render_current_run(self) -> None:
        """Отрисовка текущего прогона"""
        task = self._get_current_task()
        run = self._get_current_run()
        
        if not task or not run:
            console.print("[red]Нет прогонов для оценки[/red]")
            return
        
        # Получаем детали из эксперимента
        details = self.parser.get_run_details(
            self.experiment,
            task.task_id,
            task.model_id,
            run.run_index
        )
        
        if not details:
            console.print("[red]Не удалось загрузить детали прогона[/red]")
            return
        
        # Заголовок задачи
        console.print(Panel(
            f"[bold]Задача:[/bold] {task.task_id} │ "
            f"[bold]Модель:[/bold] {task.model_name} │ "
            f"[bold]Прогон:[/bold] {run.run_index + 1}/{len(task.runs)}",
            style="cyan"
        ))
        
        # Layout для двух панелей
        layout = Layout()
        layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Левая панель: контекст и код
        code = details.get("code", "")[:2000]  # Ограничение для TUI
        
        code_panel = Panel(
            Syntax(code, "vbnet", theme="monokai", line_numbers=True, word_wrap=True),
            title="📝 Сгенерированный код",
            border_style="green"
        )
        console.print(code_panel)
        
        # Контекст (если есть)
        if details.get("context_objects"):
            ctx_lines = []
            for obj in details["context_objects"][:5]:
                ctx_lines.append(f"• {obj.get('type', '?')}: {obj.get('name', '?')}")
            ctx_text = "\n".join(ctx_lines) if ctx_lines else "Нет контекста"
            console.print(Panel(ctx_text, title="📦 Контекст метаданных", border_style="blue"))
        
        # Панель текущих оценок
        self._render_scores_panel(run)
    
    def _render_scores_panel(self, run: RunEvaluation) -> None:
        """Отрисовка панели текущих оценок"""
        scores = run.scores
        
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Метрика", style="cyan", width=20)
        table.add_column("Оценка", justify="center", width=10)
        table.add_column("Критерий", style="dim")
        
        for metric in ["S", "M", "O", "P"]:
            score = getattr(scores, metric)
            score_str = str(score) if score is not None else "[dim]—[/dim]"
            
            criterion = ""
            if score is not None:
                criterion = self.criteria.get_criterion_description(metric, score)[:60]
                if len(self.criteria.get_criterion_description(metric, score)) > 60:
                    criterion += "..."
            
            info = self.criteria.get_metric_info(metric)
            metric_name = f"{metric} ({info.get('name', '')})"
            
            table.add_row(metric_name, score_str, criterion)
        
        # Q
        q_val = f"[bold]{scores.Q:.1f}[/bold]" if scores.Q is not None else "[dim]—[/dim]"
        quality = ""
        if scores.quality_level:
            level_colors = {"high": "green", "acceptable": "yellow", "low": "red"}
            level_names = {"high": "Высокий", "acceptable": "Приемлемый", "low": "Низкий"}
            color = level_colors.get(scores.quality_level, "white")
            quality = f"[{color}]{level_names.get(scores.quality_level, '')}[/{color}]"
        
        table.add_row("[bold]Q (Итого)[/bold]", q_val, quality)
        
        console.print(Panel(table, title="📊 Оценки SMOP", border_style="yellow"))
        
        # Комментарий
        if run.comment:
            console.print(Panel(run.comment, title="💬 Комментарий", border_style="dim"))
    
    def _prompt_action(self) -> str:
        """Запрос действия от пользователя"""
        console.print()
        console.print("[dim]Команды: [S]core  [N]ext  [P]rev  [J]ump  [R]eport  [Q]uit[/dim]")
        
        action = Prompt.ask(
            "Действие",
            choices=["s", "n", "p", "j", "r", "q", "save"],
            default="s"
        ).lower()
        
        mapping = {
            "s": "score",
            "n": "next",
            "p": "prev",
            "j": "jump",
            "r": "report",
            "q": "quit",
            "save": "save"
        }
        
        return mapping.get(action, "score")
    
    def _input_scores(self) -> None:
        """Ввод оценок для текущего прогона"""
        task = self._get_current_task()
        run = self._get_current_run()
        
        if not task or not run:
            return
        
        console.print()
        console.print("[bold cyan]Введите оценки (0, 2, 4, 6, 8, 10):[/bold cyan]")
        console.print("[dim]Нажмите Enter для пропуска[/dim]")
        console.print()
        
        scores = {}
        
        for metric in ["S", "M", "O", "P"]:
            info = self.criteria.get_metric_info(metric)
            current = getattr(run.scores, metric)
            default = str(current) if current is not None else ""
            
            # Показываем критерии
            console.print(f"[bold]{metric}[/bold]: {info.get('name', '')}")
            console.print(f"[dim]{info.get('description', '')}[/dim]")
            
            while True:
                try:
                    value = Prompt.ask(f"  Оценка {metric}", default=default)
                    
                    if not value:
                        break
                    
                    score = int(value)
                    if score not in VALID_SCORES:
                        console.print(f"[red]Допустимые значения: {sorted(VALID_SCORES)}[/red]")
                        continue
                    
                    scores[metric] = score
                    
                    # Показываем выбранный критерий
                    criterion = self.criteria.get_criterion_description(metric, score)
                    if criterion:
                        console.print(f"  [dim]→ {criterion}[/dim]")
                    break
                    
                except ValueError:
                    console.print("[red]Введите число[/red]")
        
        # Комментарий
        comment = Prompt.ask("Комментарий (опционально)", default=run.comment)
        
        # Сохраняем оценки
        if scores:
            self.evaluator.set_all_scores(
                self.evaluation,
                task.task_id,
                task.model_id,
                run.run_index,
                scores,
                comment
            )
            console.print("[green]✓ Оценки сохранены[/green]")
        
        # Автопереход к следующему
        if run.scores.is_complete:
            if Confirm.ask("Перейти к следующему прогону?", default=True):
                self._navigate_next()
    
    def _navigate_next(self) -> None:
        """Переход к следующему прогону"""
        task = self._get_current_task()
        
        if self.current_run_idx < len(task.runs) - 1:
            self.current_run_idx += 1
        elif self.current_task_idx < len(self.evaluation.tasks) - 1:
            self.current_task_idx += 1
            self.current_run_idx = 0
        else:
            console.print("[yellow]Достигнут конец списка[/yellow]")
    
    def _navigate_prev(self) -> None:
        """Переход к предыдущему прогону"""
        if self.current_run_idx > 0:
            self.current_run_idx -= 1
        elif self.current_task_idx > 0:
            self.current_task_idx -= 1
            task = self._get_current_task()
            self.current_run_idx = len(task.runs) - 1
        else:
            console.print("[yellow]Достигнуто начало списка[/yellow]")
    
    def _jump_to_run(self) -> None:
        """Переход к конкретному прогону"""
        # Показываем список задач
        table = Table(title="Задачи", show_header=True)
        table.add_column("#", justify="right")
        table.add_column("ID")
        table.add_column("Модель")
        table.add_column("Прогонов")
        table.add_column("Оценено")
        
        for idx, task in enumerate(self.evaluation.tasks):
            table.add_row(
                str(idx + 1),
                task.task_id,
                task.model_name,
                str(task.total_runs),
                f"{task.evaluated_runs}/{task.total_runs}"
            )
        
        console.print(table)
        
        try:
            task_num = IntPrompt.ask("Номер задачи", default=self.current_task_idx + 1)
            if 1 <= task_num <= len(self.evaluation.tasks):
                self.current_task_idx = task_num - 1
                self.current_run_idx = 0
            
            task = self._get_current_task()
            run_num = IntPrompt.ask("Номер прогона", default=1)
            if 1 <= run_num <= len(task.runs):
                self.current_run_idx = run_num - 1
                
        except ValueError:
            pass
    
    def _save(self) -> None:
        """Сохранить оценки"""
        if self.evaluation:
            self.evaluator.save(self.evaluation)
            self.last_save_time = time.time()
    
    def _autosave_check(self) -> None:
        """Проверка автосохранения"""
        if time.time() - self.last_save_time > self.autosave_interval:
            self._save()
    
    def _generate_report(self) -> None:
        """Генерация отчёта"""
        if not self.evaluation.is_complete:
            if not Confirm.ask("[yellow]Оценка не завершена. Сгенерировать частичный отчёт?[/yellow]"):
                return
        
        paths = generate_report(
            self.evaluation,
            self.experiment,
            self.reports_dir,
            formats=["json", "html"]
        )
        
        console.print("[green]✓ Отчёты сгенерированы:[/green]")
        for fmt, path in paths.items():
            console.print(f"  {fmt}: {path}")
    
    def _get_current_task(self) -> Optional[TaskEvaluation]:
        """Получить текущую задачу"""
        if 0 <= self.current_task_idx < len(self.evaluation.tasks):
            return self.evaluation.tasks[self.current_task_idx]
        return None
    
    def _get_current_run(self) -> Optional[RunEvaluation]:
        """Получить текущий прогон"""
        task = self._get_current_task()
        if task and 0 <= self.current_run_idx < len(task.runs):
            return task.runs[self.current_run_idx]
        return None


def run_dashboard(
    experiment_id: str,
    evaluator_id: str = "expert_01",
    results_dir: str = "raw_results",
    evaluations_dir: str = "evaluations",
    reports_dir: str = "reports"
) -> None:
    """
    Запустить интерактивную сессию оценки
    
    Args:
        experiment_id: ID эксперимента
        evaluator_id: ID эксперта
        results_dir: Директория с результатами
        evaluations_dir: Директория для оценок
        reports_dir: Директория для отчётов
    """
    dashboard = EvaluatorDashboard(results_dir, evaluations_dir, reports_dir)
    dashboard.run(experiment_id, evaluator_id)


def list_experiments_cli(results_dir: str = "raw_results") -> None:
    """Показать список экспериментов в CLI"""
    dashboard = EvaluatorDashboard(results_dir)
    dashboard.list_experiments()


def show_status_cli(experiment_id: str, evaluations_dir: str = "evaluations") -> None:
    """Показать статус оценки в CLI"""
    dashboard = EvaluatorDashboard(evaluations_dir=evaluations_dir)
    dashboard.show_status(experiment_id)
