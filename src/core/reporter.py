"""
Reporter — модуль для анализа и отчётности по результатам экспериментов

Функции:
- Парсинг JSON результатов
- Агрегация статистики
- Форматирование отчётов
- Сравнение моделей
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class RunStats:
    """Статистика одного прогона"""
    run_index: int
    seed: int
    success: bool
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    elapsed_time: float = 0.0
    cost_total: float = 0.0
    response_hash: str = ""
    error: Optional[str] = None


@dataclass
class TaskStats:
    """Статистика по задаче"""
    task_id: str
    task_name: str
    model_id: str
    model_name: str
    category: str = ""
    
    # Результаты прогонов
    runs: List[RunStats] = field(default_factory=list)
    
    # Детерминизм
    determinism_rate: float = 0.0
    unique_responses: int = 0
    
    # Контекст (для категории B)
    context_loaded: bool = False
    context_objects: List[str] = field(default_factory=list)
    context_cost: float = 0.0
    
    # Агрегаты
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_time: float = 0.0
    success_rate: float = 0.0


@dataclass
class ExperimentStats:
    """Статистика эксперимента"""
    experiment_file: str
    category: str
    timestamp: str
    
    # Задачи
    tasks: List[TaskStats] = field(default_factory=list)
    
    # Модели
    models_used: List[str] = field(default_factory=list)
    
    # Общие метрики
    total_tokens: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    
    # Агрегаты
    avg_determinism: float = 0.0
    avg_success_rate: float = 0.0


class ResultsParser:
    """Парсер JSON результатов экспериментов"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
    
    def list_experiments(self, category: str = None) -> List[Path]:
        """Получить список файлов экспериментов"""
        if not self.results_dir.exists():
            return []
        
        files = sorted(self.results_dir.glob("experiment_*.json"))
        
        if category:
            files = [f for f in files if f"_{category}_" in f.name]
        
        return files
    
    def parse_experiment(self, filepath: Path) -> Optional[ExperimentStats]:
        """Распарсить один эксперимент"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Reporter] Error reading {filepath}: {e}")
            return None
        
        experiment = ExperimentStats(
            experiment_file=filepath.name,
            category=data.get("category", "?"),
            timestamp=data.get("timestamp", ""),
            models_used=data.get("models_used", []),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            total_time=data.get("total_time", 0.0)
        )
        
        # Парсим задачи
        for task_data in data.get("task_results", []):
            task = self._parse_task(task_data, experiment.category)
            experiment.tasks.append(task)
        
        # Считаем агрегаты
        if experiment.tasks:
            experiment.avg_determinism = sum(t.determinism_rate for t in experiment.tasks) / len(experiment.tasks)
            experiment.avg_success_rate = sum(t.success_rate for t in experiment.tasks) / len(experiment.tasks)
        
        return experiment
    
    def _parse_task(self, data: Dict, category: str) -> TaskStats:
        """Распарсить задачу"""
        task = TaskStats(
            task_id=data.get("task_id", ""),
            task_name=data.get("task_name", ""),
            model_id=data.get("model_id", ""),
            model_name=data.get("model_name", ""),
            category=category,
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            avg_time=data.get("avg_time", 0.0)
        )
        
        # Контекст (категория B)
        task.context_loaded = data.get("context_loaded", False)
        task.context_cost = data.get("context_analysis_cost", 0.0)
        context_objs = data.get("context_objects", [])
        task.context_objects = [obj.get("name", "") for obj in context_objs if obj.get("name")]
        
        # Детерминизм
        det = data.get("determinism", {})
        task.determinism_rate = det.get("match_rate", 0.0)
        task.unique_responses = det.get("unique_responses", 0)
        
        # Прогоны
        for run_data in data.get("runs", []):
            run = RunStats(
                run_index=run_data.get("run_index", 0),
                seed=run_data.get("seed", 0),
                success=run_data.get("success", False),
                tokens_input=run_data.get("tokens_input", 0),
                tokens_output=run_data.get("tokens_output", 0),
                tokens_total=run_data.get("tokens_total", 0),
                elapsed_time=run_data.get("elapsed_time", 0.0),
                cost_total=run_data.get("cost_total", 0.0),
                response_hash=run_data.get("response_hash", ""),
                error=run_data.get("error")
            )
            task.runs.append(run)
        
        # Success rate
        if task.runs:
            successful = sum(1 for r in task.runs if r.success)
            task.success_rate = successful / len(task.runs)
        
        return task


class ReportFormatter:
    """Форматирование отчётов"""
    
    @staticmethod
    def format_cost(cost: float) -> str:
        """Форматировать стоимость"""
        if cost < 0.01:
            return f"${cost:.6f}"
        elif cost < 1:
            return f"${cost:.4f}"
        else:
            return f"${cost:.2f}"
    
    @staticmethod
    def format_tokens(tokens: int) -> str:
        """Форматировать токены"""
        if tokens >= 1_000_000:
            return f"{tokens/1_000_000:.2f}M"
        elif tokens >= 1_000:
            return f"{tokens/1_000:.1f}K"
        return str(tokens)
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Форматировать время"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def format_percent(value: float) -> str:
        """Форматировать процент"""
        return f"{value * 100:.1f}%"
    
    @staticmethod
    def determinism_emoji(rate: float) -> str:
        """Эмодзи для уровня детерминизма"""
        if rate >= 1.0:
            return "🟢"  # 100%
        elif rate >= 0.8:
            return "🟡"  # 80%+
        elif rate >= 0.5:
            return "🟠"  # 50%+
        else:
            return "🔴"  # <50%
    
    @staticmethod
    def success_emoji(rate: float) -> str:
        """Эмодзи для success rate"""
        if rate >= 1.0:
            return "✅"
        elif rate >= 0.8:
            return "⚠️"
        else:
            return "❌"


class ExperimentReporter:
    """Генератор отчётов по экспериментам"""
    
    def __init__(self, results_dir: str = "results"):
        self.parser = ResultsParser(results_dir)
        self.fmt = ReportFormatter()
    
    def print_experiment_summary(self, experiment: ExperimentStats):
        """Вывести краткую сводку по эксперименту"""
        print()
        print("=" * 70)
        print(f"📊 ОТЧЁТ: {experiment.experiment_file}")
        print("=" * 70)
        print()
        
        # Основная информация
        print(f"  Категория:     {experiment.category}")
        print(f"  Дата:          {experiment.timestamp[:19] if experiment.timestamp else 'N/A'}")
        print(f"  Модели:        {', '.join(experiment.models_used)}")
        print(f"  Задач:         {len(experiment.tasks)}")
        print()
        
        # Общие метрики
        print("─" * 70)
        print("  📈 ОБЩИЕ МЕТРИКИ")
        print("─" * 70)
        print(f"  Токены:        {self.fmt.format_tokens(experiment.total_tokens)}")
        print(f"  Стоимость:     {self.fmt.format_cost(experiment.total_cost)}")
        print(f"  Время:         {self.fmt.format_time(experiment.total_time)}")
        print(f"  Детерминизм:   {self.fmt.format_percent(experiment.avg_determinism)} {self.fmt.determinism_emoji(experiment.avg_determinism)}")
        print(f"  Success Rate:  {self.fmt.format_percent(experiment.avg_success_rate)} {self.fmt.success_emoji(experiment.avg_success_rate)}")
        print()
    
    def print_task_details(self, task: TaskStats):
        """Вывести детали по задаче"""
        print("─" * 70)
        print(f"  📝 {task.task_id}: {task.task_name}")
        print("─" * 70)
        print(f"     Модель:        {task.model_name} ({task.model_id})")
        print(f"     Токены:        {self.fmt.format_tokens(task.total_tokens)}")
        print(f"     Стоимость:     {self.fmt.format_cost(task.total_cost)}")
        print(f"     Ср. время:     {self.fmt.format_time(task.avg_time)}")
        print(f"     Детерминизм:   {self.fmt.format_percent(task.determinism_rate)} ({task.unique_responses} уник.)")
        print(f"     Success Rate:  {self.fmt.format_percent(task.success_rate)} ({sum(1 for r in task.runs if r.success)}/{len(task.runs)})")
        
        if task.context_loaded:
            print(f"     Контекст:      {', '.join(task.context_objects) or 'N/A'}")
            print(f"     Контекст $:    {self.fmt.format_cost(task.context_cost)}")
        
        # Детали прогонов
        if task.runs:
            print()
            print(f"     Прогоны:")
            for run in task.runs:
                status = "✅" if run.success else "❌"
                hash_short = run.response_hash[:8] if run.response_hash else "N/A"
                print(f"       {status} Run {run.run_index}: seed={run.seed}, "
                      f"tokens={run.tokens_total}, time={self.fmt.format_time(run.elapsed_time)}, "
                      f"hash={hash_short}")
                if run.error:
                    print(f"          ⚠️  Error: {run.error[:60]}...")
        print()
    
    def print_full_report(self, filepath: Path):
        """Полный отчёт по эксперименту"""
        experiment = self.parser.parse_experiment(filepath)
        if not experiment:
            print(f"[Reporter] Failed to parse {filepath}")
            return
        
        self.print_experiment_summary(experiment)
        
        for task in experiment.tasks:
            self.print_task_details(task)
    
    def print_comparison_table(self, experiments: List[ExperimentStats]):
        """Таблица сравнения экспериментов"""
        if not experiments:
            print("[Reporter] No experiments to compare")
            return
        
        print()
        print("=" * 90)
        print("📊 СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
        print("=" * 90)
        print()
        
        # Заголовок
        print(f"{'Файл':<40} {'Кат.':<4} {'Токены':<10} {'Стоим.':<12} {'Детерм.':<10} {'Success':<10}")
        print("-" * 90)
        
        for exp in experiments:
            print(f"{exp.experiment_file:<40} "
                  f"{exp.category:<4} "
                  f"{self.fmt.format_tokens(exp.total_tokens):<10} "
                  f"{self.fmt.format_cost(exp.total_cost):<12} "
                  f"{self.fmt.format_percent(exp.avg_determinism):<10} "
                  f"{self.fmt.format_percent(exp.avg_success_rate):<10}")
        
        print("-" * 90)
        
        # Итого
        total_tokens = sum(e.total_tokens for e in experiments)
        total_cost = sum(e.total_cost for e in experiments)
        avg_det = sum(e.avg_determinism for e in experiments) / len(experiments)
        avg_success = sum(e.avg_success_rate for e in experiments) / len(experiments)
        
        print(f"{'ИТОГО':<40} "
              f"{'':4} "
              f"{self.fmt.format_tokens(total_tokens):<10} "
              f"{self.fmt.format_cost(total_cost):<12} "
              f"{self.fmt.format_percent(avg_det):<10} "
              f"{self.fmt.format_percent(avg_success):<10}")
        print()
    
    def print_model_comparison(self, experiments: List[ExperimentStats]):
        """Сравнение по моделям"""
        if not experiments:
            return
        
        # Собираем статистику по моделям
        model_stats: Dict[str, Dict] = defaultdict(lambda: {
            "tasks": 0,
            "tokens": 0,
            "cost": 0.0,
            "determinism_sum": 0.0,
            "success_sum": 0.0
        })
        
        for exp in experiments:
            for task in exp.tasks:
                stats = model_stats[task.model_name]
                stats["tasks"] += 1
                stats["tokens"] += task.total_tokens
                stats["cost"] += task.total_cost
                stats["determinism_sum"] += task.determinism_rate
                stats["success_sum"] += task.success_rate
        
        print()
        print("=" * 90)
        print("🤖 СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 90)
        print()
        
        print(f"{'Модель':<30} {'Задач':<8} {'Токены':<12} {'Стоим.':<12} {'Детерм.':<10} {'Success':<10}")
        print("-" * 90)
        
        for model_name, stats in sorted(model_stats.items()):
            tasks = stats["tasks"]
            avg_det = stats["determinism_sum"] / tasks if tasks > 0 else 0
            avg_success = stats["success_sum"] / tasks if tasks > 0 else 0
            
            print(f"{model_name:<30} "
                  f"{tasks:<8} "
                  f"{self.fmt.format_tokens(stats['tokens']):<12} "
                  f"{self.fmt.format_cost(stats['cost']):<12} "
                  f"{self.fmt.format_percent(avg_det):<10} "
                  f"{self.fmt.format_percent(avg_success):<10}")
        
        print()
    
    def generate_report(self, category: str = None, latest: int = None):
        """
        Сгенерировать отчёт
        
        Args:
            category: Фильтр по категории (A, B)
            latest: Показать только последние N экспериментов
        """
        files = self.parser.list_experiments(category)
        
        if latest:
            files = files[-latest:]
        
        if not files:
            print(f"[Reporter] No experiments found in {self.parser.results_dir}")
            return
        
        experiments = []
        for f in files:
            exp = self.parser.parse_experiment(f)
            if exp:
                experiments.append(exp)
        
        if len(experiments) == 1:
            # Один эксперимент — подробный отчёт
            self.print_full_report(files[0])
        else:
            # Несколько — сравнительная таблица
            self.print_comparison_table(experiments)
            self.print_model_comparison(experiments)
    
    def export_markdown(self, experiment: ExperimentStats, output_path: str = None) -> str:
        """
        Экспортировать отчёт в Markdown
        
        Args:
            experiment: Статистика эксперимента
            output_path: Путь для сохранения (опционально)
            
        Returns:
            Markdown текст
        """
        lines = []
        
        # Заголовок
        lines.append(f"# 📊 Отчёт: {experiment.experiment_file}")
        lines.append("")
        lines.append(f"**Категория:** {experiment.category}  ")
        lines.append(f"**Дата:** {experiment.timestamp[:19] if experiment.timestamp else 'N/A'}  ")
        lines.append(f"**Модели:** {', '.join(experiment.models_used)}  ")
        lines.append("")
        
        # Общие метрики
        lines.append("## 📈 Общие метрики")
        lines.append("")
        lines.append("| Метрика | Значение |")
        lines.append("|---------|----------|")
        lines.append(f"| Токены | {self.fmt.format_tokens(experiment.total_tokens)} |")
        lines.append(f"| Стоимость | {self.fmt.format_cost(experiment.total_cost)} |")
        lines.append(f"| Время | {self.fmt.format_time(experiment.total_time)} |")
        lines.append(f"| Детерминизм | {self.fmt.format_percent(experiment.avg_determinism)} {self.fmt.determinism_emoji(experiment.avg_determinism)} |")
        lines.append(f"| Success Rate | {self.fmt.format_percent(experiment.avg_success_rate)} {self.fmt.success_emoji(experiment.avg_success_rate)} |")
        lines.append("")
        
        # Задачи
        lines.append("## 📝 Результаты по задачам")
        lines.append("")
        
        for task in experiment.tasks:
            lines.append(f"### {task.task_id}: {task.task_name}")
            lines.append("")
            lines.append(f"**Модель:** {task.model_name}  ")
            lines.append(f"**Токены:** {self.fmt.format_tokens(task.total_tokens)}  ")
            lines.append(f"**Стоимость:** {self.fmt.format_cost(task.total_cost)}  ")
            lines.append(f"**Детерминизм:** {self.fmt.format_percent(task.determinism_rate)} ({task.unique_responses} уник.)  ")
            lines.append(f"**Success Rate:** {self.fmt.format_percent(task.success_rate)}  ")
            
            if task.context_loaded:
                lines.append(f"**Контекст:** {', '.join(task.context_objects) or 'N/A'}  ")
            
            lines.append("")
            
            # Таблица прогонов
            lines.append("| Run | Seed | Status | Tokens | Time | Hash |")
            lines.append("|-----|------|--------|--------|------|------|")
            
            for run in task.runs:
                status = "✅" if run.success else "❌"
                hash_short = run.response_hash[:8] if run.response_hash else "-"
                lines.append(f"| {run.run_index} | {run.seed} | {status} | {run.tokens_total} | {self.fmt.format_time(run.elapsed_time)} | `{hash_short}` |")
            
            lines.append("")
        
        markdown = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"[Reporter] Markdown saved to {output_path}")
        
        return markdown


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ результатов бенчмарка")
    parser.add_argument("-c", "--category", choices=["A", "B"], help="Фильтр по категории")
    parser.add_argument("-n", "--latest", type=int, help="Показать последние N экспериментов")
    parser.add_argument("-f", "--file", help="Конкретный файл результатов")
    parser.add_argument("--dir", default="results", help="Директория с результатами")
    
    args = parser.parse_args()
    
    reporter = ExperimentReporter(args.dir)
    
    if args.file:
        reporter.print_full_report(Path(args.file))
    else:
        reporter.generate_report(category=args.category, latest=args.latest)


if __name__ == "__main__":
    main()
