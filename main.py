"""
AI-1C-Code-Generation-Benchmark CLI
Универсальный entry point для всех операций фреймворка.

Примеры:
    # Запуск эксперимента
    python main.py run -c A -m gemini -t A1
    python main.py run -c B --all-models
    
    # Информация
    python main.py info --balance
    python main.py info --models
    python main.py info --tasks A
"""

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import setup_logging
from src.core.benchmark import BenchmarkRunner
from src.utils.file_ops import load_yaml

# Настраиваем логирование при старте
setup_logging()


# =============================================================================
# CLI утилиты вывода
# =============================================================================

def print_section(title: str) -> None:
    """Вывести заголовок секции"""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_kv(label: str, value: str) -> None:
    """Вывести пару ключ-значение"""
    print(f"  {label}: {value}")


def get_runner():
    """Создать BenchmarkRunner (использует настройки из Settings)"""
    return BenchmarkRunner()


async def cmd_run(args):
    """Запустить эксперимент"""
    runner = get_runner()
    models = args.models
    if args.all_models:
        models = None
    if args.category == "B":
        await runner.init_mcp(use_mock=not args.no_mock)
    try:
        result = await runner.run_experiment(
            category=args.category,
            model_keys=models,
            task_ids=args.tasks
        )
        
        print_section("Итоги")
        print_kv("Задач выполнено", str(len(result.task_results)))
        print_kv("Всего токенов", f"{result.total_tokens:,}")
        print_kv("Общая стоимость", f"${result.total_cost:.4f}")
        print_kv("Общее время", f"{result.total_time:.1f} сек")
        
        # Средний процент детерминизма
        if result.task_results:
            avg_match = sum(t.determinism.match_percent for t in result.task_results if t.determinism) / len(result.task_results)
            print_kv("Детерминизм", f"{avg_match:.1f}% (среднее совпадение ответов)")
        print()
        
    finally:
        await runner.close_mcp()


def cmd_info(args):
    """Показать информацию"""
    from src.config.settings import get_settings
    settings = get_settings()
    runner = get_runner()
    
    if args.balance:
        balance = runner.llm.get_balance()
        if balance:
            print_section("Баланс OpenRouter")
            print_kv("Лимит", f"${balance['limit']:.2f}")
            print_kv("Использовано", f"${balance['usage']:.4f}")
            print_kv("Доступно", f"${balance['available']:.4f}")
        else:
            print("Ошибка: не удалось получить баланс")
    
    if args.models:
        print_section("Доступные модели")
        models_path = settings.paths.get_models_path()
        models_config = load_yaml(models_path)
        print(f"  {'Ключ':<10} {'Название':<25} {'Детерминизм':<12} {'Цена (in/out)':<15}")
        print("  " + "-" * 65)
        for key, model in models_config["models"].items():
            meta = model.get("meta", {})
            det = meta.get("determinism_param", "temperature")
            price_in = meta.get("price_input", 0)
            price_out = meta.get("price_output", 0)
            print(f"  {key:<10} {model['name']:<25} {det:<12} ${price_in}/{price_out}")
    
    if args.tasks:
        category = args.tasks
        print_section(f"Задачи категории {category}")
        tasks_path = settings.paths.get_tasks_path(category)
        tasks_config = load_yaml(tasks_path)
        print(f"  {'ID':<6} {'Название':<35} {'Сложность':<10}")
        print("  " + "-" * 55)
        for task in tasks_config.get("tasks", []):
            print(f"  {task['id']:<6} {task['name']:<35} {task.get('difficulty', 'medium'):<10}")
    print()


def cmd_evaluate(args):
    """Оценка эксперимента SMOP"""
    from src.config.settings import get_settings
    from src.evaluator import (
        run_dashboard,
        list_experiments_cli,
        show_status_cli,
    )
    
    settings = get_settings()
    
    if args.list:
        list_experiments_cli(settings.paths.raw_results_dir)
        return
    
    if args.status:
        show_status_cli(args.status, settings.paths.evaluations_dir)
        return
    
    if not args.experiment_id:
        print("Ошибка: укажите ID эксперимента или используйте --list")
        return
    
    run_dashboard(
        experiment_id=args.experiment_id,
        evaluator_id=args.evaluator,
        results_dir=settings.paths.raw_results_dir,
        evaluations_dir=settings.paths.evaluations_dir,
        reports_dir=settings.paths.reports_dir,
    )


def cmd_report(args):
    """Генерация отчёта"""
    from src.config.settings import get_settings
    from src.evaluator import (
        ExperimentParser,
        SMOPEvaluator,
        ReportGenerator,
        generate_report,
    )
    
    settings = get_settings()
    
    # Загружаем эксперимент и оценку
    parser = ExperimentParser(settings.paths.raw_results_dir)
    evaluator = SMOPEvaluator(settings.paths.evaluations_dir)
    
    experiment = parser.load_experiment(args.experiment_id)
    if not experiment:
        print(f"Ошибка: эксперимент не найден: {args.experiment_id}")
        return
    
    # Ищем любую оценку для этого эксперимента
    evaluations = evaluator.list_evaluations(args.experiment_id)
    if not evaluations:
        print(f"Ошибка: оценка для эксперимента не найдена. Сначала выполните: evaluate {args.experiment_id}")
        return
    
    # Берём первую (или самую полную) оценку
    best_eval_info = max(evaluations, key=lambda e: e["progress_percent"])
    evaluation = evaluator.load(args.experiment_id, best_eval_info["evaluator_id"])
    
    if not evaluation:
        print("Ошибка: не удалось загрузить оценку")
        return
    
    # Определяем форматы
    if args.format == "all":
        formats = ["json", "html", "latex"]
    else:
        formats = [args.format]
    
    # Генерируем отчёт
    paths = generate_report(
        evaluation,
        experiment,
        settings.paths.reports_dir,
        formats=formats
    )
    
    print_section("Отчёты сгенерированы")
    for fmt, path in paths.items():
        print(f"  {fmt.upper()}: {path}")
    print()
    
    # Сравнительный отчёт если указан --compare
    if args.compare:
        experiment2 = parser.load_experiment(args.compare)
        evaluations2 = evaluator.list_evaluations(args.compare)
        
        if experiment2 and evaluations2:
            best_eval2 = max(evaluations2, key=lambda e: e["progress_percent"])
            evaluation2 = evaluator.load(args.compare, best_eval2["evaluator_id"])
            
            if evaluation2:
                generator = ReportGenerator(settings.paths.reports_dir)
                report1 = generator.generate(evaluation, experiment)
                report2 = generator.generate(evaluation2, experiment2)
                
                comparison = generator.generate_comparison_report(report1, report2)
                
                print_section("Сравнительный анализ")
                delta = comparison.get("delta", {})
                for metric, data in delta.items():
                    diff = data.get("diff", 0)
                    sign = "+" if diff > 0 else ""
                    print(f"  {metric}: {data.get('exp1', 0):.1f} → {data.get('exp2', 0):.1f} ({sign}{diff:.2f})")
                print()


def cmd_stats(args):
    """Показать статистику эксперимента"""
    from src.config.settings import get_settings
    from src.evaluator import (
        ExperimentParser,
        SMOPEvaluator,
        StatisticsCalculator,
    )
    
    settings = get_settings()
    
    parser = ExperimentParser(settings.paths.raw_results_dir)
    evaluator = SMOPEvaluator(settings.paths.evaluations_dir)
    
    experiment = parser.load_experiment(args.experiment_id)
    evaluations = evaluator.list_evaluations(args.experiment_id)
    
    if not evaluations:
        print(f"Оценка для {args.experiment_id} не найдена")
        return
    
    best_eval_info = max(evaluations, key=lambda e: e["progress_percent"])
    evaluation = evaluator.load(args.experiment_id, best_eval_info["evaluator_id"])
    
    if not evaluation:
        print("Не удалось загрузить оценку")
        return
    
    calc = StatisticsCalculator(evaluation, experiment)
    summary = calc.calculate_summary()
    
    print_section(f"Статистика: {args.experiment_id}")
    
    # Общие метрики
    overall_q = summary.get("overall_Q", {})
    print(f"  Интегральный Q: {overall_q.get('mean', 0):.2f} ± {overall_q.get('std', 0):.2f}")
    print(f"  95% ДИ: [{overall_q.get('ci_lower', 0):.2f}, {overall_q.get('ci_upper', 0):.2f}]")
    print()
    
    # По метрикам
    print("  Метрики SMOP:")
    for metric in ["S", "M", "O", "P"]:
        data = summary.get("by_metric", {}).get(metric, {})
        print(f"    {metric}: {data.get('mean', 0):.1f} (σ={data.get('std', 0):.2f})")
    print()
    
    # Прогресс
    print(f"  Оценено: {summary.get('total_evaluated', 0)}/{summary.get('total_runs', 0)} прогонов")
    
    # Детерминизм если есть
    det = summary.get("determinism")
    if det:
        print(f"  Детерминизм: {det.get('mean', 0)*100:.1f}%")
    
    # Корреляция
    corr = calc.calculate_correlation_det_quality()
    if corr is not None:
        print(f"  Корреляция детерминизм-качество: {corr:.3f}")
    print()


def cmd_charts(args):
    """Генерация графиков эксперимента"""
    from src.config.settings import get_settings
    from src.evaluator import (
        ExperimentParser,
        SMOPEvaluator,
        ChartGenerator,
        check_matplotlib_available,
    )
    
    settings = get_settings()
    
    # Проверка matplotlib
    if not check_matplotlib_available():
        print("Ошибка: matplotlib не установлен")
        print("Установите: pip install matplotlib")
        return
    
    parser = ExperimentParser(settings.paths.raw_results_dir)
    evaluator = SMOPEvaluator(settings.paths.evaluations_dir)
    
    experiment = parser.load_experiment(args.experiment_id)
    evaluations = evaluator.list_evaluations(args.experiment_id)
    
    if not evaluations:
        print(f"Оценка для {args.experiment_id} не найдена")
        return
    
    best_eval_info = max(evaluations, key=lambda e: e["progress_percent"])
    evaluation = evaluator.load(args.experiment_id, best_eval_info["evaluator_id"])
    
    if not evaluation:
        print("Не удалось загрузить оценку")
        return
    
    # Форматы экспорта
    formats = args.format if isinstance(args.format, list) else [args.format]
    if "all" in formats:
        formats = ["png", "svg", "pdf"]
    
    # Директория для графиков
    charts_dir = Path(settings.paths.reports_dir) / "charts" / args.experiment_id
    
    print_section(f"Генерация графиков: {args.experiment_id}")
    print(f"  Форматы: {', '.join(formats)}")
    print(f"  Директория: {charts_dir}")
    print()
    
    generator = ChartGenerator(
        evaluation,
        experiment,
        str(charts_dir),
        formats=formats
    )
    
    # Выбор графиков
    if args.chart == "all":
        results = generator.generate_all()
    else:
        chart_methods = {
            "dashboard": generator.plot_summary_dashboard,
            "radar": generator.plot_smop_radar,
            "comparison": generator.plot_models_comparison,
            "q_by_model": generator.plot_q_by_model,
            "distribution": generator.plot_scores_distribution,
            "boxplot": generator.plot_boxplot_by_model,
            "heatmap": generator.plot_heatmap_tasks_models,
            "det_quality": generator.plot_determinism_vs_quality,
        }
        
        if args.chart in chart_methods:
            paths = chart_methods[args.chart]()
            results = {args.chart: paths} if paths else {}
        else:
            print(f"Неизвестный тип графика: {args.chart}")
            return
    
    print_section("Графики созданы")
    for name, paths in results.items():
        print(f"  {name}:")
        for p in paths:
            print(f"    - {p}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI-1C-Code-Generation-Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s run -c A -m gemini -t A1      # Запуск одной задачи
  %(prog)s run -c A --all-models         # Запуск всех моделей
  %(prog)s info --balance                # Проверка баланса OpenRouter
  %(prog)s info --models                 # Список доступных моделей
  %(prog)s info --tasks A                # Список задач категории
  %(prog)s evaluate experiment_B_123     # Оценка эксперимента
  %(prog)s evaluate --list               # Список экспериментов
  %(prog)s report experiment_B_123       # Генерация отчёта
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Команды")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Запустить эксперимент")
    run_parser.add_argument("-c", "--category", choices=["A", "B"], default="A",
                           help="Категория задач (по умолчанию: A)")
    run_parser.add_argument("-m", "--models", nargs="+", choices=["claude", "gpt", "gemini"],
                           help="Модели для тестирования")
    run_parser.add_argument("--all-models", action="store_true",
                           help="Тестировать все доступные модели")
    run_parser.add_argument("-t", "--tasks", nargs="+",
                           help="ID задач (например: A1 A2)")
    run_parser.add_argument("--no-mock", action="store_true",
                           help="Использовать реальный MCP сервер (категория B)")
    
    # info command
    info_parser = subparsers.add_parser("info", help="Показать информацию")
    info_parser.add_argument("--balance", action="store_true",
                            help="Показать баланс OpenRouter")
    info_parser.add_argument("--models", action="store_true",
                            help="Список доступных моделей")
    info_parser.add_argument("--tasks", metavar="CATEGORY",
                            help="Список задач категории (A или B)")
    
    # evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Оценка эксперимента SMOP")
    evaluate_parser.add_argument("experiment_id", nargs="?",
                                help="ID эксперимента для оценки")
    evaluate_parser.add_argument("--list", action="store_true",
                                help="Показать список доступных экспериментов")
    evaluate_parser.add_argument("--status", metavar="EXPERIMENT_ID",
                                help="Показать прогресс оценки эксперимента")
    evaluate_parser.add_argument("--evaluator", default="expert_01",
                                help="ID эксперта (по умолчанию: expert_01)")
    
    # report command
    report_parser = subparsers.add_parser("report", help="Генерация отчёта")
    report_parser.add_argument("experiment_id",
                              help="ID эксперимента")
    report_parser.add_argument("--format", choices=["json", "html", "latex", "all"],
                              default="all",
                              help="Формат отчёта (по умолчанию: all)")
    report_parser.add_argument("--compare", metavar="EXPERIMENT_ID",
                              help="ID второго эксперимента для сравнения")
    
    # stats command
    stats_parser = subparsers.add_parser("stats", help="Показать статистику")
    stats_parser.add_argument("experiment_id",
                             help="ID эксперимента")
    
    # charts command
    charts_parser = subparsers.add_parser("charts", help="Генерация графиков")
    charts_parser.add_argument("experiment_id",
                              help="ID эксперимента")
    charts_parser.add_argument("--chart", 
                              choices=["all", "dashboard", "radar", "comparison", 
                                      "q_by_model", "distribution", "boxplot", 
                                      "heatmap", "det_quality"],
                              default="all",
                              help="Тип графика (по умолчанию: all)")
    charts_parser.add_argument("--format", nargs="+",
                              choices=["png", "svg", "pdf", "all"],
                              default=["all"],
                              help="Формат экспорта (по умолчанию: all)")
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "charts":
        cmd_charts(args)


if __name__ == "__main__":
    main()
