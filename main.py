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

from src.core.benchmark import BenchmarkRunner
from src.core.reporter import ExperimentReporter
from src.utils.file_ops import load_yaml
from src.utils.cli import print_section, print_kv, print_table_header, print_table_row


def get_runner():
    """Создать BenchmarkRunner"""
    return BenchmarkRunner(config_dir="config", results_dir="results")


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
        models_config = load_yaml(Path("config/models.yaml"))
        print_table_header(("Ключ", 10), ("Название", 20), ("Детерминизм", 12), ("Цена (in/out за 1M)", 20))
        for key, model in models_config["models"].items():
            meta = model.get("meta", {})
            det = meta.get("determinism_param", "temperature")
            price_in = meta.get("price_input", 0)
            price_out = meta.get("price_output", 0)
            print_table_row((key, 10), (model['name'], 20), (det, 12), (f"${price_in} / ${price_out}", 20))
    
    if args.tasks:
        category = args.tasks
        print_section(f"Задачи категории {category}")
        tasks_config = load_yaml(Path(f"config/tasks_category_{category}.yaml"))
        print_table_header(("ID", 6), ("Название", 30), ("Сложность", 10))
        for task in tasks_config.get("tasks", []):
            print_table_row((task['id'], 6), (task['name'], 30), (task.get('difficulty', 'medium'), 10))
    print()


def cmd_report(args):
    """Показать отчёт по результатам"""
    reporter = ExperimentReporter(results_dir="results")
    
    if args.file:
        if args.markdown:
            exp = reporter.parser.parse_experiment(Path(args.file))
            if exp:
                md_path = args.file.replace('.json', '.md')
                reporter.export_markdown(exp, md_path)
        else:
            reporter.print_full_report(Path(args.file))
    else:
        reporter.generate_report(category=args.category, latest=args.latest)


def main():
    parser = argparse.ArgumentParser(
        description="AI-1C-Code-Generation-Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s run -c A -m gemini -t A1      # Запуск одной задачи
  %(prog)s run -c A --all-models         # Запуск всех моделей
  %(prog)s report -c B                   # Отчёт по категории B
  %(prog)s report -n 5                   # Последние 5 экспериментов
  %(prog)s report -f results/exp.json    # Конкретный файл
  %(prog)s info --balance                # Проверка баланса OpenRouter
  %(prog)s info --models                 # Список доступных моделей
  %(prog)s info --tasks A                # Список задач категории
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
    
    # report command
    report_parser = subparsers.add_parser("report", help="Отчёт по результатам")
    report_parser.add_argument("-c", "--category", choices=["A", "B"],
                              help="Фильтр по категории")
    report_parser.add_argument("-n", "--latest", type=int,
                              help="Показать последние N экспериментов")
    report_parser.add_argument("-f", "--file",
                              help="Конкретный файл результатов")
    report_parser.add_argument("--markdown", action="store_true",
                              help="Экспортировать в Markdown (с -f)")
    
    # info command
    info_parser = subparsers.add_parser("info", help="Показать информацию")
    info_parser.add_argument("--balance", action="store_true",
                            help="Показать баланс OpenRouter")
    info_parser.add_argument("--models", action="store_true",
                            help="Список доступных моделей")
    info_parser.add_argument("--tasks", metavar="CATEGORY",
                            help="Список задач категории (A или B)")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
