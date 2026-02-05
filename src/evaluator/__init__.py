"""
Evaluator — модуль экспертной оценки и аналитики SMOP

Предоставляет:
- Парсинг результатов экспериментов (parser)
- Логику оценки SMOP (smop)
- Статистический анализ (statistics)
- Генерацию отчётов (report)
- Интерактивный интерфейс (dashboard)

Публичный API:
    from src.evaluator import (
        # Парсер
        ExperimentParser,
        list_available_experiments,
        
        # SMOP оценка
        SMOPEvaluator,
        SMOPCriteria,
        get_smop_criteria,
        
        # Схемы
        SMOPScores,
        RunEvaluation,
        TaskEvaluation,
        ExperimentEvaluation,
        ReportSummary,
        
        # Статистика
        StatisticsCalculator,
        calculate_experiment_statistics,
        
        # Отчёты
        ReportGenerator,
        generate_report,
        
        # Интерфейс
        EvaluatorDashboard,
        run_dashboard,
        list_experiments_cli,
        show_status_cli,
    )

Пример использования:
    # Загрузка эксперимента
    parser = ExperimentParser("raw_results")
    experiments = parser.list_experiments()
    experiment = parser.load_experiment("experiment_B_123")
    
    # Создание оценки
    evaluator = SMOPEvaluator("evaluations")
    evaluation = parser.create_evaluation(experiment)
    
    # Проставление оценок
    evaluator.set_score(evaluation, "B1", "google/gemini", 0, "S", 10)
    evaluator.save(evaluation)
    
    # Генерация отчёта
    report = generate_report(evaluation, experiment)
    
    # Интерактивный режим
    run_dashboard("experiment_B_123")
"""

from .schemas import (
    VALID_SCORES,
    QUALITY_HIGH,
    QUALITY_ACCEPTABLE,
    SMOPScores,
    RunEvaluation,
    TaskEvaluation,
    ExperimentEvaluation,
    MetricStats,
    QualityStats,
    ModelSummary,
    TaskSummary,
    ReportSummary,
)

from .parser import (
    ExperimentParser,
    list_available_experiments,
)

from .smop import (
    SMOPCriteria,
    SMOPEvaluator,
    get_smop_criteria,
)

from .statistics import (
    StatisticsCalculator,
    calculate_experiment_statistics,
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_ci_95,
    calculate_metric_stats,
    calculate_quality_stats,
)

from .report import (
    ReportGenerator,
    generate_report,
)

from .dashboard import (
    EvaluatorDashboard,
    run_dashboard,
    list_experiments_cli,
    show_status_cli,
)

from .charts import (
    ChartGenerator,
    generate_charts,
    check_matplotlib_available,
)


__all__ = [
    # Константы
    "VALID_SCORES",
    "QUALITY_HIGH",
    "QUALITY_ACCEPTABLE",
    
    # Схемы
    "SMOPScores",
    "RunEvaluation",
    "TaskEvaluation",
    "ExperimentEvaluation",
    "MetricStats",
    "QualityStats",
    "ModelSummary",
    "TaskSummary",
    "ReportSummary",
    
    # Парсер
    "ExperimentParser",
    "list_available_experiments",
    
    # SMOP
    "SMOPCriteria",
    "SMOPEvaluator",
    "get_smop_criteria",
    
    # Статистика
    "StatisticsCalculator",
    "calculate_experiment_statistics",
    "calculate_mean",
    "calculate_median",
    "calculate_std",
    "calculate_ci_95",
    "calculate_metric_stats",
    "calculate_quality_stats",
    
    # Отчёты
    "ReportGenerator",
    "generate_report",
    
    # Интерфейс
    "EvaluatorDashboard",
    "run_dashboard",
    "list_experiments_cli",
    "show_status_cli",
    
    # Графики
    "ChartGenerator",
    "generate_charts",
    "check_matplotlib_available",
]
