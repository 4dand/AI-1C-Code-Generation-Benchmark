"""
Report — генерация отчётов по результатам оценки

Отвечает за:
- FR-401: Генерация итогового JSON-отчёта
- FR-402: Генерация HTML-отчёта
- FR-403: Таблица результатов по задачам
- FR-404: Таблица сравнения моделей
- FR-405: Графики распределения оценок
- FR-406: Экспорт таблиц в LaTeX
- FR-407: Метаданные эксперимента
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..utils.file_ops import save_json, ensure_dir
from ..schemas.results import ExperimentResult
from .schemas import ExperimentEvaluation, ReportSummary, ModelSummary, TaskSummary
from .statistics import StatisticsCalculator, calculate_experiment_statistics


logger = logging.getLogger(__name__)


# HTML-шаблон для отчёта
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчёт SMOP: {{ experiment_id }}</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --accent-green: #4caf50;
            --accent-yellow: #ffc107;
            --border: #2a2a4e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: var(--accent); margin-bottom: 10px; }
        h2 { color: var(--text-primary); margin: 30px 0 15px; border-bottom: 2px solid var(--accent); padding-bottom: 5px; }
        h3 { color: var(--text-secondary); margin: 20px 0 10px; }
        .meta { color: var(--text-secondary); font-size: 0.9em; margin-bottom: 30px; }
        .meta span { margin-right: 20px; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .card {
            background: var(--bg-card);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--border);
        }
        .card-label { color: var(--text-secondary); font-size: 0.85em; text-transform: uppercase; }
        .card-value { font-size: 2em; font-weight: bold; margin: 5px 0; }
        .card-detail { font-size: 0.9em; color: var(--text-secondary); }
        .high { color: var(--accent-green); }
        .acceptable { color: var(--accent-yellow); }
        .low { color: var(--accent); }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: var(--bg-card); color: var(--text-primary); font-weight: 600; }
        tr:hover { background: rgba(233, 69, 96, 0.1); }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .badge-high { background: var(--accent-green); color: #000; }
        .badge-acceptable { background: var(--accent-yellow); color: #000; }
        .badge-low { background: var(--accent); color: #fff; }
        .bar-container { width: 100px; height: 8px; background: var(--border); border-radius: 4px; display: inline-block; vertical-align: middle; }
        .bar { height: 100%; border-radius: 4px; }
        .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid var(--border); color: var(--text-secondary); font-size: 0.85em; text-align: center; }
        @media (max-width: 768px) {
            .cards { grid-template-columns: 1fr 1fr; }
            table { font-size: 0.9em; }
            th, td { padding: 8px 10px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Отчёт SMOP</h1>
        <div class="meta">
            <span>🧪 <strong>{{ experiment_id }}</strong></span>
            <span>📅 {{ generated_at }}</span>
            <span>📁 Категория {{ category }}</span>
        </div>

        <h2>📈 Общие метрики</h2>
        <div class="cards">
            <div class="card">
                <div class="card-label">Интегральный Q</div>
                <div class="card-value {{ quality_class }}">{{ q_mean }}</div>
                <div class="card-detail">± {{ q_std }} (95% ДИ: {{ q_ci_lower }}–{{ q_ci_upper }})</div>
            </div>
            <div class="card">
                <div class="card-label">Синтаксис (S)</div>
                <div class="card-value">{{ s_mean }}</div>
                <div class="card-detail">σ = {{ s_std }}</div>
            </div>
            <div class="card">
                <div class="card-label">Семантика (M)</div>
                <div class="card-value">{{ m_mean }}</div>
                <div class="card-detail">σ = {{ m_std }}</div>
            </div>
            <div class="card">
                <div class="card-label">Оптимальность (O)</div>
                <div class="card-value">{{ o_mean }}</div>
                <div class="card-detail">σ = {{ o_std }}</div>
            </div>
            <div class="card">
                <div class="card-label">Платформа (P)</div>
                <div class="card-value">{{ p_mean }}</div>
                <div class="card-detail">σ = {{ p_std }}</div>
            </div>
            <div class="card">
                <div class="card-label">Оценено прогонов</div>
                <div class="card-value">{{ evaluated_runs }}</div>
                <div class="card-detail">из {{ total_runs }}</div>
            </div>
        </div>

        <h2>🤖 Сравнение моделей</h2>
        <table>
            <thead>
                <tr>
                    <th>Модель</th>
                    <th>S</th>
                    <th>M</th>
                    <th>O</th>
                    <th>P</th>
                    <th>Q</th>
                    <th>Уровень</th>
                    <th>Детерминизм</th>
                </tr>
            </thead>
            <tbody>
                {{ model_rows }}
            </tbody>
        </table>

        <h2>📝 Результаты по задачам</h2>
        <table>
            <thead>
                <tr>
                    <th>Задача</th>
                    <th>Прогонов</th>
                    <th>Q (среднее)</th>
                    <th>Q (медиана)</th>
                    <th>Разброс</th>
                </tr>
            </thead>
            <tbody>
                {{ task_rows }}
            </tbody>
        </table>

        <h2>ℹ️ Метаданные</h2>
        <div class="card" style="max-width: 500px;">
            <p><strong>Эксперимент:</strong> {{ experiment_id }}</p>
            <p><strong>Эксперт:</strong> {{ evaluator_id }}</p>
            <p><strong>Моделей:</strong> {{ models_count }}</p>
            <p><strong>Задач:</strong> {{ tasks_count }}</p>
            <p><strong>Прогонов на задачу:</strong> {{ runs_per_task }}</p>
            <p><strong>Версия фреймворка:</strong> {{ framework_version }}</p>
        </div>

        <div class="footer">
            Сгенерировано AI-1C-Code-Generation-Benchmark SMOP Evaluator<br>
            {{ generated_at }}
        </div>
    </div>
</body>
</html>
'''


class ReportGenerator:
    """
    Генератор отчётов по результатам оценки SMOP
    
    Создаёт JSON и HTML отчёты с агрегированными метриками,
    таблицами сравнения моделей и визуализацией.
    
    Example:
        generator = ReportGenerator("reports")
        
        report = generator.generate(evaluation, experiment)
        
        generator.save_json(report)
        generator.save_html(report)
    """
    
    def __init__(self, reports_dir: str = "reports"):
        """
        Инициализация генератора
        
        Args:
            reports_dir: Директория для сохранения отчётов
        """
        self.reports_dir = Path(reports_dir)
        ensure_dir(self.reports_dir)
    
    def generate(
        self,
        evaluation: ExperimentEvaluation,
        experiment: Optional[ExperimentResult] = None
    ) -> ReportSummary:
        """
        Сгенерировать полный отчёт
        
        Args:
            evaluation: Оценка эксперимента
            experiment: Сырые результаты (опционально)
            
        Returns:
            ReportSummary с полными данными
            
        FR-401, FR-407: Генерация отчёта с метаданными
        """
        calc = StatisticsCalculator(evaluation, experiment)
        
        # Метаданные
        metadata = {
            "category": experiment.category if experiment else "?",
            "models_tested": list(set(t.model_name for t in evaluation.tasks)),
            "tasks_count": len(set(t.task_id for t in evaluation.tasks)),
            "runs_per_task": evaluation.total_runs // max(1, len(evaluation.tasks)),
            "total_runs_evaluated": evaluation.evaluated_runs,
            "total_runs": evaluation.total_runs,
            "evaluator_id": evaluation.evaluator_id,
            "framework_version": evaluation.framework_version,
            "started_at": evaluation.started_at,
            "completed_at": evaluation.completed_at,
        }
        
        # Сводка
        summary = calc.calculate_summary()
        
        # По моделям и задачам
        by_model = calc.aggregate_by_model()
        by_task = calc.aggregate_by_task()
        
        # Корреляция
        correlation = calc.calculate_correlation_det_quality()
        if correlation is not None:
            summary["correlation_determinism_quality"] = correlation
        
        report = ReportSummary(
            experiment_id=evaluation.experiment_id,
            metadata=metadata,
            summary=summary,
            by_model=by_model,
            by_task=by_task,
        )
        
        logger.info(f"Отчёт сгенерирован: {evaluation.experiment_id}")
        
        return report
    
    def save_json(self, report: ReportSummary) -> Path:
        """
        Сохранить отчёт в JSON
        
        FR-401: Генерация JSON-отчёта
        """
        path = self.reports_dir / f"{report.experiment_id}_report.json"
        save_json(report.model_dump(), path)
        
        logger.info(f"JSON отчёт сохранён: {path}")
        return path
    
    def save_html(self, report: ReportSummary) -> Path:
        """
        Сохранить отчёт в HTML
        
        FR-402: Генерация HTML-отчёта
        FR-403, FR-404: Таблицы результатов
        """
        # Подготовка данных для шаблона
        summary = report.summary
        overall_q = summary.get("overall_Q", {})
        by_metric = summary.get("by_metric", {})
        
        q_mean = overall_q.get("mean", 0)
        quality_class = "high" if q_mean >= 8 else ("acceptable" if q_mean >= 5 else "low")
        
        # Строки таблицы моделей
        model_rows = []
        for m in report.by_model:
            q_val = m.Q.mean
            level_class = "high" if q_val >= 8 else ("acceptable" if q_val >= 5 else "low")
            level_text = "Высокий" if q_val >= 8 else ("Приемлемый" if q_val >= 5 else "Низкий")
            
            det_bar_width = int(m.determinism_mean)
            
            row = f'''<tr>
                <td><strong>{m.model_name}</strong></td>
                <td>{m.S.mean:.1f}</td>
                <td>{m.M.mean:.1f}</td>
                <td>{m.O.mean:.1f}</td>
                <td>{m.P.mean:.1f}</td>
                <td><strong>{m.Q.mean:.1f}</strong></td>
                <td><span class="badge badge-{level_class}">{level_text}</span></td>
                <td>
                    <div class="bar-container">
                        <div class="bar" style="width: {det_bar_width}%; background: var(--accent-green);"></div>
                    </div>
                    {m.determinism_mean:.0f}%
                </td>
            </tr>'''
            model_rows.append(row)
        
        # Строки таблицы задач
        task_rows = []
        for t in report.by_task:
            spread = t.Q.max - t.Q.min if t.Q.count > 0 else 0
            
            row = f'''<tr>
                <td><strong>{t.task_id}</strong>: {t.task_name}</td>
                <td>{t.runs_count}</td>
                <td>{t.Q.mean:.1f}</td>
                <td>{t.Q.median:.1f}</td>
                <td>{spread:.1f}</td>
            </tr>'''
            task_rows.append(row)
        
        # Заполняем шаблон
        html = HTML_TEMPLATE
        replacements = {
            "{{ experiment_id }}": report.experiment_id,
            "{{ generated_at }}": datetime.fromisoformat(report.generated_at).strftime("%d.%m.%Y %H:%M"),
            "{{ category }}": report.metadata.get("category", "?"),
            "{{ quality_class }}": quality_class,
            "{{ q_mean }}": f"{q_mean:.1f}",
            "{{ q_std }}": f"{overall_q.get('std', 0):.2f}",
            "{{ q_ci_lower }}": f"{overall_q.get('ci_lower', 0):.1f}",
            "{{ q_ci_upper }}": f"{overall_q.get('ci_upper', 0):.1f}",
            "{{ s_mean }}": f"{by_metric.get('S', {}).get('mean', 0):.1f}",
            "{{ s_std }}": f"{by_metric.get('S', {}).get('std', 0):.2f}",
            "{{ m_mean }}": f"{by_metric.get('M', {}).get('mean', 0):.1f}",
            "{{ m_std }}": f"{by_metric.get('M', {}).get('std', 0):.2f}",
            "{{ o_mean }}": f"{by_metric.get('O', {}).get('mean', 0):.1f}",
            "{{ o_std }}": f"{by_metric.get('O', {}).get('std', 0):.2f}",
            "{{ p_mean }}": f"{by_metric.get('P', {}).get('mean', 0):.1f}",
            "{{ p_std }}": f"{by_metric.get('P', {}).get('std', 0):.2f}",
            "{{ evaluated_runs }}": str(summary.get("total_evaluated", 0)),
            "{{ total_runs }}": str(summary.get("total_runs", 0)),
            "{{ model_rows }}": "\n".join(model_rows),
            "{{ task_rows }}": "\n".join(task_rows),
            "{{ evaluator_id }}": report.metadata.get("evaluator_id", "expert_01"),
            "{{ models_count }}": str(len(report.by_model)),
            "{{ tasks_count }}": str(report.metadata.get("tasks_count", 0)),
            "{{ runs_per_task }}": str(report.metadata.get("runs_per_task", 0)),
            "{{ framework_version }}": report.metadata.get("framework_version", "1.0.0"),
        }
        
        for key, value in replacements.items():
            html = html.replace(key, value)
        
        # Сохраняем
        path = self.reports_dir / f"{report.experiment_id}_report.html"
        path.write_text(html, encoding="utf-8")
        
        logger.info(f"HTML отчёт сохранён: {path}")
        return path
    
    def save_latex_tables(self, report: ReportSummary) -> Path:
        """
        Экспорт таблиц в формате LaTeX
        
        FR-406: Экспорт LaTeX для статьи
        """
        lines = []
        
        # Заголовок
        lines.append("% SMOP Evaluation Report")
        lines.append(f"% Experiment: {report.experiment_id}")
        lines.append(f"% Generated: {report.generated_at}")
        lines.append("")
        
        # Таблица сравнения моделей
        lines.append("% Model Comparison Table")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Сравнение качества генерации кода по моделям}")
        lines.append("\\label{tab:model-comparison}")
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\toprule")
        lines.append("Модель & S & M & O & P & Q & Det. \\\\")
        lines.append("\\midrule")
        
        for m in report.by_model:
            name = m.model_name.replace("_", "\\_")
            lines.append(
                f"{name} & {m.S.mean:.1f} & {m.M.mean:.1f} & {m.O.mean:.1f} & "
                f"{m.P.mean:.1f} & \\textbf{{{m.Q.mean:.1f}}} & {m.determinism_mean:.0f}\\% \\\\"
            )
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
        
        # Таблица результатов по задачам
        lines.append("% Task Results Table")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Результаты оценки по задачам}")
        lines.append("\\label{tab:task-results}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Задача & Прогонов & Q (mean) & Q (median) & $\\sigma$ \\\\")
        lines.append("\\midrule")
        
        for t in report.by_task:
            task_name = t.task_name[:30].replace("_", "\\_")
            lines.append(
                f"{t.task_id} & {t.runs_count} & {t.Q.mean:.1f} & "
                f"{t.Q.median:.1f} & {t.Q.std:.2f} \\\\"
            )
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        # Сохраняем
        path = self.reports_dir / f"{report.experiment_id}_tables.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        
        logger.info(f"LaTeX таблицы сохранены: {path}")
        return path
    
    def generate_comparison_report(
        self,
        report1: ReportSummary,
        report2: ReportSummary
    ) -> Dict[str, Any]:
        """
        Сравнительный отчёт двух экспериментов
        
        Например: baseline vs MCP
        
        Returns:
            Словарь с разницей метрик
        """
        comparison = {
            "experiment_1": report1.experiment_id,
            "experiment_2": report2.experiment_id,
            "generated_at": datetime.now().isoformat(),
            "delta": {},
        }
        
        # Разница по Q
        q1 = report1.summary.get("overall_Q", {}).get("mean", 0)
        q2 = report2.summary.get("overall_Q", {}).get("mean", 0)
        
        comparison["delta"]["Q"] = {
            "exp1": q1,
            "exp2": q2,
            "diff": round(q2 - q1, 3),
            "improvement_percent": round((q2 - q1) / max(q1, 0.001) * 100, 1),
        }
        
        # Разница по метрикам
        for metric in ["S", "M", "O", "P"]:
            m1 = report1.summary.get("by_metric", {}).get(metric, {}).get("mean", 0)
            m2 = report2.summary.get("by_metric", {}).get(metric, {}).get("mean", 0)
            
            comparison["delta"][metric] = {
                "exp1": m1,
                "exp2": m2,
                "diff": round(m2 - m1, 3),
            }
        
        return comparison


def generate_report(
    evaluation: ExperimentEvaluation,
    experiment: Optional[ExperimentResult] = None,
    reports_dir: str = "reports",
    formats: List[str] = None
) -> Dict[str, Path]:
    """
    Удобная функция для генерации отчётов
    
    Args:
        evaluation: Оценка
        experiment: Результаты эксперимента
        reports_dir: Директория для отчётов
        formats: Форматы ["json", "html", "latex"]
        
    Returns:
        Словарь {формат: путь}
    """
    if formats is None:
        formats = ["json", "html"]
    
    generator = ReportGenerator(reports_dir)
    report = generator.generate(evaluation, experiment)
    
    paths = {}
    
    if "json" in formats:
        paths["json"] = generator.save_json(report)
    
    if "html" in formats:
        paths["html"] = generator.save_html(report)
    
    if "latex" in formats:
        paths["latex"] = generator.save_latex_tables(report)
    
    return paths
