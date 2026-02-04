"""
Сохраняет код 1С в .bsl файлы для удобного просмотра и копирования
в 1С Предприятие 8.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .file_ops import ensure_dir, load_json


def sanitize_filename(name: str) -> str:
    """
    Преобразовать имя в безопасное имя файла
    
    Args:
        name: Исходное имя
        
    Returns:
        Безопасное имя файла
    """
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
    safe_name = safe_name.replace(' ', '_')
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name[:100]


def export_code_to_bsl(
    response: str,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Сохранить код 1С в .bsl файл
    
    Args:
        response: Ответ модели (код 1С)
        output_path: Путь для сохранения файла
        metadata: Метаданные для добавления в заголовок комментария
    """
    ensure_dir(output_path.parent)
    header_lines = ["// ═══════════════════════════════════════════════════════════════"]
    header_lines.append("// Автоматически сгенерированный код")
    header_lines.append("// AI-1C-Code-Generation-Benchmark")
    header_lines.append("// ═══════════════════════════════════════════════════════════════")
    
    if metadata:
        header_lines.append("//")
        if "task_id" in metadata:
            header_lines.append(f"// Задача: {metadata.get('task_id', '')} - {metadata.get('task_name', '')}")
        if "model_name" in metadata:
            header_lines.append(f"// Модель: {metadata.get('model_name', '')} ({metadata.get('model_id', '')})")
        if "run_index" in metadata:
            header_lines.append(f"// Прогон: {metadata.get('run_index', 0) + 1}")
        if "seed" in metadata and metadata["seed"] is not None:
            header_lines.append(f"// Seed: {metadata['seed']}")
        if "temperature" in metadata:
            header_lines.append(f"// Temperature: {metadata['temperature']}")
        if "response_hash" in metadata:
            header_lines.append(f"// Hash: {metadata['response_hash']}")
        if "timestamp" in metadata:
            header_lines.append(f"// Время: {metadata['timestamp']}")
        header_lines.append("//")
        header_lines.append("// ═══════════════════════════════════════════════════════════════")
    
    header = "\n".join(header_lines)
    content = f"{header}\n\n{response}\n"
    
    with open(output_path, 'w', encoding='utf-8-sig') as f:  # utf-8-sig для BOM
        f.write(content)


def export_experiment_code(
    experiment_data: Dict[str, Any],
    output_dir: Union[str, Path],
    include_all_runs: bool = False
) -> Dict[str, Any]:
    """
    Экспортировать код из результатов эксперимента в .bsl файлы
    
    Args:
        experiment_data: Данные эксперимента (словарь из JSON)
        output_dir: Базовая папка для экспорта
        include_all_runs: Если True - сохранить все прогоны, иначе только первый успешный
        
    Returns:
        Словарь с информацией об экспортированных файлах
    """
    output_dir = Path(output_dir)
    
    # создаем подпапку для эксперимента
    category = experiment_data.get("category", "X")
    timestamp = experiment_data.get("timestamp", datetime.now().isoformat())
    try:
        dt = datetime.fromisoformat(timestamp)
        folder_name = f"experiment_{category}_{dt.strftime('%Y%m%d_%H%M%S')}"
    except:
        folder_name = f"experiment_{category}_{sanitize_filename(timestamp)}"
    
    experiment_dir = output_dir / folder_name
    ensure_dir(experiment_dir)
    exported_files = []
    
    # обрабатываем каждую задачу
    for task_result in experiment_data.get("task_results", []):
        task_id = task_result.get("task_id", "unknown")
        task_name = task_result.get("task_name", "Без названия")
        model_id = task_result.get("model_id", "unknown_model")
        model_name = task_result.get("model_name", "Unknown Model")
        
        # создаем папку для задачи
        task_folder_name = f"{task_id}_{sanitize_filename(task_name)}"
        task_dir = experiment_dir / task_folder_name
        runs = task_result.get("runs", [])
        
        if include_all_runs:
            # сохраняем все прогоны
            for run in runs:
                if not run.get("success", False):
                    continue
                    
                run_index = run.get("run_index", 0)
                response = run.get("response", "")
                
                if not response.strip():
                    continue
                
                filename = f"run_{run_index + 1}.bsl"
                output_path = task_dir / filename
                
                metadata = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "model_id": model_id,
                    "model_name": model_name,
                    "run_index": run_index,
                    "seed": run.get("seed"),
                    "temperature": run.get("temperature", 0.0),
                    "response_hash": run.get("response_hash", ""),
                    "timestamp": timestamp
                }
                
                export_code_to_bsl(response, output_path, metadata)
                exported_files.append(str(output_path))
        else:
            for run in runs:
                if run.get("success", False) and run.get("response", "").strip():
                    response = run["response"]
                    run_index = run.get("run_index", 0)
                    
                    filename = f"code.bsl"
                    output_path = task_dir / filename
                    
                    metadata = {
                        "task_id": task_id,
                        "task_name": task_name,
                        "model_id": model_id,
                        "model_name": model_name,
                        "run_index": run_index,
                        "seed": run.get("seed"),
                        "temperature": run.get("temperature", 0.0),
                        "response_hash": run.get("response_hash", ""),
                        "timestamp": timestamp
                    }
                    
                    export_code_to_bsl(response, output_path, metadata)
                    exported_files.append(str(output_path))
                    break
        
        # также сохраняем сводку по детерминизму
        determinism = task_result.get("determinism", {})
        if determinism:
            summary = create_task_summary(task_result, timestamp)
            summary_path = task_dir / "summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            exported_files.append(str(summary_path))
    
    # создаем общий README для эксперимента
    readme_content = create_experiment_readme(experiment_data)
    readme_path = experiment_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    exported_files.append(str(readme_path))
    
    return {
        "experiment_dir": str(experiment_dir),
        "files_count": len(exported_files),
        "files": exported_files
    }


def create_task_summary(task_result: Dict[str, Any], timestamp: str) -> str:
    """Создать текстовую сводку по задаче"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Задача: {task_result.get('task_id', '')} - {task_result.get('task_name', '')}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Модель: {task_result.get('model_name', '')} ({task_result.get('model_id', '')})")
    lines.append(f"Время эксперимента: {timestamp}")
    lines.append("")
    
    # статистика по прогонам
    runs = task_result.get("runs", [])
    successful = sum(1 for r in runs if r.get("success", False))
    lines.append(f"Прогонов: {len(runs)} (успешных: {successful})")
    lines.append("")
    
    # детерминизм
    det = task_result.get("determinism", {})
    if det:
        lines.append("--- Детерминизм ---")
        total_runs = det.get("total_runs", 0)
        unique = det.get("unique_responses", 0)
        match_rate = det.get("match_rate", 0.0) * 100
        most_common_count = det.get("most_common_count", 0)
        
        lines.append(f"Процент совпадений: {match_rate:.1f}%")
        lines.append(f"Совпадающих ответов: {most_common_count} из {total_runs}")
        lines.append(f"Уникальных ответов: {unique}")
        
        if det.get("note"):
            lines.append(f"Примечание: {det.get('note')}")
        lines.append("")
        lines.append("Хеши ответов:")
        for i, h in enumerate(det.get("hashes", [])):
            lines.append(f"  Run {i+1}: {h}")
    
    lines.append("")
    
    # метрики
    lines.append("--- Метрики ---")
    lines.append(f"Всего токенов: {task_result.get('total_tokens', 0):,}")
    lines.append(f"Общая стоимость: ${task_result.get('total_cost', 0):.6f}")
    lines.append(f"Среднее время: {task_result.get('avg_time', 0):.2f} сек")
    
    return "\n".join(lines)


def create_experiment_readme(experiment_data: Dict[str, Any]) -> str:
    """Создать README для эксперимента"""
    lines = []
    lines.append(f"# Эксперимент: {experiment_data.get('experiment_name', 'Без названия')}")
    lines.append("")
    lines.append(f"**Категория:** {experiment_data.get('category', 'X')}")
    lines.append(f"**Дата:** {experiment_data.get('timestamp', 'N/A')}")
    lines.append("")
    
    lines.append("## Конфигурация")
    lines.append("")
    lines.append(f"- Модели: {', '.join(experiment_data.get('models_used', []))}")
    lines.append(f"- Задач: {experiment_data.get('tasks_count', 0)}")
    lines.append(f"- Прогонов на задачу: {experiment_data.get('runs_per_task', 0)}")
    lines.append("")
    
    lines.append("## Итоги")
    lines.append("")
    lines.append(f"- **Токенов:** {experiment_data.get('total_tokens', 0):,}")
    lines.append(f"- **Стоимость:** ${experiment_data.get('total_cost', 0):.6f}")
    lines.append(f"- **Время:** {experiment_data.get('total_time', 0):.1f} сек")
    lines.append("")
    
    lines.append("## Задачи")
    lines.append("")
    
    for task in experiment_data.get("task_results", []):
        task_id = task.get("task_id", "?")
        task_name = task.get("task_name", "Без названия")
        det = task.get("determinism", {})
        is_det = det.get("is_deterministic", False)
        det_symbol = "Да" if is_det else "Нет"
        
        lines.append(f"### {task_id}: {task_name}")
        lines.append("")
        lines.append(f"- Детерминизм: {det_symbol}")
        lines.append(f"- Токенов: {task.get('total_tokens', 0):,}")
        lines.append(f"- Стоимость: ${task.get('total_cost', 0):.6f}")
        lines.append("")
    
    lines.append("---")
    lines.append("*Сгенерировано AI-1C-Code-Generation-Benchmark*")
    
    return "\n".join(lines)


def export_from_json_file(
    json_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    include_all_runs: bool = False
) -> Dict[str, Any]:
    """
    Экспортировать код из JSON файла результатов
    
    Args:
        json_path: Путь к JSON файлу с результатами
        output_dir: Папка для экспорта (по умолчанию - code_outputs рядом с results)
        include_all_runs: Экспортировать все прогоны или только первый
        
    Returns:
        Информация об экспорте
    """
    json_path = Path(json_path)
    
    if output_dir is None:
        output_dir = json_path.parent.parent / "code_outputs"
    
    experiment_data = load_json(json_path)
    
    return export_experiment_code(experiment_data, output_dir, include_all_runs)
