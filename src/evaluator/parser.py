"""
Parser — загрузка и парсинг результатов экспериментов

Отвечает за:
- FR-101: Загрузка JSON-файлов из raw_results/
- FR-102: Валидация структуры согласно schemas/results.py
- FR-103: Извлечение кода из response каждого прогона
- FR-104: Группировка прогонов по задачам и моделям
- FR-105: Обработка ошибок с логированием
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from ..schemas.results import ExperimentResult, TaskResult, RunResult
from ..utils.file_ops import load_json
from .schemas import (
    ExperimentEvaluation,
    TaskEvaluation,
    RunEvaluation,
    SMOPScores,
)


logger = logging.getLogger(__name__)


class ExperimentParser:
    """
    Парсер результатов экспериментов из raw_results/
    
    Загружает JSON-файлы экспериментов и преобразует их
    в структуру для оценки SMOP.
    
    Example:
        parser = ExperimentParser("raw_results")
        experiments = parser.list_experiments()
        
        result = parser.load_experiment("experiment_B_20260205_221310")
        evaluation = parser.create_evaluation(result)
    """
    
    def __init__(self, results_dir: str = "raw_results"):
        """
        Инициализация парсера
        
        Args:
            results_dir: Путь к директории с результатами
        """
        self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            logger.warning(f"Директория raw_results не существует: {self.results_dir}")
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        Получить список доступных экспериментов
        
        Returns:
            Список словарей с информацией:
            - id: ID эксперимента
            - path: Путь к файлу
            - category: Категория (A/B)
            - timestamp: Время эксперимента
            - models: Список моделей
            - tasks_count: Количество задач
        
        FR-502: Отображение списка экспериментов
        """
        experiments = []
        
        if not self.results_dir.exists():
            return experiments
        
        for json_file in sorted(self.results_dir.glob("*.json")):
            try:
                info = self._extract_experiment_info(json_file)
                if info:
                    experiments.append(info)
            except Exception as e:
                logger.warning(f"Ошибка при чтении {json_file.name}: {e}")
                continue
        
        return experiments
    
    def _extract_experiment_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """Извлечь базовую информацию об эксперименте без полной загрузки"""
        try:
            data = load_json(path)
            
            return {
                "id": data.get("experiment_name", path.stem),
                "path": str(path),
                "category": data.get("category", "?"),
                "timestamp": data.get("timestamp", ""),
                "models": data.get("models_used", []),
                "tasks_count": data.get("tasks_count", 0),
                "runs_per_task": data.get("runs_per_task", 0),
                "total_tokens": data.get("total_tokens", 0),
                "total_cost": data.get("total_cost", 0.0),
            }
        except Exception as e:
            logger.debug(f"Не удалось извлечь информацию из {path}: {e}")
            return None
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """
        Загрузить эксперимент по ID
        
        Args:
            experiment_id: ID эксперимента (имя файла без .json)
            
        Returns:
            ExperimentResult или None если не найден
            
        FR-101, FR-102: Загрузка и валидация JSON
        """
        # Ищем файл
        json_path = self.results_dir / f"{experiment_id}.json"
        
        if not json_path.exists():
            logger.error(f"Файл эксперимента не найден: {json_path}")
            return None
        
        try:
            data = load_json(json_path)
            result = ExperimentResult(**data)
            
            logger.info(
                f"Загружен эксперимент: {result.experiment_name}, "
                f"задач={len(result.task_results)}, "
                f"моделей={len(result.models_used)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка валидации эксперимента {experiment_id}: {e}")
            return None
    
    def create_evaluation(
        self,
        experiment: ExperimentResult,
        evaluator_id: str = "expert_01"
    ) -> ExperimentEvaluation:
        """
        Создать структуру для оценки на основе результатов эксперимента
        
        Args:
            experiment: Загруженный эксперимент
            evaluator_id: ID эксперта
            
        Returns:
            ExperimentEvaluation готовый для заполнения оценок
            
        FR-104: Группировка прогонов по задачам и моделям
        """
        evaluation = ExperimentEvaluation(
            experiment_id=experiment.experiment_name,
            evaluator_id=evaluator_id,
        )
        
        for task_result in experiment.task_results:
            task_eval = TaskEvaluation(
                task_id=task_result.task_id,
                model_id=task_result.model_id,
                model_name=task_result.model_name,
            )
            
            # Создаём RunEvaluation для каждого прогона
            for run in task_result.runs:
                run_eval = RunEvaluation(
                    run_index=run.run_index,
                    response_hash=run.response_hash,
                )
                task_eval.runs.append(run_eval)
            
            evaluation.tasks.append(task_eval)
        
        logger.debug(
            f"Создана структура оценки: задач={len(evaluation.tasks)}, "
            f"прогонов={evaluation.total_runs}"
        )
        
        return evaluation
    
    def get_run_details(
        self,
        experiment: ExperimentResult,
        task_id: str,
        model_id: str,
        run_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Получить детали прогона для отображения в интерфейсе
        
        Args:
            experiment: Эксперимент
            task_id: ID задачи
            model_id: ID модели
            run_index: Индекс прогона
            
        Returns:
            Словарь с деталями прогона или None
            
        FR-503, FR-504: Просмотр прогонов и кода
        """
        # Находим задачу
        task_result = None
        for tr in experiment.task_results:
            if tr.task_id == task_id and tr.model_id == model_id:
                task_result = tr
                break
        
        if not task_result:
            return None
        
        # Находим прогон
        run = None
        for r in task_result.runs:
            if r.run_index == run_index:
                run = r
                break
        
        if not run:
            return None
        
        # Извлекаем код из ответа
        code = self._extract_code(run.response)
        
        return {
            "task_id": task_id,
            "task_name": task_result.task_name,
            "model_id": model_id,
            "model_name": task_result.model_name,
            "run_index": run_index,
            "total_runs": len(task_result.runs),
            "response_hash": run.response_hash,
            "code": code,
            "full_response": run.response,
            "tokens_total": run.tokens_total,
            "elapsed_time": run.elapsed_time,
            "success": run.success,
            "error": run.error,
            # Контекст если есть
            "context_objects": task_result.context_objects,
            "context_loaded": task_result.context_loaded,
            # Детерминизм
            "determinism": task_result.determinism.model_dump() if task_result.determinism else None,
        }
    
    def _extract_code(self, response: str) -> str:
        """
        Извлечь код из ответа модели
        
        Ищет блоки кода в markdown-формате:
        ```bsl или ```1c или просто ```
        
        FR-103: Извлечение сгенерированного кода
        """
        if not response:
            return ""
        
        # Паттерны блоков кода
        code_markers = ["```bsl", "```1c", "```1С", "```"]
        
        best_code = ""
        
        for marker in code_markers:
            if marker in response:
                parts = response.split(marker)
                for i, part in enumerate(parts):
                    if i == 0:
                        continue
                    # Находим закрывающий ```
                    if "```" in part:
                        code_block = part.split("```")[0].strip()
                        # Предпочитаем более длинные блоки
                        if len(code_block) > len(best_code):
                            best_code = code_block
        
        # Если не нашли блоки кода, возвращаем весь ответ
        if not best_code:
            return response.strip()
        
        return best_code
    
    def get_navigation_info(
        self,
        experiment: ExperimentResult
    ) -> List[Dict[str, Any]]:
        """
        Получить информацию для навигации по эксперименту
        
        Returns:
            Список задач с прогонами для навигации
            
        FR-508: Навигация между прогонами
        """
        nav = []
        
        for task_result in experiment.task_results:
            nav.append({
                "task_id": task_result.task_id,
                "task_name": task_result.task_name,
                "model_id": task_result.model_id,
                "model_name": task_result.model_name,
                "runs_count": len(task_result.runs),
                "run_indices": [r.run_index for r in task_result.runs],
            })
        
        return nav


def list_available_experiments(results_dir: str = "raw_results") -> List[Dict[str, Any]]:
    """
    Утилита для получения списка экспериментов
    
    Args:
        results_dir: Путь к директории с результатами
        
    Returns:
        Список информации об экспериментах
    """
    parser = ExperimentParser(results_dir)
    return parser.list_experiments()
