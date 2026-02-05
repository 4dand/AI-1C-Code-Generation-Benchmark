"""
SMOP — логика экспертной оценки

Отвечает за:
- FR-201: Хранение оценок S, M, O, P для каждого прогона
- FR-202: Валидация значений (0, 2, 4, 6, 8, 10)
- FR-203: Расчёт интегрального показателя Q
- FR-204: Загрузка критериев из smop_criteria.yaml
- FR-205: Сохранение оценок в evaluations/
- FR-206: Поддержка нескольких экспертов
- FR-207: Редактирование ранее проставленных оценок
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal

from ..utils.file_ops import load_yaml, load_json, save_json, ensure_dir
from .schemas import (
    SMOPScores,
    RunEvaluation,
    TaskEvaluation,
    ExperimentEvaluation,
    VALID_SCORES,
)


logger = logging.getLogger(__name__)


class SMOPCriteria:
    """
    Загрузчик и хранилище критериев оценки SMOP
    
    Загружает критерии из configs/smop_criteria.yaml
    и предоставляет доступ к описаниям для интерфейса.
    
    FR-204: Загрузка критериев оценки
    """
    
    def __init__(self, config_path: str = "configs/smop_criteria.yaml"):
        """
        Инициализация критериев
        
        Args:
            config_path: Путь к файлу критериев
        """
        self.config_path = Path(config_path)
        self._data: Dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Загрузить критерии из YAML"""
        if not self.config_path.exists():
            logger.warning(f"Файл критериев не найден: {self.config_path}")
            self._data = self._default_criteria()
            return
        
        try:
            self._data = load_yaml(self.config_path)
            logger.debug(f"Критерии SMOP загружены из {self.config_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки критериев: {e}")
            self._data = self._default_criteria()
    
    def _default_criteria(self) -> Dict[str, Any]:
        """Минимальные критерии по умолчанию"""
        return {
            "smop": {
                "S": {"name": "Синтаксис", "criteria": {}},
                "M": {"name": "Семантика", "criteria": {}},
                "O": {"name": "Оптимальность", "criteria": {}},
                "P": {"name": "Платформа", "criteria": {}},
            },
            "valid_scores": [0, 2, 4, 6, 8, 10],
            "quality_thresholds": {"high": 8, "acceptable": 5, "low": 0},
        }
    
    @property
    def valid_scores(self) -> List[int]:
        """Допустимые значения оценок"""
        return self._data.get("valid_scores", [0, 2, 4, 6, 8, 10])
    
    @property
    def quality_thresholds(self) -> Dict[str, float]:
        """Пороговые значения качества"""
        return self._data.get("quality_thresholds", {"high": 8, "acceptable": 5, "low": 0})
    
    def get_metric_info(self, metric: Literal["S", "M", "O", "P"]) -> Dict[str, Any]:
        """
        Получить информацию о метрике
        
        Args:
            metric: Код метрики (S, M, O, P)
            
        Returns:
            Словарь с name, description, criteria
        """
        smop = self._data.get("smop", {})
        return smop.get(metric, {"name": metric, "description": "", "criteria": {}})
    
    def get_criterion_description(
        self,
        metric: Literal["S", "M", "O", "P"],
        score: int
    ) -> str:
        """
        Получить описание критерия для конкретной оценки
        
        Args:
            metric: Код метрики
            score: Значение оценки
            
        Returns:
            Описание критерия
            
        FR-507: Отображение критериев оценки
        """
        info = self.get_metric_info(metric)
        criteria = info.get("criteria", {})
        return criteria.get(score, criteria.get(str(score), ""))
    
    def get_all_criteria_for_metric(
        self,
        metric: Literal["S", "M", "O", "P"]
    ) -> Dict[int, str]:
        """
        Получить все критерии для метрики
        
        Returns:
            Словарь {оценка: описание}
        """
        info = self.get_metric_info(metric)
        criteria = info.get("criteria", {})
        # Приводим ключи к int
        return {int(k): v for k, v in criteria.items()}


class SMOPEvaluator:
    """
    Менеджер оценок SMOP
    
    Управляет загрузкой, сохранением и редактированием оценок.
    Обеспечивает автосохранение и восстановление сессии.
    
    Example:
        evaluator = SMOPEvaluator("evaluations")
        
        # Загрузить или создать оценку
        evaluation = evaluator.load_or_create("experiment_B_123", experiment_result)
        
        # Установить оценку
        evaluator.set_score(evaluation, "B1", "google/gemini", 0, "S", 10)
        
        # Сохранить
        evaluator.save(evaluation)
    """
    
    def __init__(
        self,
        evaluations_dir: str = "evaluations",
        criteria_path: str = "configs/smop_criteria.yaml"
    ):
        """
        Инициализация
        
        Args:
            evaluations_dir: Директория для сохранения оценок
            criteria_path: Путь к файлу критериев
        """
        self.evaluations_dir = Path(evaluations_dir)
        self.criteria = SMOPCriteria(criteria_path)
        
        # Создаём директорию если не существует
        ensure_dir(self.evaluations_dir)
    
    def get_evaluation_path(self, experiment_id: str, evaluator_id: str = "expert_01") -> Path:
        """Получить путь к файлу оценки"""
        return self.evaluations_dir / f"{experiment_id}_{evaluator_id}_evaluation.json"
    
    def exists(self, experiment_id: str, evaluator_id: str = "expert_01") -> bool:
        """Проверить существует ли оценка"""
        return self.get_evaluation_path(experiment_id, evaluator_id).exists()
    
    def load(
        self,
        experiment_id: str,
        evaluator_id: str = "expert_01"
    ) -> Optional[ExperimentEvaluation]:
        """
        Загрузить существующую оценку
        
        Args:
            experiment_id: ID эксперимента
            evaluator_id: ID эксперта
            
        Returns:
            ExperimentEvaluation или None
            
        NFR-05: Восстановление сессии
        """
        path = self.get_evaluation_path(experiment_id, evaluator_id)
        
        if not path.exists():
            logger.debug(f"Файл оценки не найден: {path}")
            return None
        
        try:
            data = load_json(path)
            evaluation = ExperimentEvaluation(**data)
            
            logger.info(
                f"Загружена оценка: {experiment_id}, "
                f"прогресс={evaluation.progress_percent:.0f}%"
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Ошибка загрузки оценки {experiment_id}: {e}")
            return None
    
    def save(self, evaluation: ExperimentEvaluation) -> Path:
        """
        Сохранить оценку
        
        Args:
            evaluation: Оценка для сохранения
            
        Returns:
            Путь к сохранённому файлу
            
        FR-205: Сохранение оценок
        NFR-04: Автосохранение (вызывается из dashboard)
        """
        # Обновляем время и статус
        evaluation.update_status()
        
        path = self.get_evaluation_path(evaluation.experiment_id, evaluation.evaluator_id)
        
        # Сериализуем с использованием pydantic
        data = evaluation.model_dump()
        save_json(data, path)
        
        logger.debug(f"Оценка сохранена: {path}")
        
        return path
    
    def set_score(
        self,
        evaluation: ExperimentEvaluation,
        task_id: str,
        model_id: str,
        run_index: int,
        metric: Literal["S", "M", "O", "P"],
        score: int,
        comment: Optional[str] = None
    ) -> bool:
        """
        Установить оценку для прогона
        
        Args:
            evaluation: Объект оценки эксперимента
            task_id: ID задачи
            model_id: ID модели
            run_index: Индекс прогона
            metric: Метрика (S, M, O, P)
            score: Значение оценки
            comment: Комментарий (опционально)
            
        Returns:
            True если оценка установлена успешно
            
        FR-201, FR-202, FR-207: Установка и валидация оценок
        """
        # Валидация
        if score not in VALID_SCORES:
            logger.error(f"Недопустимое значение оценки: {score}")
            return False
        
        # Находим задачу
        task = evaluation.get_task(task_id, model_id)
        if not task:
            logger.error(f"Задача не найдена: {task_id}/{model_id}")
            return False
        
        # Находим прогон
        run = task.get_run(run_index)
        if not run:
            logger.error(f"Прогон не найден: {run_index}")
            return False
        
        # Устанавливаем оценку
        setattr(run.scores, metric, score)
        run.mark_evaluated()
        
        if comment is not None:
            run.comment = comment
        
        # Обновляем статус
        evaluation.update_status()
        
        logger.debug(f"Оценка установлена: {task_id}/{run_index} {metric}={score}")
        
        return True
    
    def set_all_scores(
        self,
        evaluation: ExperimentEvaluation,
        task_id: str,
        model_id: str,
        run_index: int,
        scores: Dict[str, int],
        comment: str = ""
    ) -> bool:
        """
        Установить все оценки SMOP для прогона
        
        Args:
            evaluation: Объект оценки
            task_id: ID задачи
            model_id: ID модели
            run_index: Индекс прогона
            scores: Словарь {"S": 10, "M": 8, ...}
            comment: Комментарий
            
        Returns:
            True если успешно
        """
        # Валидация всех оценок
        for metric, score in scores.items():
            if metric not in ["S", "M", "O", "P"]:
                logger.error(f"Неизвестная метрика: {metric}")
                return False
            if score not in VALID_SCORES:
                logger.error(f"Недопустимое значение {metric}={score}")
                return False
        
        # Находим задачу и прогон
        task = evaluation.get_task(task_id, model_id)
        if not task:
            return False
        
        run = task.get_run(run_index)
        if not run:
            return False
        
        # Устанавливаем все оценки
        for metric, score in scores.items():
            setattr(run.scores, metric, score)
        
        run.comment = comment
        run.mark_evaluated()
        evaluation.update_status()
        
        return True
    
    def get_progress(self, evaluation: ExperimentEvaluation) -> Dict[str, Any]:
        """
        Получить информацию о прогрессе оценки
        
        Returns:
            Словарь с метриками прогресса
            
        FR-509: Индикация прогресса
        """
        return {
            "total_runs": evaluation.total_runs,
            "evaluated_runs": evaluation.evaluated_runs,
            "progress_percent": evaluation.progress_percent,
            "status": evaluation.status,
            "is_complete": evaluation.is_complete,
            "started_at": evaluation.started_at,
            "last_modified_at": evaluation.last_modified_at,
        }
    
    def list_evaluations(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получить список всех оценок
        
        Args:
            experiment_id: Фильтр по эксперименту (опционально)
            
        Returns:
            Список информации об оценках
        """
        evaluations = []
        
        for json_file in self.evaluations_dir.glob("*_evaluation.json"):
            try:
                data = load_json(json_file)
                
                if experiment_id and data.get("experiment_id") != experiment_id:
                    continue
                
                evaluation = ExperimentEvaluation(**data)
                
                evaluations.append({
                    "experiment_id": evaluation.experiment_id,
                    "evaluator_id": evaluation.evaluator_id,
                    "status": evaluation.status,
                    "progress_percent": evaluation.progress_percent,
                    "total_runs": evaluation.total_runs,
                    "evaluated_runs": evaluation.evaluated_runs,
                    "started_at": evaluation.started_at,
                    "last_modified_at": evaluation.last_modified_at,
                    "path": str(json_file),
                })
                
            except Exception as e:
                logger.warning(f"Ошибка чтения оценки {json_file}: {e}")
                continue
        
        return evaluations


# Глобальный экземпляр критериев
_criteria: Optional[SMOPCriteria] = None


def get_smop_criteria() -> SMOPCriteria:
    """Получить глобальный экземпляр критериев SMOP"""
    global _criteria
    if _criteria is None:
        _criteria = SMOPCriteria()
    return _criteria
