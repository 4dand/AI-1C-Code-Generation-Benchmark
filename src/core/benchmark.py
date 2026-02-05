"""
BenchmarkRunner — оркестратор запуска экспериментов

Центральный компонент фреймворка, отвечающий за:
- Загрузку конфигурации эксперимента
- Инициализацию клиентов (OpenRouter, MCP)
- Запуск задач и сбор результатов
- Анализ детерминизма
- Сохранение результатов

Использование:
    from src.core import BenchmarkRunner
    
    runner = BenchmarkRunner()
    await runner.init_mcp()
    
    result = await runner.run_experiment(
        category="B",
        model_keys=["gemini"],
        task_ids=["B1", "B2"]
    )
    
    await runner.close_mcp()
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..config.settings import get_settings
from ..clients.openrouter import OpenRouterClient
from ..clients.mcp import MCPClient
from ..schemas.models import ModelConfig, ModelsRegistry
from ..schemas.tasks import TaskConfig, TasksFile
from ..schemas.messages import ChatMessage
from ..schemas.results import (
    RunResult,
    TaskResult,
    ExperimentResult,
    DeterminismResult,
    ContextLoadResult,
)
from ..utils.file_ops import load_yaml, save_json, ensure_dir
from ..utils.hashing import compute_hash, compare_hashes
from ..utils.code_export import export_experiment_code
from .context_loader import AgenticContextLoader


logger = logging.getLogger(__name__)


def _format_cost(cost: float) -> str:
    """Форматировать стоимость для логов"""
    if cost < 0.0001:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


def _format_time(seconds: float) -> str:
    """Форматировать время для логов"""
    if seconds < 1:
        return f"{seconds*1000:.0f}мс"
    return f"{seconds:.1f}с"


# =============================================================================
# Основной класс
# =============================================================================

class BenchmarkRunner:
    """
    Оркестратор запуска бенчмарка
    
    Координирует весь процесс эксперимента:
    1. Загрузка конфигурации (модели, задачи)
    2. Инициализация клиентов API
    3. Запуск задач с несколькими прогонами
    4. Анализ детерминизма ответов
    5. Сохранение результатов
    
    Атрибуты:
        llm: Клиент OpenRouter для генерации
        mcp: Клиент MCP сервера (для категории B)
        context_loader: Агент загрузки контекста (для категории B)
        
    Example:
        runner = BenchmarkRunner()
        result = await runner.run_experiment("A", ["gemini"])
        print(f"Токенов: {result.total_tokens}")
    """
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
    ):
        """
        Инициализация BenchmarkRunner
        
        Args:
            config_dir: Путь к папке конфигов (если None — из Settings)
            results_dir: Путь к папке результатов (если None — из Settings)
        """
        self._settings = get_settings()
        
        self.config_dir = Path(config_dir or self._settings.paths.configs_dir)
        self.results_dir = Path(results_dir or self._settings.paths.raw_results_dir)
        
        # Клиенты API
        self.llm = OpenRouterClient.from_settings(self._settings)
        self.mcp: Optional[MCPClient] = None
        self.context_loader: Optional[AgenticContextLoader] = None
        
        # Кэш конфигурации
        self._models_registry: Optional[ModelsRegistry] = None
        self._tasks_cache: Dict[str, TasksFile] = {}
        
        logger.info(f"BenchmarkRunner инициализирован (configs: {self.config_dir})")
    
    # =========================================================================
    # Публичный API — инициализация
    # =========================================================================
    
    async def init_mcp(self, use_mock: bool = True) -> bool:
        """
        Инициализировать MCP клиент для категории B
        
        Args:
            use_mock: Использовать мок-сервер (True) или реальный (False)
            
        Returns:
            True если соединение установлено
        """
        if use_mock:
            logger.info("Инициализация MCP мок-сервера...")
            from tests.mocks.mcp_mock import MockMCPClient
            self.mcp = MockMCPClient()
        else:
            logger.info("Инициализация реального MCP клиента...")
            self.mcp = MCPClient.from_settings(self._settings)
        
        connected = await self.mcp.connect()
        
        if connected:
            self.context_loader = AgenticContextLoader(
                mcp_client=self.mcp,
                llm_client=self.llm,
            )
            logger.info("MCP клиент готов к работе")
        else:
            logger.error("Не удалось подключиться к MCP серверу")
        
        return connected
    
    async def close_mcp(self) -> None:
        """Закрыть соединение с MCP сервером"""
        if self.mcp:
            await self.mcp.disconnect()
            self.mcp = None
            self.context_loader = None
            logger.info("MCP соединение закрыто")
    
    # =========================================================================
    # Публичный API — запуск эксперимента
    # =========================================================================
    
    async def run_experiment(
        self,
        category: str,
        model_keys: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
    ) -> ExperimentResult:
        """
        Запустить эксперимент
        
        Args:
            category: Категория задач ("A" или "B")
            model_keys: Ключи моделей (если None — все модели)
            task_ids: ID задач (если None — все задачи категории)
            
        Returns:
            ExperimentResult с полными результатами
            
        Raises:
            ValueError: Если категория B требует MCP, но он не инициализирован
        """
        print()
        print("=" * 60)
        print(f" Запуск эксперимента: Категория {category}")
        print("=" * 60)
        
        # Загружаем конфигурацию
        tasks_config = self._load_tasks_config(category)
        models = self._get_models(model_keys)
        tasks = self._filter_tasks(tasks_config, task_ids)
        
        runs_per_task = self._get_runs_per_task(models, tasks_config)
        model_names = ', '.join(m.name for m in models.values())
        
        logger.info(
            f"Запуск эксперимента: категория={category}, "
            f"модели=[{model_names}], задач={len(tasks)}, прогонов={runs_per_task}"
        )
        
        print(f" Модели: {model_names}")
        print(f" Задач: {len(tasks)}")
        print(f" Прогонов на задачу: {runs_per_task}")
        print("=" * 60)
        print()
        
        # Проверяем MCP для категории B
        if tasks_config.category.requires_mcp and not self.mcp:
            raise ValueError(
                f"Категория {category} требует MCP сервер. "
                "Вызовите await runner.init_mcp() перед запуском."
            )
        
        # Создаём результат эксперимента
        timestamp = datetime.now().isoformat()
        experiment = ExperimentResult(
            experiment_name=f"experiment_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            category=category,
            timestamp=timestamp,
            models_used=[m.name for m in models.values()],
            tasks_count=len(tasks),
            runs_per_task=self._get_runs_per_task(models, tasks_config),
        )
        
        # Запускаем задачи
        start_time = datetime.now()
        total_tasks = len(tasks) * len(models)
        current_task = 0
        
        for task in tasks:
            for model_key, model in models.items():
                current_task += 1
                print(f"[{current_task}/{total_tasks}] {task.id}: {task.name}")
                print(f"     Модель: {model.name}")
                
                logger.info(f"Запуск задачи {task.id} ({task.name}) с моделью {model.name}")
                
                task_result = await self._run_task(
                    task=task,
                    model=model,
                    model_key=model_key,
                    tasks_config=tasks_config,
                )
                
                # Выводим результаты задачи
                self._print_task_summary(task_result)
                
                # Логируем результат задачи
                det_info = ""
                if task_result.determinism:
                    det_info = f", детерминизм={task_result.determinism.match_percent:.0f}%"
                logger.info(
                    f"Задача {task.id} завершена: "
                    f"токенов={task_result.total_tokens}, "
                    f"стоимость=${task_result.total_cost:.4f}{det_info}"
                )
                
                experiment.task_results.append(task_result)
        
        # Подсчитываем итоги
        experiment.calculate_totals()
        experiment.total_time = (datetime.now() - start_time).total_seconds()
        
        # Сохраняем результаты JSON
        result_path = self._save_experiment(experiment)
        logger.info(f"Результаты сохранены: {result_path}")
        
        # Экспортируем код в BSL файлы
        self._export_code(experiment)
        
        print()
        print("=" * 60)
        print(f" Эксперимент завершён")
        print(f"    Результаты: {result_path}")
        print("=" * 60)
        
        logger.info(
            f"Эксперимент {experiment.experiment_name} завершён: "
            f"задач={len(experiment.task_results)}, "
            f"токенов={experiment.total_tokens}, "
            f"стоимость=${experiment.total_cost:.4f}, "
            f"время={experiment.total_time:.1f}с"
        )
        
        return experiment
    
    # =========================================================================
    # Приватные методы — запуск задач
    # =========================================================================
    
    async def _run_task(
        self,
        task: TaskConfig,
        model: ModelConfig,
        model_key: str,
        tasks_config: TasksFile,
    ) -> TaskResult:
        """
        Выполнить одну задачу с несколькими прогонами
        
        Args:
            task: Конфигурация задачи
            model: Конфигурация модели
            model_key: Ключ модели (claude, gpt, gemini)
            tasks_config: Конфигурация категории задач
            
        Returns:
            TaskResult с результатами всех прогонов
        """
        result = TaskResult(
            task_id=task.id,
            task_name=task.name,
            model_id=model.id,
            model_name=model.name,
        )
        
        # Загружаем контекст для категории B
        context_text = ""
        if tasks_config.category.requires_mcp and self.context_loader:
            print(f"     Загрузка контекста...")
            
            self.context_loader.reset_metrics()
            context_result = await self.context_loader.load_context(task.prompt)
            
            if context_result.success:
                context_text = context_result.context_text
                result.context_loaded = True
                result.context_objects = context_result.objects_loaded
                result.context_analysis_cost = context_result.analysis_cost
                
                print(f"     Контекст: {context_result.objects_count} объектов, "
                      f"{_format_cost(context_result.analysis_cost)}")
        
        # Формируем сообщения
        messages = self._build_messages(task, tasks_config, context_text)
        
        # Получаем параметры генерации (включая seeds из категории)
        gen_params = self._get_generation_params(model, model_key, tasks_config)
        
        # Определяем количество прогонов и seeds
        # Приоритет: category params > model params
        category_seeds = gen_params.get("seeds")
        category_runs = gen_params.get("runs")
        
        if category_seeds:
            runs_count = len(category_seeds)
        elif category_runs:
            runs_count = category_runs
        else:
            runs_count = model.runs_count
        
        # Запускаем прогоны
        hashes: List[str] = []
        
        for run_index in range(runs_count):
            # Получаем seed: сначала из категории, потом из модели
            if category_seeds and run_index < len(category_seeds):
                seed = category_seeds[run_index]
            else:
                seed = model.get_seed_for_run(run_index)
            
            run_result = self._execute_run(
                run_index=run_index,
                messages=messages,
                model=model,
                seed=seed,
                gen_params=gen_params,
            )
            
            # Лог прогона
            status = "+" if run_result.success else "x"
            seed_str = f"seed={seed}" if seed else "no-seed"
            print(f"     [{status}] Прогон {run_index + 1}: {seed_str}, "
                  f"{run_result.tokens_total} токенов, {_format_time(run_result.elapsed_time)}")
            
            if run_result.error:
                print(f"       Ошибка: {run_result.error[:60]}...")
            
            result.runs.append(run_result)
            
            if run_result.success:
                hashes.append(run_result.response_hash)
        
        # Анализ детерминизма
        if hashes:
            result.determinism = self._analyze_determinism(hashes)
        
        # Пересчёт агрегатов
        result.calculate_aggregates()
        
        return result
    
    def _print_task_summary(self, task_result: TaskResult) -> None:
        """Вывести краткую сводку по задаче"""
        det_str = ""
        if task_result.determinism:
            rate = task_result.determinism.match_rate * 100
            unique = task_result.determinism.unique_responses
            det_str = f", детерминизм: {rate:.0f}% ({unique} уник.)"
        
        print(f"     Итого: {task_result.total_tokens} токенов, "
              f"{_format_cost(task_result.total_cost)}{det_str}")
        print()
    
    def _execute_run(
        self,
        run_index: int,
        messages: List[ChatMessage],
        model: ModelConfig,
        seed: Optional[int],
        gen_params: Dict[str, Any],
    ) -> RunResult:
        """
        Выполнить один прогон генерации
        
        Args:
            run_index: Индекс прогона
            messages: Список сообщений
            model: Конфигурация модели
            seed: Seed для детерминизма
            gen_params: Параметры генерации
            
        Returns:
            RunResult с результатом
        """
        temperature = gen_params.get("temperature", model.generation.temperature)
        max_tokens = gen_params.get("max_tokens", 4096)
        
        # Вызываем LLM
        llm_result = self.llm.chat_completion(
            model=model.id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        
        # Формируем результат прогона
        run = RunResult(
            run_index=run_index,
            seed=seed,
            temperature=temperature,
            success=llm_result.success,
            response=llm_result.content,
            tokens_input=llm_result.tokens_input,
            tokens_output=llm_result.tokens_output,
            tokens_total=llm_result.tokens_total,
            elapsed_time=llm_result.elapsed_time,
            error=llm_result.error,
        )
        
        # Хеш ответа для детерминизма
        if llm_result.success and llm_result.content:
            run.response_hash = compute_hash(
                llm_result.content,
                normalize=self._settings.hashing.normalize,
                algorithm=self._settings.hashing.algorithm,
            )
        
        # Расчёт стоимости
        run.cost_input, run.cost_output, run.cost_total = self._calculate_cost(
            model, run.tokens_input, run.tokens_output
        )
        
        return run
    
    # =========================================================================
    # Приватные методы — построение сообщений
    # =========================================================================
    
    def _build_messages(
        self,
        task: TaskConfig,
        tasks_config: TasksFile,
        context_text: str = "",
    ) -> List[ChatMessage]:
        """
        Построить список сообщений для LLM
        
        Args:
            task: Задача
            tasks_config: Конфигурация категории
            context_text: Контекст метаданных (для категории B)
            
        Returns:
            Список ChatMessage
        """
        # Системный промпт
        system_content = tasks_config.system_prompt
        
        # Добавляем контекст если есть
        if context_text:
            system_content = f"{system_content}\n\n# Доступные метаданные:\n{context_text}"
        
        messages = [
            ChatMessage.system(system_content),
            ChatMessage.user(task.prompt),
        ]
        
        return messages
    
    # =========================================================================
    # Приватные методы — анализ
    # =========================================================================
    
    def _analyze_determinism(self, hashes: List[str]) -> DeterminismResult:
        """
        Проанализировать детерминизм ответов
        
        Args:
            hashes: Список хешей ответов
            
        Returns:
            DeterminismResult с анализом
        """
        stats = compare_hashes(hashes)
        
        return DeterminismResult(
            total_runs=stats["total_runs"],
            unique_responses=stats["unique_count"],
            match_rate=stats["match_rate"],
            most_common_hash=stats["most_common_hash"],
            most_common_count=stats["most_common_count"],
            hashes=hashes,
        )
    
    def _calculate_cost(
        self,
        model: ModelConfig,
        tokens_input: int,
        tokens_output: int,
    ) -> tuple[float, float, float]:
        """
        Рассчитать стоимость запроса
        
        Args:
            model: Конфигурация модели
            tokens_input: Входные токены
            tokens_output: Выходные токены
            
        Returns:
            Кортеж (cost_input, cost_output, cost_total)
        """
        # Цены указаны за 1M токенов
        cost_input = (tokens_input / 1_000_000) * model.meta.price_input
        cost_output = (tokens_output / 1_000_000) * model.meta.price_output
        cost_total = cost_input + cost_output
        
        return cost_input, cost_output, cost_total
    
    # =========================================================================
    # Приватные методы — конфигурация
    # =========================================================================
    
    def _load_models_registry(self) -> ModelsRegistry:
        """Загрузить реестр моделей"""
        if self._models_registry is not None:
            return self._models_registry
        
        models_path = self.config_dir / "models.yaml"
        raw_data = load_yaml(models_path)
        
        self._models_registry = ModelsRegistry(models=raw_data.get("models", {}))
        return self._models_registry
    
    def _load_tasks_config(self, category: str) -> TasksFile:
        """Загрузить конфигурацию задач категории"""
        if category in self._tasks_cache:
            return self._tasks_cache[category]
        
        tasks_path = self.config_dir / f"tasks_category_{category}.yaml"
        raw_data = load_yaml(tasks_path)
        
        tasks_config = TasksFile(**raw_data)
        self._tasks_cache[category] = tasks_config
        
        return tasks_config
    
    def _get_models(
        self, 
        model_keys: Optional[List[str]] = None
    ) -> Dict[str, ModelConfig]:
        """
        Получить модели для эксперимента
        
        Args:
            model_keys: Ключи моделей (если None — все)
            
        Returns:
            Словарь {key: ModelConfig}
        """
        registry = self._load_models_registry()
        
        if model_keys is None:
            return dict(registry)
        
        result = {}
        for key in model_keys:
            model = registry.get(key)
            if model:
                result[key] = model
            else:
                logger.warning(f"Модель '{key}' не найдена в реестре")
        
        return result
    
    def _filter_tasks(
        self,
        tasks_config: TasksFile,
        task_ids: Optional[List[str]] = None,
    ) -> List[TaskConfig]:
        """
        Отфильтровать задачи
        
        Args:
            tasks_config: Конфигурация категории
            task_ids: ID задач (если None — все)
            
        Returns:
            Список TaskConfig
        """
        if task_ids is None:
            return tasks_config.tasks
        
        return [t for t in tasks_config.tasks if t.id in task_ids]
    
    def _get_generation_params(
        self,
        model: ModelConfig,
        model_key: str,
        tasks_config: TasksFile,
    ) -> Dict[str, Any]:
        """
        Получить параметры генерации для модели
        
        Приоритет: category config > model config > defaults
        
        Args:
            model: Конфигурация модели
            model_key: Ключ модели
            tasks_config: Конфигурация категории
            
        Returns:
            Словарь параметров
        """
        params = {
            "temperature": model.generation.temperature,
            "max_tokens": 4096,
        }
        
        # Переопределения из категории
        if tasks_config.generation.max_tokens:
            params["max_tokens"] = tasks_config.generation.max_tokens
        
        category_params = tasks_config.generation.get_model_params(model_key)
        params.update(category_params)
        
        return params
    
    def _get_runs_per_task(
        self, 
        models: Dict[str, ModelConfig],
        tasks_config: Optional[TasksFile] = None,
    ) -> int:
        """
        Получить количество прогонов на задачу
        
        Учитывает параметры категории (seeds/runs) если они заданы.
        
        Args:
            models: Словарь моделей
            tasks_config: Конфигурация категории (опционально)
            
        Returns:
            Максимальное количество прогонов среди моделей
        """
        if not models:
            return 0
        
        max_runs = 0
        for model_key, model in models.items():
            # Проверяем параметры категории для модели
            if tasks_config:
                category_params = tasks_config.generation.get_model_params(model_key)
                category_seeds = category_params.get("seeds")
                category_runs = category_params.get("runs")
                
                if category_seeds:
                    runs = len(category_seeds)
                elif category_runs:
                    runs = category_runs
                else:
                    runs = model.runs_count
            else:
                runs = model.runs_count
            
            max_runs = max(max_runs, runs)
        
        return max_runs
    
    # =========================================================================
    # Приватные методы — сохранение и экспорт
    # =========================================================================
    
    def _save_experiment(self, experiment: ExperimentResult) -> Path:
        """
        Сохранить результаты эксперимента в JSON
        
        Args:
            experiment: Результат эксперимента
            
        Returns:
            Путь к сохранённому файлу
        """
        ensure_dir(self.results_dir)
        
        filename = f"{experiment.experiment_name}.json"
        filepath = self.results_dir / filename
        
        save_json(experiment.model_dump(), filepath)
        
        return filepath
    
    def _export_code(self, experiment: ExperimentResult) -> None:
        """
        Экспортировать сгенерированный код в BSL файлы
        
        Args:
            experiment: Результат эксперимента
        """
        if not self._settings.export.code_to_bsl:
            return
        
        output_dir = Path(self._settings.paths.code_outputs_dir)
        
        try:
            result = export_experiment_code(
                experiment_data=experiment.model_dump(),
                output_dir=output_dir,
                include_all_runs=True  # Сохраняем все прогоны для анализа детерминизма
            )
            
            print(f"     Код экспортирован: {result['files_count']} файлов")
            print(f"        {result['experiment_dir']}")
            
        except Exception as e:
            logger.warning(f"Ошибка экспорта кода: {e}")
