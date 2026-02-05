"""
Benchmark Runner - ядро для запуска экспериментов

Функции:
- Загрузка конфигов
- Запуск генерации кода
- Сбор результатов
- Сохранение и экспорт
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from ..schemas.config import (
    ModelConfig, 
    TaskConfig, 
    GenerationParams,
    ExperimentConfig,
    MCPConfig
)
from ..schemas.results import (
    ChatMessage,
    RunResult,
    TaskResult, 
    ExperimentResult,
    DeterminismResult
)
from ..clients.openrouter import OpenRouterClient
from ..clients.mcp import MCPClient
from ..utils.file_ops import load_yaml, save_json, ensure_dir
from ..utils.hashing import compute_hash, compare_hashes
from ..utils.logging import log, log_section
from ..utils.code_export import export_experiment_code

from .context_loader import SmartContextLoader


class BenchmarkRunner:
    """
    Ядро бенчмарка — координирует весь процесс тестирования моделей
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        results_dir: str = "results",
        code_outputs_dir: str = "code_outputs",
        api_key: str = None,
        export_code: bool = True
    ):
        """
        Аргументы:
            config_dir: Папка с конфигурациями
            results_dir: Папка для сохранения результатов
            code_outputs_dir: Папка для экспорта .bsl файлов
            api_key: API ключ OpenRouter (или из переменной окружения)
            export_code: Экспортировать ли код в .bsl файлы
        """
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.code_outputs_dir = Path(code_outputs_dir)
        self.export_code = export_code
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("API ключ не предоставлен. Установите переменную окружения OPENROUTER_API_KEY.")
        
        self.experiment_config = load_yaml(self.config_dir / "experiment.yaml")
        self.models_config = load_yaml(self.config_dir / "models.yaml")
        self.llm = OpenRouterClient(
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://1c-benchmark.local",
                "X-Title": "AI-1C-Code-Generation-Benchmark"
            }
        )
        self.mcp: Optional[MCPClient] = None
        self.context_loader: Optional[SmartContextLoader] = None
        ensure_dir(self.results_dir)
    
    def get_model_config(self, model_key: str) -> ModelConfig:
        """Получить конфигурацию модели по ключу"""
        model_data = self.models_config["models"][model_key]
        meta = model_data.get("meta", {})
        
        return ModelConfig(
            id=model_data["id"],
            name=model_data["name"],
            context_window=meta.get("context_window", 0),
            price_input=meta.get("price_input", 0),
            price_output=meta.get("price_output", 0),
            supports_seed=meta.get("supports_seed", False),
            supports_tools=meta.get("supports_tools", False),
            determinism_param=meta.get("determinism_param", "temperature")
        )
    
    def load_tasks(self, category: str) -> List[TaskConfig]:
        """Загрузить задачи указанной категории"""
        config = load_yaml(self.config_dir / f"tasks_category_{category}.yaml")
        tasks = []
        for task_data in config.get("tasks", []):
            tasks.append(TaskConfig(
                id=task_data["id"],
                name=task_data["name"],
                difficulty=task_data.get("difficulty", "medium"),
                prompt=task_data["prompt"],
                expected_objects=task_data.get("expected_objects", [])
            ))
        
        return tasks
    
    def get_system_prompt(self, category: str) -> str:
        """Получить системный промпт для категории задач"""
        config = load_yaml(self.config_dir / f"tasks_category_{category}.yaml")
        return config.get("system_prompt", "")
    
    def get_generation_params(self, category: str, model_key: str) -> Dict[str, Any]:
        """
        Получить параметры генерации для модели из конфига категории
        
        Возвращает:
            Dict с ключами: temperature, seeds (или runs для Claude)
        """
        config = load_yaml(self.config_dir / f"tasks_category_{category}.yaml")
        generation = config.get("generation", {})
        model_params = generation.get("model_params", {}).get(model_key, {})
        
        return {
            "temperature": model_params.get("temperature", 0.0),
            "seeds": model_params.get("seeds", [42, 42, 999]),
            "runs": model_params.get("runs", 3),
            "max_tokens": generation.get("max_tokens", 4096)
        }
    
    async def init_mcp(self, use_mock: bool = True):
        """Инициализировать MCP клиент для работы с метаданными 1С"""
        if use_mock:
            # Импортируем mock только при необходимости
            from tests.mocks import MockMCPClient
            self.mcp = MockMCPClient()
        else:
            mcp_config = MCPConfig(
                url=self.experiment_config.get("mcp", {}).get("url", "http://localhost:8000")
            )
            self.mcp = MCPClient(mcp_config)
        
        await self.mcp.connect()
        self.context_loader = SmartContextLoader(
            mcp_client=self.mcp,
            llm_client=self.llm,
            analysis_model=self.models_config.get("analysis_model", "google/gemini-2.0-flash-001")
        )
    
    async def close_mcp(self):
        """Закрыть MCP соединение"""
        if self.mcp:
            await self.mcp.disconnect()
    
    def _run_generation(
        self,
        model_config: ModelConfig,
        messages: List[ChatMessage],
        run_index: int,
        seed: Optional[int] = None,
        temperature: float = 0.0
    ) -> RunResult:
        """
        Выполнить один прогон генерации кода
        """
        result = self.llm.chat_completion(
            model=model_config.id,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=4096
        )
        if result.success:
            # Вычисляем хеш ответа и стоимость
            response_hash = compute_hash(result.content)
            costs = self.llm.calculate_cost(
                model_config,
                result.tokens_input,
                result.tokens_output
            )
            return RunResult(
                run_index=run_index,
                seed=seed,
                temperature=temperature,
                response=result.content,
                response_hash=response_hash,
                tokens_input=result.tokens_input,
                tokens_output=result.tokens_output,
                tokens_total=result.tokens_total,
                elapsed_time=result.elapsed_time,
                cost_input=costs["input"],
                cost_output=costs["output"],
                cost_total=costs["total"],
                success=True
            )
        else:
            return RunResult(
                run_index=run_index,
                seed=seed,
                temperature=temperature,
                response="",
                response_hash="",
                success=False,
                error=result.error
            )
    
    def _analyze_determinism(self, runs: List[RunResult]) -> DeterminismResult:
        """Анализ детерминизма по результатам прогонов"""
        hashes = [r.response_hash for r in runs if r.success]
        
        if len(hashes) == 0:
            return DeterminismResult(
                total_runs=0,
                unique_responses=0,
                match_rate=0.0,
                most_common_hash="",
                most_common_count=0,
                hashes=[],
                note="Нет успешных прогонов"
            )
        
        comparison = compare_hashes(hashes)
        
        note = None
        if comparison["unique_count"] == 1:
            note = "Все ответы идентичны (100% детерминизм)"
        elif comparison["match_rate"] >= 0.8:
            note = "Высокая стабильность (≥80%)"
        elif comparison["match_rate"] < 0.5:
            note = "Низкая стабильность (<50%)"
        
        return DeterminismResult(
            total_runs=comparison["total_runs"],
            unique_responses=comparison["unique_count"],
            match_rate=comparison["match_rate"],
            most_common_hash=comparison["most_common_hash"],
            most_common_count=comparison["most_common_count"],
            hashes=hashes,
            note=note
        )
    
    async def run_task(
        self,
        task: TaskConfig,
        model_key: str,
        category: str
    ) -> TaskResult:
        """
        Выполнить задачу для одной модели
        """
        model_config = self.get_model_config(model_key)
        gen_params = self.get_generation_params(category, model_key)
        
        log(f" Задача: {task.name} | Модель: {model_config.name}")
        
        # Формируем сообщения
        system_prompt = self.get_system_prompt(category)
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=task.prompt)
        ]
        
        # Для категории B добавляем контекст метаданных
        context_loaded = False
        context_objects = []
        context_cost = 0.0
        
        if category == "B" and self.context_loader:
            log("  Загрузка контекста метаданных...")
            context_result = await self.context_loader.load_context(task.prompt)
            
            if context_result.success and context_result.context_text:
                # Добавляем контекст в системный промпт
                enhanced_system = f"""{system_prompt}

## Контекст метаданных конфигурации:
{context_result.context_text}"""
                messages[0] = ChatMessage(role="system", content=enhanced_system)
                
                context_loaded = True
                context_objects = context_result.objects_loaded
                context_cost = context_result.analysis_cost
                log(f"  Контекст загружен: {len(context_objects)} объектов")
        
        # Выполняем прогоны
        runs = []
        
        if model_config.determinism_param == "seed":
            # Для GPT/Gemini используем seed из конфига
            seeds = gen_params["seeds"]
            temperature = gen_params["temperature"]
            for i, seed in enumerate(seeds):
                log(f"  Прогон {i+1} (seed={seed}, temp={temperature})...")
                run_result = self._run_generation(
                    model_config, messages, i, 
                    seed=seed, temperature=temperature
                )
                runs.append(run_result)
                
                if run_result.success:
                    log(f"    Хеш: {run_result.response_hash[:16]}... | {run_result.tokens_total} токенов")
                else:
                    log(f"    ОШИБКА: {run_result.error}", "ERROR")
        else:
            # Для Claude используем temperature (без seed)
            num_runs = gen_params["runs"]
            base_temp = gen_params["temperature"]
            # Первые (n-1) прогонов с базовой температурой, последний с 0.7 для разнообразия
            temperatures = [base_temp] * (num_runs - 1) + [0.7]
            for i, temp in enumerate(temperatures):
                log(f"  Прогон {i+1} (temp={temp})...")
                run_result = self._run_generation(
                    model_config, messages, i,
                    temperature=temp
                )
                runs.append(run_result)
                
                if run_result.success:
                    log(f"    Хеш: {run_result.response_hash[:16]}... | {run_result.tokens_total} токенов")
                else:
                    log(f"    ОШИБКА: {run_result.error}", "ERROR")
        
        # Анализ детерминизма
        determinism = self._analyze_determinism(runs)
        
        # Агрегируем метрики
        successful_runs = [r for r in runs if r.success]
        total_tokens = sum(r.tokens_total for r in successful_runs)
        total_cost = sum(r.cost_total for r in successful_runs) + context_cost
        avg_time = sum(r.elapsed_time for r in successful_runs) / len(successful_runs) if successful_runs else 0
        
        # Логируем результат детерминизма
        match_pct = determinism.match_percent
        if match_pct == 100:
            log(f"   Детерминизм: {match_pct:.0f}% ({determinism.most_common_count}/{determinism.total_runs} идентичных)")
        elif match_pct >= 67:
            log(f"   Детерминизм: {match_pct:.0f}% ({determinism.most_common_count}/{determinism.total_runs} идентичных)")
        else:
            log(f"   Детерминизм: {match_pct:.0f}% ({determinism.most_common_count}/{determinism.total_runs} идентичных)")
        
        return TaskResult(
            task_id=task.id,
            task_name=task.name,
            model_id=model_config.id,
            model_name=model_config.name,
            context_loaded=context_loaded,
            context_objects=context_objects,
            context_analysis_cost=context_cost,
            runs=runs,
            determinism=determinism,
            total_tokens=total_tokens,
            total_cost=total_cost,
            avg_time=avg_time
        )
    
    async def run_experiment(
        self,
        category: str,
        model_keys: List[str] = None,
        task_ids: List[str] = None
    ) -> ExperimentResult:
        """
        Запустить полный эксперимент
        
        Аргументы:
            category: Категория задач ("A" или "B")
            model_keys: Ключи моделей (или все из конфига)
            task_ids: ID задач (или все из категории)
        """
        log_section(f" Запуск эксперимента: Категория {category}")
        
        # Определяем модели
        if model_keys is None:
            model_keys = list(self.models_config["models"].keys())
        
        # Загружаем задачи
        tasks = self.load_tasks(category)
        if task_ids:
            tasks = [t for t in tasks if t.id in task_ids]
        
        log(f"Модели: {model_keys}")
        log(f"Задачи: {[t.id for t in tasks]}")
        
        # Запускаем задачи
        task_results = []
        
        for model_key in model_keys:
            log_section(f"Модель: {model_key}", char="-")
            
            for task in tasks:
                result = await self.run_task(task, model_key, category)
                task_results.append(result)
        
        # Собираем итоги
        total_tokens = sum(r.total_tokens for r in task_results)
        total_cost = sum(r.total_cost for r in task_results)
        total_time = sum(r.avg_time * len(r.runs) for r in task_results)
        
        # Имя эксперимента из конфига
        exp_config = self.experiment_config.get("experiment", {})
        experiment_name = exp_config.get("name", "AI-1C-Benchmark")
        
        experiment = ExperimentResult(
            experiment_name=experiment_name,
            category=category,
            timestamp=datetime.now().isoformat(),
            models_used=model_keys,
            tasks_count=len(tasks),
            runs_per_task=3,
            task_results=task_results,
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_time=total_time
        )
        
        # Сохраняем результат
        self._save_result(experiment)
        
        log_section("Эксперимент завершён")
        log(f"Всего токенов: {total_tokens:,}")
        log(f"Общая стоимость: ${total_cost:.4f}")
        log(f"Общее время: {total_time:.1f}с")
        
        return experiment
    
    def _save_result(self, experiment: ExperimentResult):
        """Сохранить результат эксперимента в JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{experiment.category}_{timestamp}.json"
        path = self.results_dir / filename
        
        # Конвертируем в словарь
        data = self._to_dict(experiment)
        
        save_json(data, path)
        log(f"Результаты сохранены: {path}")
        
        # Экспортируем код в .bsl файлы для удобного просмотра
        if self.export_code:
            log(" Экспорт кода в .bsl файлы...")
            export_result = export_experiment_code(
                data, 
                self.code_outputs_dir,
                include_all_runs=True  # Сохраняем все прогоны для анализа
            )
            log(f"   Код экспортирован: {export_result['experiment_dir']}")
            log(f"   Файлов создано: {export_result['files_count']}")
    
    def _to_dict(self, obj: Any) -> Any:
        """Рекурсивно конвертировать dataclass в словарь"""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: self._to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [self._to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        return obj
