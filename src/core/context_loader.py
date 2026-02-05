"""
Agentic Context Loader — агент для сбора контекста метаданных 1С

Модуль реализует агентный подход к сбору контекста:
- Модель сама выбирает нужные объекты метаданных через MCP tools
- Поддержка кэширования результатов
- Интеграция с Settings для конфигурации

Логика работы:
1. Получаем список tools от MCP сервера (tools/list)
2. Конвертируем в формат OpenAI tools
3. Даём модели задачу + tools
4. Модель вызывает tools → проксируем через tools/call
5. Собираем контекст из ответов

Использование:
    from src.config import get_settings
    from src.clients import MCPClient, OpenRouterClient
    from src.core import AgenticContextLoader
    
    settings = get_settings()
    mcp = MCPClient.from_settings(settings)
    llm = OpenRouterClient.from_settings(settings)
    
    loader = AgenticContextLoader(mcp, llm)
    result = await loader.load_context("Написать запрос остатков товаров")
"""

import json
import logging
from typing import List, Dict, Optional, Any

from ..clients.openrouter import OpenRouterClient
from ..clients.mcp import MCPClient
from ..config.settings import get_settings
from ..schemas.messages import ChatMessage
from ..schemas.results import ContextLoadResult


logger = logging.getLogger(__name__)


# =============================================================================
# Константы — определение finish tool
# =============================================================================

FINISH_TOOL_NAME = "finish_research"

FINISH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": FINISH_TOOL_NAME,
        "description": "Завершить исследование метаданных. Вызови когда собрал достаточно информации для написания кода.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Краткое резюме: какие объекты будешь использовать и почему"
                }
            },
            "required": ["summary"]
        }
    }
}


# =============================================================================
# Основной класс
# =============================================================================

class AgenticContextLoader:
    """
    Агентный загрузчик контекста метаданных 1С
    
    Использует LLM агента для интеллектуального выбора объектов метаданных,
    которые нужны для решения задачи. Агент сам решает какие tools вызывать
    на основе описания задачи.
    
    Атрибуты:
        mcp: Клиент MCP сервера (должен быть connected)
        llm: Клиент OpenRouter для LLM вызовов
        agent_model: ID модели для агента
        
    Метрики:
        total_tokens: Общее количество использованных токенов
        total_cost: Приблизительная стоимость
        tool_calls_count: Количество вызовов инструментов
    """
    
    def __init__(
        self,
        mcp_client: MCPClient,
        llm_client: OpenRouterClient,
        analysis_model: Optional[str] = None,
    ):
        """
        Инициализация загрузчика контекста
        
        Args:
            mcp_client: Подключенный клиент MCP сервера
            llm_client: Клиент OpenRouter
            analysis_model: Модель для агента (если None — берётся из Settings)
        """
        self._settings = get_settings()
        
        self.mcp = mcp_client
        self.llm = llm_client
        self.agent_model = analysis_model or self._settings.agent.model
        
        # Кэш инструментов
        self._mcp_tools: Optional[List[Dict]] = None
        
        # Метрики
        self.total_tokens = 0
        self.total_cost = 0.0
        self.tool_calls_count = 0
        
        logger.debug(f"AgenticContextLoader создан, модель агента: {self.agent_model}")
    
    # =========================================================================
    # Публичный API
    # =========================================================================
    
    async def load_context(
        self, 
        task_prompt: str, 
        max_iterations: Optional[int] = None
    ) -> ContextLoadResult:
        """
        Запустить агента для сбора контекста
        
        Агент анализирует задачу и сам выбирает какие объекты метаданных
        нужно загрузить для её решения.
        
        Args:
            task_prompt: Текст задания для анализа
            max_iterations: Максимум итераций (защита от зацикливания)
            
        Returns:
            ContextLoadResult с собранным контекстом и метриками
            
        Example:
            result = await loader.load_context("Вывести остатки товаров на складе")
            if result.success:
                print(result.context_text)
        """
        max_iter = max_iterations or self._settings.agent.max_iterations
        max_objects = getattr(self._settings.agent, 'max_objects', 3)
        max_context_chars = getattr(self._settings.agent, 'max_total_context_chars', 15000)
        
        logger.info("Запуск агента сбора контекста...")
        
        # Получаем tools от MCP сервера
        tools = await self._get_tools()
        logger.info(f"Получено {len(tools)} инструментов от MCP сервера")
        
        # Собираем промпты
        prompts = self._get_agent_prompts()
        
        messages = [
            ChatMessage.system(prompts["system"]),
            ChatMessage.user(prompts["user_template"].format(task_prompt=task_prompt))
        ]
        
        loaded_objects: List[Dict[str, str]] = []
        collected_context: List[str] = []
        
        try:
            for iteration in range(max_iter):
                logger.debug(f"Итерация {iteration + 1}/{max_iter}")
                
                # Вызываем LLM с tools
                result = self.llm.chat_completion(
                    model=self.agent_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=1024,
                    tools=tools
                )
                
                self.total_tokens += result.tokens_total
                self.total_cost += result.tokens_total * 0.000001
                
                if not result.success:
                    logger.error(f"Ошибка LLM: {result.error}")
                    break
                
                # Проверяем есть ли tool calls
                if not result.tool_calls:
                    logger.info("Нет вызовов инструментов, завершаю...")
                    break
                
                # Обрабатываем tool calls
                for tool_call in result.tool_calls:
                    tool_name = tool_call.name
                    tool_args_str = tool_call.arguments_raw
                    tool_id = tool_call.id
                    
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # Выполняем tool
                    tool_result = await self._execute_tool(tool_name, tool_args)
                    
                    # finish_research — завершаем
                    if tool_name == FINISH_TOOL_NAME:
                        logger.info(f"Агент завершил исследование за {iteration + 1} итераций")
                        return ContextLoadResult(
                            success=True,
                            context_text="\n\n---\n\n".join(collected_context),
                            objects_loaded=loaded_objects,
                            analysis_tokens=self.total_tokens,
                            analysis_cost=self.total_cost,
                            iterations_count=iteration + 1
                        )
                    
                    # Сохраняем структуры объектов
                    if self._is_valid_metadata_response(tool_name, tool_result):
                        collected_context.append(tool_result)
                        loaded_objects.append({
                            "type": tool_args.get("meta_type", tool_args.get("metaType", "")),
                            "name": tool_args.get("name", "")
                        })
                        
                        # Проверяем лимит объектов
                        if len(loaded_objects) >= max_objects:
                            logger.info(f"Достигнут лимит объектов ({max_objects}), завершаю")
                            return ContextLoadResult(
                                success=True,
                                context_text="\n\n---\n\n".join(collected_context),
                                objects_loaded=loaded_objects,
                                analysis_tokens=self.total_tokens,
                                analysis_cost=self.total_cost,
                                iterations_count=iteration + 1
                            )
                        
                        # Проверяем лимит размера контекста
                        total_chars = sum(len(c) for c in collected_context)
                        if total_chars >= max_context_chars:
                            logger.info(f"Достигнут лимит контекста ({total_chars} символов), завершаю")
                            return ContextLoadResult(
                                success=True,
                                context_text="\n\n---\n\n".join(collected_context),
                                objects_loaded=loaded_objects,
                                analysis_tokens=self.total_tokens,
                                analysis_cost=self.total_cost,
                                iterations_count=iteration + 1
                            )
                    
                    # Добавляем assistant message с tool_call
                    messages.append(ChatMessage.assistant(
                        content="",
                        tool_calls=[tool_call]
                    ))
                    
                    # Добавляем tool response
                    messages.append(ChatMessage.tool_response(
                        content=tool_result,
                        tool_call_id=tool_id
                    ))
            
            logger.warning("Достигнут лимит итераций агента")
            
            return ContextLoadResult(
                success=True,
                context_text="\n\n---\n\n".join(collected_context),
                objects_loaded=loaded_objects,
                analysis_tokens=self.total_tokens,
                analysis_cost=self.total_cost,
                iterations_count=max_iter
            )
            
        except Exception as e:
            logger.exception(f"Ошибка агента: {e}")
            return ContextLoadResult(
                success=False,
                error=str(e),
                analysis_tokens=self.total_tokens,
                analysis_cost=self.total_cost
            )
    
    def reset_metrics(self) -> None:
        """Сбросить метрики для нового эксперимента"""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.tool_calls_count = 0
    
    # =========================================================================
    # Приватные методы
    # =========================================================================
    
    def _get_agent_prompts(self) -> Dict[str, str]:
        """
        Получить промпты агента из Settings
        
        Returns:
            Словарь с ключами 'system' и 'user_template'
        """
        prompts = self._settings.agent.prompts
        
        # Дефолтные значения если не заданы в конфиге
        system = prompts.system or "Ты эксперт по 1С. Изучи метаданные и вызови finish_research."
        user_template = prompts.user_template or "Задача:\n{task_prompt}\n\nИзучи метаданные."
        
        return {
            "system": system,
            "user_template": user_template
        }
    
    async def _get_tools(self) -> List[Dict]:
        """
        Получить tools от MCP сервера и конвертировать в формат OpenAI
        
        Returns:
            Список tools в формате OpenAI function calling
        """
        if self._mcp_tools is not None:
            return self._mcp_tools
        
        mcp_tools_raw = await self.mcp.list_tools()
        
        if not mcp_tools_raw:
            logger.warning("MCP сервер не вернул инструменты")
            self._mcp_tools = [FINISH_TOOL_SCHEMA]
            return self._mcp_tools
        
        # Конвертируем MCP tools в формат OpenAI
        openai_tools = []
        for tool in mcp_tools_raw:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                }
            }
            openai_tools.append(openai_tool)
            logger.debug(f"Загружен инструмент: {tool.get('name')}")
        
        # Добавляем finish_research tool
        openai_tools.append(FINISH_TOOL_SCHEMA)
        
        self._mcp_tools = openai_tools
        return self._mcp_tools
    
    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Выполнить вызов инструмента
        
        Args:
            name: Имя инструмента
            arguments: Аргументы вызова
            
        Returns:
            Текстовый результат выполнения
        """
        self.tool_calls_count += 1
        
        # finish_research обрабатываем локально
        if name == FINISH_TOOL_NAME:
            summary = arguments.get("summary", "")
            logger.debug(f"Исследование завершено: {summary[:100]}...")
            return "DONE"
        
        # Все остальные tools — через MCP
        logger.debug(f"Вызов инструмента: {name}({json.dumps(arguments, ensure_ascii=False)[:80]})")
        
        result = await self.mcp.call_tool(name, arguments)
        
        if result:
            # Сокращаем для экономии токенов
            result = self._compact_structure(result)
            return result
        
        return f"Инструмент {name} вернул пустой результат"
    
    def _compact_structure(self, structure: str, max_lines: Optional[int] = None) -> str:
        """
        Сократить структуру метаданных для экономии токенов
        
        Args:
            structure: Исходная структура
            max_lines: Максимум строк (если None — из Settings)
            
        Returns:
            Сокращённая структура
        """
        max_lines = max_lines or self._settings.agent.max_context_lines
        
        lines = structure.split('\n')
        # Фильтруем пустые строки и строки с пустыми значениями
        filtered = [line for line in lines if line.strip() and not line.strip().endswith('- ""')]
        
        if len(filtered) > max_lines:
            filtered = filtered[:max_lines]
            filtered.append("... (сокращено)")
        
        return '\n'.join(filtered)
    
    @staticmethod
    def _is_valid_metadata_response(tool_name: str, result: str) -> bool:
        """
        Проверить является ли результат валидным ответом с метаданными
        
        Args:
            tool_name: Имя вызванного инструмента
            result: Результат вызова
            
        Returns:
            True если результат содержит полезные метаданные
        """
        if not result:
            return False
        
        # Проверяем что это tool получения структуры и результат не ошибка
        metadata_tools = {"get_metadata_structure", "get_structure", "getMetadataStructure"}
        error_markers = ["не найден", "не найдена", "error", "ошибка"]
        
        if tool_name not in metadata_tools:
            return False
        
        result_lower = result.lower()
        return not any(marker in result_lower for marker in error_markers)


# =============================================================================
# Алиас для обратной совместимости
# =============================================================================

SmartContextLoader = AgenticContextLoader
