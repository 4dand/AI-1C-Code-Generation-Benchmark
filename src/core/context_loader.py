"""
Agentic Context Loader — модель сама выбирает нужные объекты через MCP tools

Логика:
1. Получаем список tools от MCP сервера (tools/list)
2. Конвертируем в формат OpenAI tools
3. Даём модели задачу + tools
4. Модель вызывает tools → проксируем через tools/call
5. Собираем контекст из ответов
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from ..clients.openrouter import OpenRouterClient
from ..clients.mcp import MCPClient
from ..schemas.results import ContextLoadResult, ChatMessage
from ..utils.file_ops import load_yaml


# Finish tool — добавляется к MCP tools
FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_research",
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


class AgenticContextLoader:
    """
    Agentic загрузчик контекста — модель сама выбирает что загружать через MCP tools
    
    Получает tools динамически от MCP сервера (vladimir-kharin/1c_mcp)
    Промпты загружаются из конфига категории
    """
    
    def __init__(
        self,
        mcp_client: MCPClient,
        llm_client: OpenRouterClient,
        analysis_model: str = "google/gemini-2.0-flash-001",
        config_dir: str = "config"
    ):
        """
        Аргументы:
            mcp_client: Клиент MCP сервера (должен быть connected)
            llm_client: Клиент OpenRouter
            analysis_model: Модель для агента (должна поддерживать tools)
            config_dir: Путь к папке с конфигами
        """
        self.mcp = mcp_client
        self.llm = llm_client
        self.agent_model = analysis_model
        self.config_dir = Path(config_dir)
        self._structure_cache: Dict[Tuple[str, str], str] = {}
        self._mcp_tools: Optional[List[Dict]] = None
        self._agent_prompts: Optional[Dict[str, str]] = None
        
        # Метрики
        self.total_tokens = 0
        self.total_cost = 0.0
        self.tool_calls_count = 0
    
    def _load_agent_prompts(self, category: str = "B") -> Dict[str, str]:
        """Загрузить промпты агента из конфига категории"""
        if self._agent_prompts is not None:
            return self._agent_prompts
        
        config_path = self.config_dir / f"tasks_category_{category}.yaml"
        config = load_yaml(config_path)
        
        agent_prompts = config.get("agent_prompts", {})
        
        # Дефолтные значения если не заданы в конфиге
        self._agent_prompts = {
            "system": agent_prompts.get("system", "Ты эксперт по 1С. Изучи метаданные и вызови finish_research."),
            "user_template": agent_prompts.get("user_template", "Задача:\n{task_prompt}\n\nИзучи метаданные.")
        }
        
        return self._agent_prompts
    
    async def _get_tools(self) -> List[Dict]:
        """
        Получить tools от MCP сервера и конвертировать в формат OpenAI
        """
        if self._mcp_tools is not None:
            return self._mcp_tools
        
        mcp_tools_raw = await self.mcp.list_tools()
        
        if not mcp_tools_raw:
            print("[Агент] Предупреждение: MCP сервер не вернул инструменты")
            self._mcp_tools = [FINISH_TOOL]
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
            print(f"   Инструмент: {tool.get('name')}")
        
        # Добавляем finish_research tool
        openai_tools.append(FINISH_TOOL)
        
        self._mcp_tools = openai_tools
        return self._mcp_tools
    
    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Выполнить tool call через MCP сервер"""
        self.tool_calls_count += 1
        
        # finish_research обрабатываем локально
        if name == "finish_research":
            summary = arguments.get("summary", "")
            print(f"   Исследование завершено: {summary[:100]}...")
            return "DONE"
        
        # Все остальные tools — через MCP
        print(f"  🔧 {name}({json.dumps(arguments, ensure_ascii=False)[:80]}...)")
        
        result = await self.mcp.call_tool(name, arguments)
        
        if result:
            # Сокращаем для экономии токенов
            result = self._compact_structure(result)
            return result
        
        return f"Инструмент {name} вернул пустой результат"
    
    def _compact_structure(self, structure: str, max_lines: int = 80) -> str:
        """Сократить структуру для экономии токенов"""
        lines = structure.split('\n')
        filtered = [l for l in lines if l.strip() and not l.strip().endswith('- ""')]
        
        if len(filtered) > max_lines:
            filtered = filtered[:max_lines]
            filtered.append("... (сокращено)")
        
        return '\n'.join(filtered)
    
    async def load_context(self, task_prompt: str, max_iterations: int = 10) -> ContextLoadResult:
        """
        Запустить агента для сбора контекста
        
        Аргументы:
            task_prompt: Текст задания
            max_iterations: Максимум итераций (защита от зацикливания)
            
        Возвращает:
            ContextLoadResult с собранным контекстом
        """
        print("[Агент] Начинаю исследование метаданных...")
        
        # Получаем tools от MCP сервера
        tools = await self._get_tools()
        print(f"[Агент] Получено {len(tools)} инструментов от MCP сервера")
        
        # Загружаем промпты из конфига
        prompts = self._load_agent_prompts()
        
        messages = [
            ChatMessage(role="system", content=prompts["system"]),
            ChatMessage(role="user", content=prompts["user_template"].format(task_prompt=task_prompt))
        ]
        
        loaded_objects: List[Dict[str, str]] = []
        collected_context: List[str] = []
        
        try:
            for iteration in range(max_iterations):
                print(f"[Агент] Итерация {iteration + 1}/{max_iterations}")
                
                # Вызываем LLM с tools от MCP сервера
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
                    print(f"[Агент] Ошибка LLM: {result.error}")
                    break
                
                # Проверяем есть ли tool calls
                if not result.tool_calls:
                    print("[Агент] Нет вызовов инструментов, завершаю...")
                    break
                
                # Обрабатываем tool calls
                for tool_call in result.tool_calls:
                    tool_name = tool_call.get("function", {}).get("name")
                    tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                    tool_id = tool_call.get("id", "")
                    
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # Выполняем tool
                    tool_result = await self._execute_tool(tool_name, tool_args)
                    
                    # finish_research — завершаем
                    if tool_name == "finish_research":
                        print(f"[Агент] Исследование завершено за {iteration + 1} итераций")
                        return ContextLoadResult(
                            success=True,
                            context_text="\n\n---\n\n".join(collected_context),
                            objects_loaded=loaded_objects,
                            analysis_tokens=self.total_tokens,
                            analysis_cost=self.total_cost
                        )
                    
                    # Сохраняем структуры объектов
                    if tool_name == "get_metadata_structure" and tool_result and "не найдена" not in tool_result:
                        collected_context.append(tool_result)
                        loaded_objects.append({
                            "type": tool_args.get("meta_type"),
                            "name": tool_args.get("name")
                        })
                    
                    # Добавляем assistant message с tool_call
                    messages.append(ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[tool_call]
                    ))
                    
                    # Добавляем tool response
                    messages.append(ChatMessage(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tool_id
                    ))
            
            print("[Агент] Достигнут лимит итераций")
            
            return ContextLoadResult(
                success=True,
                context_text="\n\n---\n\n".join(collected_context),
                objects_loaded=loaded_objects,
                analysis_tokens=self.total_tokens,
                analysis_cost=self.total_cost
            )
            
        except Exception as e:
            print(f"[Агент] Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return ContextLoadResult(
                success=False,
                error=str(e),
                analysis_tokens=self.total_tokens,
                analysis_cost=self.total_cost
            )


# Алиас для обратной совместимости
SmartContextLoader = AgenticContextLoader
