"""
Message Schemas — схемы для работы с LLM API

Отвечает за:
- Структуру сообщений чата (ChatMessage)
- Результаты вызовов API (GenerationResult)
- Tool calls для agentic режима
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Вызов инструмента от LLM"""
    id: str = Field(..., description="Уникальный ID вызова")
    type: Literal["function"] = "function"
    function: Dict[str, Any] = Field(
        ...,
        description="Информация о функции: {name: str, arguments: str}"
    )
    
    @property
    def name(self) -> str:
        """Имя вызываемой функции"""
        return self.function.get("name", "")
    
    @property
    def arguments_raw(self) -> str:
        """Сырые аргументы (JSON строка)"""
        return self.function.get("arguments", "{}")


class ChatMessage(BaseModel):
    """
    Сообщение чата для LLM API
    
    Поддерживает роли:
    - system: системный промпт
    - user: сообщение пользователя
    - assistant: ответ модели (может содержать tool_calls)
    - tool: результат выполнения инструмента
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    
    # Для assistant с вызовами инструментов
    tool_calls: Optional[List[ToolCall]] = None
    
    # Для tool response
    tool_call_id: Optional[str] = None
    
    def to_api_dict(self) -> Dict[str, Any]:
        """Преобразовать в формат для API"""
        result: Dict[str, Any] = {
            "role": self.role,
            "content": self.content
        }
        
        if self.tool_calls:
            result["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        
        return result
    
    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Создать системное сообщение"""
        return cls(role="system", content=content)
    
    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Создать сообщение пользователя"""
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[ToolCall]] = None) -> "ChatMessage":
        """Создать сообщение ассистента"""
        return cls(role="assistant", content=content, tool_calls=tool_calls)
    
    @classmethod
    def tool_response(cls, content: str, tool_call_id: str) -> "ChatMessage":
        """Создать ответ инструмента"""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


class GenerationResult(BaseModel):
    """
    Результат генерации от LLM API
    
    Содержит ответ модели, метрики токенов и информацию об ошибках
    """
    success: bool = True
    content: str = ""
    
    # Метрики токенов
    tokens_input: int = Field(default=0, ge=0)
    tokens_output: int = Field(default=0, ge=0)
    tokens_total: int = Field(default=0, ge=0)
    
    # Время выполнения
    elapsed_time: float = Field(default=0.0, ge=0.0)
    
    # Информация о модели
    model_used: str = ""
    
    # Tool calls (для agentic режима)
    tool_calls: Optional[List[ToolCall]] = None
    
    # Ошибка
    error: Optional[str] = None
    
    # Сырой ответ (для отладки)
    raw_response: Optional[Dict[str, Any]] = None
    
    @property
    def has_tool_calls(self) -> bool:
        """Есть ли вызовы инструментов"""
        return bool(self.tool_calls)
    
    @classmethod
    def failure(cls, error: str) -> "GenerationResult":
        """Создать результат с ошибкой"""
        return cls(success=False, error=error)
