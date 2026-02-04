"""
OpenRouter Client - клиент для работы с OpenRouter API
"""

import time
import requests
from typing import Optional, List, Dict, Any

from ..schemas.config import ModelConfig
from ..schemas.results import ChatMessage, GenerationResult


class OpenRouterClient:
    """
    Клиент для работы с OpenRouter API
    
    Поддерживает:
    - Chat completions
    - Tool calling
    - Детерминизм через seed/temperature
    """
    
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        default_headers: Dict[str, str] = None,
        timeout: int = 60
    ):
        """
        Args:
            api_key: API ключ OpenRouter
            base_url: Базовый URL API (по умолчанию OpenRouter)
            default_headers: Дополнительные заголовки
            timeout: Таймаут запроса в секундах
        """
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if default_headers:
            self.headers.update(default_headers)
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict]:
        """
        Форматировать сообщения для API с поддержкой tool calls
        """
        formatted = []
        for msg in messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            
            # Для assistant с tool_calls
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            
            # Для tool response
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            
            formatted.append(msg_dict)
        return formatted
    
    def chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto"
    ) -> GenerationResult:
        """
        Отправить запрос на генерацию
        
        Args:
            model: ID модели
            messages: Список сообщений
            temperature: Температура генерации
            max_tokens: Максимум токенов
            seed: Seed для детерминизма (для GPT/Gemini)
            tools: Инструменты для tool calling
            tool_choice: Режим выбора инструментов
            
        Returns:
            GenerationResult с ответом
        """
        request_body = {
            "model": model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            request_body["seed"] = seed
        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = tool_choice
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=request_body,
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                content = ""
                tool_calls = None
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "") or ""
                    tool_calls = message.get("tool_calls")
                usage = data.get("usage", {})
                
                return GenerationResult(
                    success=True,
                    content=content,
                    tokens_input=usage.get("prompt_tokens", 0),
                    tokens_output=usage.get("completion_tokens", 0),
                    tokens_total=usage.get("total_tokens", 0),
                    elapsed_time=elapsed,
                    model_used=data.get("model", model),
                    tool_calls=tool_calls,
                    raw_response=data
                )
            else:
                return GenerationResult(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    elapsed_time=elapsed
                )
                
        except requests.exceptions.Timeout:
            return GenerationResult(
                success=False,
                error=f"Timeout after {self.timeout}s",
                elapsed_time=time.time() - start_time
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                error=str(e),
                elapsed_time=time.time() - start_time
            )
    
    def get_balance(self) -> Optional[Dict]:
        """Получить информацию о балансе"""
        try:
            response = requests.get(
                f"{self.base_url}/auth/key",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                return {
                    "limit": data.get("limit", 0),
                    "usage": data.get("usage", 0),
                    "available": data.get("limit", 0) - data.get("usage", 0)
                }
        except Exception:
            pass
        return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Получить информацию о модели"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                for model in models:
                    if model.get("id") == model_id:
                        return model
        except Exception:
            pass
        return None
    
    def calculate_cost(
        self,
        model_config: ModelConfig,
        tokens_input: int,
        tokens_output: int
    ) -> Dict[str, float]:
        """
        Рассчитать стоимость запроса
        
        Args:
            model_config: Конфигурация модели с ценами
            tokens_input: Входные токены
            tokens_output: Выходные токены
            
        Returns:
            {"input": cost, "output": cost, "total": cost}
        """
        # цены в конфиге за 1M токенов
        cost_input = (tokens_input / 1_000_000) * model_config.price_input
        cost_output = (tokens_output / 1_000_000) * model_config.price_output
        
        return {
            "input": cost_input,
            "output": cost_output,
            "total": cost_input + cost_output
        }
