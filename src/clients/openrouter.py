"""
OpenRouter Client - HTTP клиент для OpenRouter API

Ответственность (SRP):
- Отправка запросов к OpenRouter API
- Форматирование сообщений
- Парсинг ответов

НЕ отвечает за:
- Расчёт стоимости (бизнес-логика)
- Конфигурацию моделей (schemas)
- Логику retry (можно добавить декоратором)

Использование:
    from src.config import get_settings
    from src.clients import OpenRouterClient
    
    settings = get_settings()
    client = OpenRouterClient.from_settings(settings)
    
    result = client.chat_completion(
        model="anthropic/claude-sonnet-4",
        messages=[ChatMessage.user("Привет!")]
    )
"""

import time
import logging
import requests
from typing import Optional, List, Dict, Any

from ..schemas.messages import ChatMessage, GenerationResult, ToolCall

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    HTTP клиент для OpenRouter API
    
    Поддерживает:
    - Chat completions (с streaming в будущем)
    - Tool calling для agentic режима
    - Детерминизм через seed/temperature
    
    Принципы:
    - Stateless (кроме конфигурации)
    - Не знает о бизнес-логике бенчмарка
    - Возвращает типизированные результаты
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str = "https://1c-benchmark.local",
        app_title: str = "AI-1C-Code-Generation-Benchmark",
        timeout: int = 120
    ):
        """
        Args:
            api_key: API ключ OpenRouter
            base_url: Базовый URL API
            http_referer: HTTP Referer для OpenRouter
            app_title: Название приложения для OpenRouter
            timeout: Таймаут запроса в секундах
        """
        if not api_key:
            raise ValueError("API ключ не может быть пустым")
        
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": http_referer,
            "X-Title": app_title,
        }
        
        logger.debug(f"OpenRouterClient инициализирован, base_url={self.base_url}")
    
    @classmethod
    def from_settings(cls, settings) -> "OpenRouterClient":
        """
        Создать клиент из Settings
        
        Args:
            settings: Объект Settings из src.config
            
        Returns:
            Настроенный OpenRouterClient
        """
        settings.validate_api_key()
        
        return cls(
            api_key=settings.openrouter.api_key,
            base_url=settings.openrouter.base_url,
            http_referer=settings.openrouter.http_referer,
            app_title=settings.openrouter.app_title,
            timeout=settings.openrouter.timeout,
        )
    
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
            model: ID модели (например "anthropic/claude-sonnet-4")
            messages: Список сообщений чата
            temperature: Температура генерации (0.0 = детерминизм)
            max_tokens: Максимум токенов в ответе
            seed: Seed для детерминизма (поддерживают GPT, Gemini)
            tools: Инструменты для tool calling
            tool_choice: Режим выбора инструментов ("auto", "none", "required")
            
        Returns:
            GenerationResult с ответом или ошибкой
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
        
        logger.debug(f"Запрос к {model}, сообщений={len(messages)}, температура={temperature}")
        
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
                return self._parse_success_response(response.json(), model, elapsed)
            else:
                return self._parse_error_response(response, elapsed)
                
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            logger.warning(f"Таймаут после {self.timeout}s для {model}")
            result = GenerationResult.failure(f"Таймаут после {self.timeout}s")
            result.elapsed_time = elapsed
            return result
        
        except requests.exceptions.ConnectionError as e:
            elapsed = time.time() - start_time
            logger.error(f"Ошибка соединения: {e}")
            result = GenerationResult.failure(f"Ошибка соединения: {e}")
            result.elapsed_time = elapsed
            return result
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"Неожиданная ошибка: {e}")
            result = GenerationResult.failure(str(e))
            result.elapsed_time = elapsed
            return result
    
    def get_balance(self) -> Optional[Dict[str, float]]:
        """
        Получить информацию о балансе аккаунта
        
        Returns:
            {"limit": float, "usage": float, "available": float} или None
        """
        try:
            response = requests.get(
                f"{self.base_url}/auth/key",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json().get("data", {})
                limit = data.get("limit", 0)
                usage = data.get("usage", 0)
                
                return {
                    "limit": float(limit),
                    "usage": float(usage),
                    "available": float(limit - usage)
                }
            
            logger.warning(f"Не удалось получить баланс: HTTP {response.status_code}")
            
        except Exception as e:
            logger.error(f"Ошибка при получении баланса: {e}")
        
        return None
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Получить список доступных моделей
        
        Returns:
            Список моделей с их характеристиками
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get("data", [])
            
        except Exception as e:
            logger.error(f"Ошибка при получении моделей: {e}")
        
        return []
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Форматировать сообщения для API
        
        Поддерживает:
        - Обычные сообщения (system, user, assistant)
        - Tool calls от assistant
        - Tool responses
        """
        formatted = []
        
        for msg in messages:
            if hasattr(msg, 'to_api_dict'):
                formatted.append(msg.to_api_dict())
            else:
                msg_dict = {"role": msg.role, "content": msg.content}
                
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        tc.model_dump() if hasattr(tc, 'model_dump') else tc
                        for tc in msg.tool_calls
                    ]
                
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                    msg_dict["tool_call_id"] = msg.tool_call_id
                
                formatted.append(msg_dict)
        
        return formatted
    
    def _parse_success_response(
        self, 
        data: Dict[str, Any], 
        model: str,
        elapsed: float
    ) -> GenerationResult:
        """Парсить успешный ответ от API"""
        choices = data.get("choices", [])
        content = ""
        tool_calls = None
        
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "") or ""
            raw_tool_calls = message.get("tool_calls")
            
            if raw_tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=tc.get("function", {})
                    )
                    for tc in raw_tool_calls
                ]
        
        usage = data.get("usage", {})
        
        logger.debug(
            f"Ответ: {len(content)} символов, "
            f"токены={usage.get('total_tokens', 0)}, "
            f"время={elapsed:.2f}s"
        )
        
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
    
    def _parse_error_response(
        self, 
        response: requests.Response, 
        elapsed: float
    ) -> GenerationResult:
        """Парсить ошибочный ответ от API"""
        error_msg = f"HTTP {response.status_code}"
        
        try:
            error_data = response.json()
            if "error" in error_data:
                error_msg = f"{error_msg}: {error_data['error'].get('message', '')}"
        except Exception:
            error_msg = f"{error_msg}: {response.text[:200]}"
        
        logger.warning(f"Ошибка API: {error_msg}")
        
        result = GenerationResult.failure(error_msg)
        result.elapsed_time = elapsed
        return result
