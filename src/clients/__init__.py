"""
API Clients - HTTP клиенты для внешних сервисов

Клиенты:
- OpenRouterClient: Работа с OpenRouter API (LLM генерация)
- MCPClient: Работа с MCP сервером 1С (метаданные)

Принципы:
- Каждый клиент отвечает только за HTTP взаимодействие
- Клиенты stateless (кроме session в MCP)
- Используют Settings через фабричный метод from_settings()
"""

from .openrouter import OpenRouterClient
from .mcp import MCPClient

__all__ = [
    "OpenRouterClient",
    "MCPClient",
]

