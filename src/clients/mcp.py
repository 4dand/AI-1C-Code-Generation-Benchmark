"""
MCP Client - HTTP клиент для MCP сервера 1С

Клиент для работы с MCP сервером vladimir-kharin/1c_mcp.
Использует Streamable HTTP транспорт для Docker-развёртывания.

Ответственность (SRP):
- Установка/разрыв соединения с MCP сервером
- Вызов MCP tools (list_metadata_objects, get_metadata_structure, etc.)
- Парсинг SSE ответов

НЕ отвечает за:
- Бизнес-логику сбора контекста (это ContextLoader)
- Кэширование результатов (это отдельный слой)

Использование:
    from src.config import get_settings
    from src.clients import MCPClient
    
    settings = get_settings()
    client = MCPClient.from_settings(settings)
    
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("get_metadata_structure", {...})
    await client.disconnect()
"""

import json
import logging
from typing import Optional, List, Dict, Any

import requests

logger = logging.getLogger(__name__)


class MCPClient:
    """
    HTTP клиент для MCP сервера 1С:Предприятие
    
    Сервер: vladimir-kharin/1c_mcp (Docker)
    Протокол: MCP Streamable HTTP (/mcp/)
    
    Принципы:
    - Async-first API (для совместимости с agentic режимом)
    - Stateful (хранит session_id)
    - Возвращает сырые данные (парсинг — ответственность вызывающего)
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8000",
        timeout: int = 30
    ):
        """
        Args:
            url: URL MCP сервера
            timeout: Таймаут запросов в секундах
        """
        self.url = url.rstrip("/")
        self.timeout = timeout
        
        self._request_id = 0
        self._session_id: Optional[str] = None
        self._initialized = False
        
        logger.debug(f"MCPClient создан, url={self.url}")
    
    @classmethod
    def from_settings(cls, settings) -> "MCPClient":
        """
        Создать клиент из Settings
        
        Args:
            settings: Объект Settings из src.config
            
        Returns:
            Настроенный MCPClient
        """
        return cls(
            url=settings.mcp.url,
            timeout=settings.mcp.timeout,
        )
    
    @property
    def mcp_endpoint(self) -> str:
        """Полный URL для MCP запросов"""
        return f"{self.url}/mcp/"
    
    @property
    def is_connected(self) -> bool:
        """Установлено ли соединение"""
        return self._initialized and self._session_id is not None
    
    # =========================================================================
    # Connection lifecycle
    # =========================================================================
    
    async def connect(self) -> bool:
        """
        Инициализировать соединение с MCP сервером
        
        Отправляет initialize запрос и получает session_id.
        
        Returns:
            True если соединение успешно установлено
        """
        logger.info(f"Соединение с MCP сервером по адресу {self.url}...")
        
        try:
            result = self._send_request(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "AI-1C-Benchmark",
                        "version": "1.0.0"
                    }
                },
                expect_session=True
            )
            
            if result is not None:
                self._initialized = True
                logger.info(f"Подключено к MCP (session: {self._session_id[:8]}...)")
                return True
            
            logger.error("Не удалось инициализировать соединение с MCP")
            return False
            
        except Exception as e:
            logger.exception(f"Ошибка соединения: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Закрыть соединение с MCP сервером"""
        if self._initialized:
            logger.info("Отключение от MCP сервера...")
        
        self._initialized = False
        self._session_id = None
        self._request_id = 0
    
    # =========================================================================
    # MCP Tools API
    # =========================================================================
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Получить список доступных инструментов
        
        Returns:
            Список tools в формате MCP (name, description, inputSchema)
        """
        result = await self._call_method("tools/list", {})
        
        if result and "tools" in result:
            tools = result["tools"]
            logger.debug(f"Доступные инструменты: {[t.get('name') for t in tools]}")
            return tools
        
        return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """
        Вызвать инструмент по имени
        
        Args:
            name: Имя инструмента (например "get_metadata_structure")
            arguments: Аргументы для инструмента
            
        Returns:
            Текстовый результат или None при ошибке
        """
        logger.debug(f"Вызов инструмента: {name}({json.dumps(arguments, ensure_ascii=False)[:80]})")
        
        result = await self._call_method("tools/call", {
            "name": name,
            "arguments": arguments
        })
        
        return self._extract_text_content(result)
    
    async def list_metadata_objects(
        self, 
        meta_type: str, 
        name_mask: str = "*",
        max_items: int = 100
    ) -> Optional[str]:
        """
        Получить список объектов метаданных
        
        Args:
            meta_type: Тип метаданных (Catalogs, Documents, AccumulationRegisters, etc.)
            name_mask: Маска имени для фильтрации
            max_items: Максимум объектов
            
        Returns:
            Текст со списком объектов
        """
        arguments = {
            "metaType": meta_type,
            "maxItems": max_items
        }
        
        if name_mask and name_mask != "*":
            arguments["nameMask"] = name_mask
        
        return await self.call_tool("list_metadata_objects", arguments)
    
    async def get_metadata_structure(
        self,
        meta_type: str,
        name: str
    ) -> Optional[str]:
        """
        Получить структуру объекта метаданных
        
        Args:
            meta_type: Тип (Catalogs, Documents, AccumulationRegisters, etc.)
            name: Имя объекта
            
        Returns:
            Текст со структурой объекта (реквизиты, табличные части)
        """
        return await self.call_tool("get_metadata_structure", {
            "metaType": meta_type,
            "name": name
        })
    
    # =========================================================================
    # Private methods
    # =========================================================================
    
    async def _call_method(self, method: str, params: Dict) -> Optional[Dict]:
        """
        Вызвать MCP метод (требует инициализации)
        
        Args:
            method: Имя метода MCP
            params: Параметры запроса
            
        Returns:
            Результат или None при ошибке
        """
        if not self._initialized:
            logger.error("MCP не инициализирован. Сначала вызовите connect().")
            return None
        
        return self._send_request(method, params)
    
    def _send_request(
        self, 
        method: str, 
        params: Dict,
        expect_session: bool = False
    ) -> Optional[Dict]:
        """
        Отправить JSON-RPC запрос через Streamable HTTP
        
        Args:
            method: Метод MCP
            params: Параметры запроса
            expect_session: Ожидать session ID в ответе
            
        Returns:
            Результат или None при ошибке
        """
        self._request_id += 1
        
        request_body = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        
        try:
            response = requests.post(
                self.mcp_endpoint,
                json=request_body,
                headers=headers,
                timeout=self.timeout
            )
            
            # Сохраняем session ID из заголовков
            if expect_session and "mcp-session-id" in response.headers:
                self._session_id = response.headers["mcp-session-id"]
            
            if response.status_code == 200:
                return self._parse_sse_response(response.text)
            
            logger.warning(f"HTTP {response.status_code}: {response.text[:100]}")
            
        except requests.exceptions.Timeout:
            logger.warning(f"Таймаут для {method} после {self.timeout}s")
        
        except requests.exceptions.ConnectionError:
            logger.error(f"Отказано в соединении - запущен ли MCP сервер по адресу {self.url}?")
        
        except Exception as e:
            logger.exception(f"Ошибка запроса: {e}")
        
        return None
    
    def _parse_sse_response(self, text: str) -> Optional[Dict]:
        """
        Парсить Server-Sent Events ответ
        
        Формат:
            event: message
            data: {"jsonrpc": "2.0", "result": {...}}
        
        Args:
            text: Сырой текст ответа
            
        Returns:
            Содержимое result или None
        """
        for line in text.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    
                    if "result" in data:
                        return data["result"]
                    
                    if "error" in data:
                        error = data["error"]
                        logger.error(f"MCP ошибка: {error}")
                        return None
                        
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _extract_text_content(self, result: Optional[Dict]) -> Optional[str]:
        """
        Извлечь текстовый контент из ответа MCP
        
        MCP tools возвращают content в формате:
            {"content": [{"type": "text", "text": "..."}]}
        
        Args:
            result: Результат от _call_method
            
        Returns:
            Текстовое содержимое или None
        """
        if result and "content" in result:
            contents = result["content"]
            if isinstance(contents, list) and len(contents) > 0:
                return contents[0].get("text", "")
        
        return None
