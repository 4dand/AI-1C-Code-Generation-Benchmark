"""
MCP Client - клиент для работы с прокси MCP сервером 1С (vladimir-kharin/1c_mcp)
Поддерживает Streamable HTTP транспорт для Docker-развёртывания.

Tools:
- list_metadata_objects: Список объектов метаданных
- get_metadata_structure: Структура объекта (реквизиты, ТЧ)
"""

import json
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..schemas.config import MCPConfig


@dataclass
class MCPToolResult:
    """Результат вызова MCP tool"""
    success: bool
    content: str = ""
    error: Optional[str] = None


class MCPClient:
    """
    HTTP клиент для MCP сервера 1С:Предприятие
    
    Сервер: vladimir-kharin/1c_mcp (Docker)
    Протокол: MCP Streamable HTTP (/mcp/)
    """
    
    def __init__(self, config: MCPConfig = None):
        """
        Args:
            config: Конфигурация MCP сервера
        """
        self.config = config or MCPConfig()
        self._request_id = 0
        self._session_id: Optional[str] = None
        self._initialized = False
    
    @property
    def base_url(self) -> str:
        return self.config.url.rstrip('/')
    
    @property
    def mcp_endpoint(self) -> str:
        return f"{self.base_url}/mcp/"
    
    async def connect(self) -> bool:
        """
        Инициализировать соединение с MCP сервером
        
        Returns:
            True если соединение успешно
        """
        try:
            # Отправляем initialize запрос
            result = self._send_raw_request(
                "initialize",
                {
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
                print(f"[MCP] Connected (session: {self._session_id[:8]}...)")
                return True
                
        except Exception as e:
            print(f"[MCP] Connection error: {e}")
        
        return False
    
    async def disconnect(self):
        """Закрыть соединение"""
        self._initialized = False
        self._session_id = None
        print("[MCP] Disconnected")
    
    def _send_raw_request(
        self, 
        method: str, 
        params: Dict = None,
        expect_session: bool = False
    ) -> Optional[Dict]:
        """
        Отправить запрос через Streamable HTTP
        
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
            "params": params or {}
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
                timeout=self.config.timeout
            )
            if expect_session and "mcp-session-id" in response.headers:
                self._session_id = response.headers["mcp-session-id"]
            
            if response.status_code == 200:
                return self._parse_sse_response(response.text)
            else:
                print(f"[MCP] HTTP {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print(f"[MCP] Timeout for {method}")
        except requests.exceptions.ConnectionError:
            print(f"[MCP] Connection refused - is server running at {self.base_url}?")
        except Exception as e:
            print(f"[MCP] Request error: {e}")
        
        return None
    
    def _parse_sse_response(self, text: str) -> Optional[Dict]:
        """
        Парсить SSE ответ от сервера
        
        Формат:
            event: message
            data: {"jsonrpc": "2.0", ...}
        """
        for line in text.split('\n'):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    if "result" in data:
                        return data["result"]
                    elif "error" in data:
                        print(f"[MCP] Error: {data['error']}")
                        return None
                except json.JSONDecodeError:
                    continue
        return None
    
    async def _send_request(
        self, 
        method: str, 
        params: Dict = None
    ) -> Optional[Dict]:
        """
        Отправить MCP запрос (с сессией)
        """
        if not self._initialized:
            print("[MCP] Not initialized, call connect() first")
            return None
            
        return self._send_raw_request(method, params)
    
    # ========== MCP Tools ==========
    
    async def list_metadata_objects(
        self, 
        meta_type: str, 
        name_mask: str = "*",
        max_items: int = 100
    ) -> Optional[str]:
        """
        Получить список объектов метаданных
        
        Args:
            meta_type: Тип (Catalogs, Documents, AccumulationRegisters, etc.)
            name_mask: Маска имени для фильтрации
            max_items: Максимум объектов
            
        Returns:
            Текст со списком объектов
        """
        params = {
            "name": "list_metadata_objects",
            "arguments": {
                "metaType": meta_type,
                "maxItems": max_items
            }
        }
        
        # Добавляем маску только если не "*"
        if name_mask and name_mask != "*":
            params["arguments"]["nameMask"] = name_mask
        
        result = await self._send_request("tools/call", params)
        return self._extract_content(result)
    
    async def get_metadata_structure(
        self,
        meta_type: str,
        name: str
    ) -> Optional[str]:
        """
        Получить структуру объекта метаданных
        
        Args:
            meta_type: Тип (Catalogs, Documents, etc.)
            name: Имя объекта
            
        Returns:
            Текст со структурой объекта
        """
        result = await self._send_request("tools/call", {
            "name": "get_metadata_structure",
            "arguments": {
                "metaType": meta_type,
                "name": name
            }
        })
        
        return self._extract_content(result)
    
    async def list_tools(self) -> Optional[List[Dict]]:
        """Получить список доступных tools"""
        result = await self._send_request("tools/list", {})
        if result and "tools" in result:
            return result["tools"]
        return None
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """
        Вызвать tool по имени (универсальный метод)
        
        Args:
            name: Имя tool
            arguments: Аргументы для tool
            
        Returns:
            Текстовый результат или None
        """
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        return self._extract_content(result)
    
    def _extract_content(self, result: Optional[Dict]) -> Optional[str]:
        """Извлечь текстовый контент из ответа MCP"""
        if result and "content" in result:
            contents = result["content"]
            if isinstance(contents, list) and len(contents) > 0:
                return contents[0].get("text", "")
        return None
