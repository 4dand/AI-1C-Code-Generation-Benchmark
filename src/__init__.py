# AI-1C-Code-Generation-Benchmark
# Ядро для бенчмаркинга ИИ-моделей на задачах 1С:Предприятие

from .schemas import (
    # Models
    ModelConfig,
    ModelMeta,
    ModelGenerationParams,
    ModelsRegistry,
    # Tasks
    TaskConfig,
    CategoryConfig,
    TasksFile,
    # Messages
    ChatMessage,
    GenerationResult,
    ToolCall,
    # Results
    RunResult,
    TaskResult,
    ExperimentResult,
    DeterminismResult,
    ContextLoadResult,
)

from .clients.openrouter import OpenRouterClient
from .clients.mcp import MCPClient
from .core.benchmark import BenchmarkRunner
from .core.context_loader import SmartContextLoader

__all__ = [
    # Schemas - Models
    "ModelConfig",
    "ModelMeta",
    "ModelGenerationParams",
    "ModelsRegistry",
    # Schemas - Tasks
    "TaskConfig",
    "CategoryConfig",
    "TasksFile",
    # Schemas - Messages
    "ChatMessage",
    "GenerationResult",
    "ToolCall",
    # Schemas - Results
    "RunResult",
    "TaskResult",
    "ExperimentResult",
    "DeterminismResult",
    "ContextLoadResult",
    # Clients
    "OpenRouterClient",
    "MCPClient",
    # Core
    "BenchmarkRunner",
    "SmartContextLoader",
]
