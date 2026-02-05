# AI-1C-Code-Generation-Benchmark
# Ядро для бенчмаркинга ИИ-моделей на задачах 1С:Предприятие

from .schemas.config import (
    ModelConfig,
    TaskConfig,
    GenerationParams,
    ExperimentConfig,
    MCPConfig
)
from .schemas.results import (
    ChatMessage,
    GenerationResult,
    RunResult,
    TaskResult,
    ExperimentResult,
    DeterminismResult,
    ContextLoadResult
)
from .clients.openrouter import OpenRouterClient
from .clients.mcp import MCPClient
from .core.benchmark import BenchmarkRunner
from .core.context_loader import SmartContextLoader

__all__ = [
    # Schemas - config
    "ModelConfig",
    "TaskConfig",
    "GenerationParams",
    "ExperimentConfig",
    "MCPConfig",
    # Schemas - results
    "ChatMessage",
    "GenerationResult",
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
