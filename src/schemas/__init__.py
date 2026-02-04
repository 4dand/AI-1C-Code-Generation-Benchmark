"""
Data Schemas - структуры данных для бенчмарка
"""

from .config import (
    ModelConfig,
    TaskConfig,
    GenerationParams,
    ExperimentConfig,
    MCPConfig,
)

from .results import (
    ChatMessage,
    GenerationResult,
    RunResult,
    TaskResult,
    ExperimentResult,
    ContextLoadResult,
)

__all__ = [
    # Config schemas
    "ModelConfig",
    "TaskConfig", 
    "GenerationParams",
    "ExperimentConfig",
    "MCPConfig",
    # Result schemas
    "ChatMessage",
    "GenerationResult",
    "RunResult",
    "TaskResult",
    "ExperimentResult",
    "ContextLoadResult",
]
