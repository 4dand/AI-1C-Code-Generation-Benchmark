"""
Data Schemas - схемы данных для бенчмарка

Экспортирует все схемы из подмодулей:
- models: ModelConfig, ModelsRegistry
- tasks: TaskConfig, CategoryConfig, TasksFile
- messages: ChatMessage, GenerationResult, ToolCall
- results: RunResult, TaskResult, ExperimentResult, DeterminismResult, ContextLoadResult
"""

from .models import (
    ModelConfig,
    ModelMeta,
    ModelGenerationParams,
    ModelsRegistry,
)

from .tasks import (
    TaskConfig,
    CategoryConfig,
    CategoryGenerationOverrides,
    TasksFile,
)

from .messages import (
    ChatMessage,
    GenerationResult,
    ToolCall,
)

from .results import (
    RunResult,
    TaskResult,
    ExperimentResult,
    DeterminismResult,
    ContextLoadResult,
)

__all__ = [
    # Models
    "ModelConfig",
    "ModelMeta",
    "ModelGenerationParams",
    "ModelsRegistry",
    # Tasks
    "TaskConfig",
    "CategoryConfig",
    "CategoryGenerationOverrides",
    "TasksFile",
    # Messages
    "ChatMessage",
    "GenerationResult",
    "ToolCall",
    # Results
    "RunResult",
    "TaskResult",
    "ExperimentResult",
    "DeterminismResult",
    "ContextLoadResult",
]
