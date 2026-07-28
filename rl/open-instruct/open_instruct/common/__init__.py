# Common Package
# Shared components across all stages including:
# - Agent class for LLM context management
# - Configuration loading
# - Utility functions (file I/O, DAG validation, prompt loading)
# - Client interfaces (LLM)
# - Environment infrastructure

from typing import TYPE_CHECKING

from .config import Config, load_config, ConfigError
from .agent import Agent, AgentError, create_agent_from_config
from .utils import (
    load_json,
    save_json,
    ensure_dir,
    list_json_files,
    load_prompt,
    format_prompt,
    topological_sort,
    build_dependencies,
)
from .clients import LLMClient, LLMClientError, ContentPolicyError

if TYPE_CHECKING:
    pass

__all__ = [
    "Config",
    "load_config",
    "ConfigError",
    "Agent",
    "AgentError",
    "create_agent_from_config",
    "LLMClient",
    "LLMClientError",
    "ContentPolicyError",
    "load_json",
    "save_json",
    "ensure_dir",
    "list_json_files",
    "load_prompt",
    "format_prompt",
    "topological_sort",
    "build_dependencies",
]


def __getattr__(name: str):
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
