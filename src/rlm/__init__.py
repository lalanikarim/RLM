"""Recursive Language Models package."""

from .config import get_config
from .core import RecursiveLanguageModel
from .models import RLMConfig, RLMResult
from .prompts import RECURSE_SYSTEM_PROMPT, ROOT_SYSTEM_PROMPT

__all__ = [
    "RecursiveLanguageModel",
    "RLMConfig",
    "RLMResult",
    "ROOT_SYSTEM_PROMPT",
    "RECURSE_SYSTEM_PROMPT",
    "get_config",
]

__version__ = "0.1.0"
