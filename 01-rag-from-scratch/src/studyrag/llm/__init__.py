"""LLM backends. `load_llm(settings)` picks one based on settings.llm_backend."""

from __future__ import annotations

from ..config import Settings
from .base import LLM, Message, parse_json_loose, strip_thinking


def load_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "llamacpp":
        from .llamacpp import LlamaCppLLM
        return LlamaCppLLM(settings.llm_model_path, n_ctx=settings.llm_context, n_threads=settings.llm_threads)
    if settings.llm_backend == "openai":
        from .openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(settings.api_base_url, settings.api_model, settings.api_key)
    raise ValueError(f"unknown llm_backend '{settings.llm_backend}' (use 'llamacpp' or 'openai')")


__all__ = ["LLM", "Message", "load_llm", "parse_json_loose", "strip_thinking"]
