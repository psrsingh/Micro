from legalrag.llm.base import LLMAdapter, LLMError, LLMRequest, LLMResponse, parse_structured
from legalrag.settings import LLMConfig


def build_llm(cfg: LLMConfig) -> LLMAdapter:
    """The only place a provider name turns into a class."""
    if cfg.provider == "fake":
        from legalrag.llm.fake import FakeLLM

        return FakeLLM(cfg)
    if cfg.provider == "ollama":
        from legalrag.llm.ollama import OllamaAdapter

        return OllamaAdapter(cfg)
    if cfg.provider == "anthropic":
        from legalrag.llm.anthropic_api import AnthropicAdapter

        return AnthropicAdapter(cfg)
    raise ValueError(f"unknown llm provider: {cfg.provider}")


__all__ = ["LLMAdapter", "LLMError", "LLMRequest", "LLMResponse", "build_llm", "parse_structured"]
