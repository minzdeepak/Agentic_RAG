"""
LLM factory — returns a BaseLLM instance based on LLM_PROVIDER setting
"""
from llm.base_llm import BaseLLM
from config.settings import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_TOP_P, OLLAMA_TIMEOUT,
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE,
    CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS,
)

_VALID_PROVIDERS = {"ollama", "gemini", "openai", "claude"}


def get_llm() -> BaseLLM:
    """Instantiate and return the configured LLM provider."""
    provider = LLM_PROVIDER.lower().strip()

    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Valid values: {', '.join(sorted(_VALID_PROVIDERS))}"
        )

    if provider == "ollama":
        from llm.ollama_llm import OllamaLLM
        return OllamaLLM(OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_TOP_P, OLLAMA_TIMEOUT)

    if provider == "gemini":
        from llm.gemini_llm import GeminiLLM
        return GeminiLLM(GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE)

    if provider == "openai":
        from llm.openai_llm import OpenAILLM
        return OpenAILLM(OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE)

    # provider == "claude"
    from llm.claude_llm import ClaudeLLM
    return ClaudeLLM(CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS)
