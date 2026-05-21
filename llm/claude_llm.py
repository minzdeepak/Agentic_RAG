"""
LLM integration with Anthropic Claude
"""
from typing import List, Dict, Any

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from llm.base_llm import BaseLLM
from config.settings import (
    CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS
)


class ClaudeLLM(BaseLLM):
    """Anthropic Claude LLM integration"""

    def __init__(
        self,
        api_key: str = CLAUDE_API_KEY,
        model: str = CLAUDE_MODEL,
        temperature: float = CLAUDE_TEMPERATURE,
        max_tokens: int = CLAUDE_MAX_TOKENS,
    ):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic is not installed. "
                "Run: pip install anthropic>=0.34.0"
            )
        if not api_key:
            raise ValueError("CLAUDE_API_KEY is required when LLM_PROVIDER=claude")

        super().__init__(model=model, temperature=temperature)

        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)
        self.logger.info(f"Initializing Claude LLM: {model}")

    # ── BaseLLM abstract implementations ─────────────────────────────────

    def generate(self, prompt: str, stream: bool = False) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Claude generate error: {str(e)}")
            raise

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Anthropic requires system prompt as a separate top-level parameter,
        # not inside the messages array.
        try:
            system_content = None
            filtered_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    filtered_messages.append(msg)

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": filtered_messages,
            }
            if system_content:
                kwargs["system"] = system_content

            response = self._client.messages.create(**kwargs)
            return response.content[0].text

        except Exception as e:
            self.logger.error(f"Claude chat error: {str(e)}")
            raise

    def list_available_models(self) -> List[str]:
        return ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-4-5"]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "claude",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
