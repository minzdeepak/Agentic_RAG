"""
LLM integration with OpenAI
"""
from typing import List, Dict, Any

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from llm.base_llm import BaseLLM
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE


class OpenAILLM(BaseLLM):
    """OpenAI LLM integration"""

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai is not installed. "
                "Run: pip install openai>=1.40.0"
            )
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        super().__init__(model=model, temperature=temperature)

        self._client = OpenAI(api_key=api_key)
        self.logger.info(f"Initializing OpenAI LLM: {model}")

    # ── BaseLLM abstract implementations ─────────────────────────────────

    def generate(self, prompt: str, stream: bool = False) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]

            if stream:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                )
                return "".join(
                    chunk.choices[0].delta.content or ""
                    for chunk in response
                )
            else:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content or ""

        except Exception as e:
            self.logger.error(f"OpenAI generate error: {str(e)}")
            raise

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # OpenAI message format matches {role, content} exactly — pass through directly
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            self.logger.error(f"OpenAI chat error: {str(e)}")
            raise

    def list_available_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model,
            "temperature": self.temperature,
        }
