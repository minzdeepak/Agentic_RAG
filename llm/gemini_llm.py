"""
LLM integration with Google Gemini
"""
from typing import List, Dict, Any

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

from llm.base_llm import BaseLLM
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE


class GeminiLLM(BaseLLM):
    """Google Gemini LLM integration"""

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        temperature: float = GEMINI_TEMPERATURE,
    ):
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai>=0.8.0"
            )
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

        super().__init__(model=model, temperature=temperature)

        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )

        self.logger.info(f"Initializing Gemini LLM: {model}")

    # ── BaseLLM abstract implementations ─────────────────────────────────

    def generate(self, prompt: str, stream: bool = False) -> str:
        try:
            if stream:
                response = self._client.generate_content(prompt, stream=True)
                return "".join(chunk.text for chunk in response)
            else:
                response = self._client.generate_content(prompt)
                return response.text
        except Exception as e:
            self.logger.error(f"Gemini generate error: {str(e)}")
            raise

    def chat(self, messages: List[Dict[str, str]]) -> str:
        try:
            # Gemini uses "model" instead of "assistant" for the assistant role
            history = []
            last_user_message = ""

            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    # Prepend system instruction as a user turn with a model ack
                    history.append({"role": "user", "parts": [content]})
                    history.append({"role": "model", "parts": ["Understood."]})
                elif role == "user":
                    last_user_message = content
                elif role == "assistant":
                    history.append({"role": "model", "parts": [content]})

            session = self._client.start_chat(history=history[:-0] if history else [])
            response = session.send_message(last_user_message)
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini chat error: {str(e)}")
            raise

    def list_available_models(self) -> List[str]:
        return ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
        }
