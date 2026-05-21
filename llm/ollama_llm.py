"""
LLM integration with Ollama
"""
from typing import List, Dict, Any
import requests
import json

from llm.base_llm import BaseLLM
from config.settings import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P, OLLAMA_TIMEOUT
)


class OllamaLLM(BaseLLM):
    """Ollama LLM integration for local/on-prem inference"""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        temperature: float = OLLAMA_TEMPERATURE,
        top_p: float = OLLAMA_TOP_P,
        timeout: int = OLLAMA_TIMEOUT
    ):
        super().__init__(model=model, temperature=temperature)
        self.base_url = base_url.rstrip("/")
        self.top_p = top_p
        self.timeout = timeout

        self.logger.info(f"Initializing Ollama LLM: {model} at {base_url}")

        if not self._check_connection():
            self.logger.warning(f"Could not connect to Ollama at {base_url}")

    # ── Connection check ──────────────────────────────────────────────────

    def _check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Connection check failed: {str(e)}")
            return False

    # ── BaseLLM abstract implementations ─────────────────────────────────

    def generate(self, prompt: str, stream: bool = False) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=stream
            )

            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.text}")

            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        full_response += data.get("response", "")
                return full_response
            else:
                return response.json().get("response", "")

        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timeout")
            raise
        except Exception as e:
            self.logger.error(f"Error in generate: {str(e)}")
            raise

    def chat(self, messages: List[Dict[str, str]]) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.text}")

            return response.json().get("message", {}).get("content", "")

        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            raise

    def list_available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                self.logger.info(f"Available Ollama models: {models}")
                return models
            return []
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }
