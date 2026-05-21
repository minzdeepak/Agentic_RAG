"""
Abstract base class for all LLM providers
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.logger import get_logger


class BaseLLM(ABC):
    """Common interface for all LLM providers"""

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        self.logger = get_logger(__name__)

    # ── Concrete shared methods ───────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant. Use the provided context to answer "
                "the user's question.\nIf the context doesn't contain relevant "
                "information, say so clearly.\nProvide accurate, helpful, and concise answers."
            )

        context_text = "\n".join([f"- {doc}" for doc in context])

        return f"""{system_prompt}

Context:
{context_text}

Question: {query}

Answer:"""

    def generate_answer(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(query, context, system_prompt)
        self.logger.info(f"Generating answer for query: {query[:50]}...")

        try:
            answer = self.generate(prompt)
            return {
                "answer": answer,
                "model": self.model,
                "query": query,
                "context_count": len(context),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            raise

    # ── Abstract methods (each provider must implement) ───────────────────

    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text from a raw prompt."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat interface — messages is a list of {role, content} dicts."""

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """Return a list of available model names for this provider."""

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return provider/model metadata dict."""
