"""Ollama API wrapper with error handling and memory management."""

import logging
from typing import Optional

import ollama

from src.config import OLLAMA_HOST

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper around the ollama Python library."""

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list] = None,
        options: Optional[dict] = None,
        keep_alive: Optional[str] = None,
    ) -> ollama.ChatResponse:
        """Send a chat request to Ollama.

        Args:
            model: Ollama model name
            messages: List of message dicts with role and content
            tools: Optional list of tool schemas for function calling
            options: Optional model parameters (temperature, num_ctx, etc.)
            keep_alive: How long to keep model in memory

        Returns:
            ChatResponse from Ollama
        """
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if options:
            kwargs["options"] = options
        if keep_alive:
            kwargs["keep_alive"] = keep_alive

        try:
            response = ollama.chat(**kwargs)
            return response
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}")
            raise

    def unload_model(self, model: str):
        """Force-unload a model by setting keep_alive=0."""
        try:
            ollama.chat(model=model, messages=[], keep_alive="0")
        except Exception:
            pass  # Model may already be unloaded
