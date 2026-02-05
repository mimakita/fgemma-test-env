"""Ollama API wrapper with error handling and memory management."""

import logging
import time
from typing import Optional

import httpx
import ollama

from src.config import OLLAMA_HOST

logger = logging.getLogger(__name__)

# Timeout for Ollama requests (seconds)
OLLAMA_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2


class OllamaClient:
    """Wrapper around the ollama Python library."""

    def __init__(self):
        """Initialize with a custom httpx client that has timeout settings."""
        self._client = ollama.Client(
            host=OLLAMA_HOST,
            timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0),
        )

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list] = None,
        options: Optional[dict] = None,
        keep_alive: Optional[str] = None,
    ) -> ollama.ChatResponse:
        """Send a chat request to Ollama with timeout and retry.

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

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat(**kwargs)
                return response
            except httpx.TimeoutException:
                logger.warning(
                    f"Ollama timeout (attempt {attempt + 1}/{MAX_RETRIES}), retrying..."
                )
                time.sleep(RETRY_DELAY)
            except ollama.ResponseError as e:
                logger.error(f"Ollama API error: {e}")
                raise
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.warning(
                        f"Ollama timeout (attempt {attempt + 1}/{MAX_RETRIES}), retrying..."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Unexpected error calling Ollama: {e}")
                    raise

        # Final attempt - raise if it fails
        response = self._client.chat(**kwargs)
        return response

    def unload_model(self, model: str):
        """Force-unload a model by setting keep_alive=0."""
        try:
            ollama.chat(model=model, messages=[], keep_alive="0")
        except Exception:
            pass  # Model may already be unloaded
