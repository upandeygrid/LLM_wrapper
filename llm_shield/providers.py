"""
LLM provider abstraction.

Uses litellm for unified access to Groq and other providers.
Includes a MockProvider for deterministic testing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def call(
        self,
        prompt: str,
        model: str,
        timeout: float = 30.0,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM and return the response text.

        Raises:
            LLMCallError: On provider errors, timeouts, etc.
        """
        ...


class LiteLLMProvider(LLMProvider):
    """Production provider using litellm for unified LLM access.

    Supports Groq, OpenAI, Anthropic, and 100+ other providers through litellm.
    """

    async def call(
        self,
        prompt: str,
        model: str,
        timeout: float = 30.0,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        import litellm
        from litellm import acompletion

        from llm_shield.exceptions import LLMCallError

        litellm.suppress_debug_info = True

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                acompletion(model=model, messages=messages, **kwargs),
                timeout=timeout,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            content = response.choices[0].message.content
            logger.info(
                "LLM call succeeded: model=%s, duration_ms=%.1f, tokens=%s",
                model, elapsed_ms,
                getattr(response, "usage", None),
            )
            return content

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            raise LLMCallError(
                f"LLM call timed out after {timeout}s",
                model=model,
                details={"duration_ms": elapsed_ms, "timeout": timeout},
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            # Check for specific HTTP status codes from litellm
            status_code = getattr(e, "status_code", None)
            raise LLMCallError(
                f"LLM call failed: {type(e).__name__}: {e}",
                model=model,
                status_code=status_code,
                details={"duration_ms": elapsed_ms, "error_type": type(e).__name__},
            )


class MockProvider(LLMProvider):
    """Mock provider for testing with configurable failure injection.

    Usage:
        # Always return a fixed response
        provider = MockProvider(responses=["valid json response"])

        # Fail on first call, succeed on second
        provider = MockProvider(responses=[
            LLMCallError("timeout"),
            "valid response",
        ])

        # Simulate timeout
        provider = MockProvider(responses=[TimeoutError()])
    """

    def __init__(
        self,
        responses: list[str | Exception] | None = None,
        delay: float = 0.0,
    ):
        self._responses = list(responses or [])
        self._delay = delay
        self._call_count = 0
        self.call_history: list[dict[str, Any]] = []

    async def call(
        self,
        prompt: str,
        model: str,
        timeout: float = 30.0,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        from llm_shield.exceptions import LLMCallError

        self.call_history.append({
            "prompt": prompt,
            "model": model,
            "timeout": timeout,
            "system_prompt": system_prompt,
            "call_number": self._call_count,
        })

        if self._delay > 0:
            if self._delay > timeout:
                await asyncio.sleep(timeout)
                self._call_count += 1
                raise LLMCallError(
                    f"LLM call timed out after {timeout}s",
                    model=model,
                )
            await asyncio.sleep(self._delay)

        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        elif self._responses:
            response = self._responses[-1]  # Repeat last response
        else:
            response = '{"result": "mock response"}'

        self._call_count += 1

        if isinstance(response, Exception):
            if isinstance(response, LLMCallError):
                raise response
            raise LLMCallError(
                f"Mock error: {response}",
                model=model,
                details={"original_error": str(response)},
            )

        return response
