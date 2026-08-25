"""
Fallback chain and template fallback logic.

When the primary model and repair strategies fail, the fallback system:
1. Tries each fallback model in order
2. If all models fail, uses a predefined template response
"""

from __future__ import annotations

import json
import logging
from typing import Any

from llm_shield.exceptions import FallbackExhaustedError, TemplateError

logger = logging.getLogger(__name__)


class FallbackChain:
    """Manages the ordered list of fallback models to try."""

    def __init__(self, models: list[str]):
        self._models = list(models)
        self._current_index = 0

    @property
    def has_next(self) -> bool:
        """Whether there are more fallback models to try."""
        return self._current_index < len(self._models)

    @property
    def current_model(self) -> str:
        """Get the current fallback model."""
        if not self.has_next:
            raise FallbackExhaustedError(
                "All fallback models exhausted",
                models_tried=self._models,
            )
        return self._models[self._current_index]

    def advance(self) -> None:
        """Move to the next fallback model."""
        self._current_index += 1

    @property
    def models_tried(self) -> list[str]:
        """List of models that have been tried so far."""
        return self._models[: self._current_index]

    @property
    def total_models(self) -> int:
        return len(self._models)

    def reset(self) -> None:
        self._current_index = 0


class TemplateFallback:
    """Provides predefined template responses when all LLM models fail.

    Templates can be:
    - A dict (returned as-is as JSON)
    - A string (returned as-is)
    - A callable (called with the original prompt, returns a response)
    """

    def __init__(self):
        self._templates: dict[str, Any] = {}

    def register_template(self, key: str, template: dict | str | callable) -> None:
        """Register a template for a given key."""
        self._templates[key] = template

    def get_response(
        self,
        template: dict[str, Any] | str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Get a template response.

        Args:
            template: Direct template value (from request config).
            prompt: Original prompt (for callable templates).

        Returns:
            Template response as a string.

        Raises:
            TemplateError: If no template is available or template processing fails.
        """
        if template is None:
            raise TemplateError("No template response configured")

        try:
            if isinstance(template, dict):
                return json.dumps(template)
            elif isinstance(template, str):
                return template
            elif callable(template):
                result = template(prompt)
                if isinstance(result, dict):
                    return json.dumps(result)
                return str(result)
            else:
                raise TemplateError(
                    f"Invalid template type: {type(template).__name__}",
                    details={"template_type": type(template).__name__},
                )
        except TemplateError:
            raise
        except Exception as e:
            raise TemplateError(
                f"Template processing failed: {e}",
                details={"error": str(e)},
            )
