"""
Repair strategies for fixing invalid LLM responses.

When validation fails and retries are exhausted, repair strategies attempt to
fix the response rather than discarding it entirely.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from llm_shield.models import ValidationErrorDetail
from llm_shield.providers import LLMProvider

logger = logging.getLogger(__name__)


class RepairStrategy(ABC):
    """Abstract base for repair strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def repair(
        self,
        original_prompt: str,
        bad_response: str,
        errors: list[ValidationErrorDetail],
        model: str,
    ) -> str:
        """Attempt to repair a bad response. Returns the repaired text."""
        ...


class LLMRepairStrategy(RepairStrategy):
    """Sends the bad response and validation errors back to the LLM for self-repair."""

    def __init__(self, provider: LLMProvider, timeout: float = 30.0):
        self._provider = provider
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "llm_repair"

    async def repair(
        self,
        original_prompt: str,
        bad_response: str,
        errors: list[ValidationErrorDetail],
        model: str,
    ) -> str:
        error_summary = "\n".join(
            f"- [{e.severity}] {e.field or 'general'}: {e.message}"
            for e in errors
        )

        repair_prompt = (
            "The following response was generated but failed validation.\n\n"
            f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
            f"INVALID RESPONSE:\n{bad_response}\n\n"
            f"VALIDATION ERRORS:\n{error_summary}\n\n"
            "Please fix the response so it passes validation. "
            "Return ONLY the corrected response with no explanation or extra text."
        )

        system_prompt = (
            "You are a response repair assistant. Your job is to fix invalid responses. "
            "Output ONLY the corrected response, nothing else."
        )

        return await self._provider.call(
            prompt=repair_prompt,
            model=model,
            timeout=self._timeout,
            system_prompt=system_prompt,
        )


class RegexRepairStrategy(RepairStrategy):
    """Attempts simple programmatic fixes before involving the LLM.

    Handles common issues:
    - Markdown code fences around JSON
    - Trailing commas in JSON
    - Single quotes instead of double quotes
    """

    @property
    def name(self) -> str:
        return "regex_repair"

    async def repair(
        self,
        original_prompt: str,
        bad_response: str,
        errors: list[ValidationErrorDetail],
        model: str,
    ) -> str:
        text = bad_response.strip()

        # Remove markdown code fences
        match = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        # Try to fix common JSON issues
        try:
            json.loads(text)
            return text  # Already valid after fence removal
        except json.JSONDecodeError:
            pass

        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Replace single quotes with double quotes (naive but covers common cases)
        if "'" in text and '"' not in text:
            text = text.replace("'", '"')

        return text


class ChainedRepairStrategy(RepairStrategy):
    """Tries programmatic repair first, falls back to LLM repair."""

    def __init__(self, provider: LLMProvider, timeout: float = 30.0):
        self._regex = RegexRepairStrategy()
        self._llm = LLMRepairStrategy(provider, timeout)

    @property
    def name(self) -> str:
        return "chained_repair"

    async def repair(
        self,
        original_prompt: str,
        bad_response: str,
        errors: list[ValidationErrorDetail],
        model: str,
    ) -> str:
        # Try programmatic fix first (free, instant)
        try:
            fixed = await self._regex.repair(original_prompt, bad_response, errors, model)
            # Quick check: if it's valid JSON now, return it
            json.loads(fixed)
            logger.info("Regex repair succeeded — skipping LLM repair")
            return fixed
        except (json.JSONDecodeError, Exception):
            pass

        # Fall back to LLM repair
        logger.info("Regex repair insufficient — attempting LLM repair")
        return await self._llm.repair(original_prompt, bad_response, errors, model)
