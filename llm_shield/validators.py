"""
Pluggable validation system for LLM responses.

Validators check whether an LLM response meets the expected format/schema.
They return a list of ValidationErrorDetail objects — an empty list means success.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

import jsonschema

from llm_shield.models import ValidationErrorDetail


class Validator(ABC):
    """Abstract base class for response validators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable validator name."""
        ...

    @abstractmethod
    def validate(self, response: str) -> list[ValidationErrorDetail]:
        """Validate the response. Return empty list if valid."""
        ...


class JsonSchemaValidator(Validator):
    """Validates that the response is valid JSON conforming to a JSON Schema."""

    def __init__(self, schema: dict[str, Any]):
        self._schema = schema

    @property
    def name(self) -> str:
        return "json_schema"

    def validate(self, response: str) -> list[ValidationErrorDetail]:
        errors: list[ValidationErrorDetail] = []

        # Step 1: Parse as JSON
        cleaned = _strip_markdown_fences(response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            errors.append(ValidationErrorDetail(
                field=None,
                message=f"Invalid JSON: {e}",
                severity="error",
                validator_name=self.name,
            ))
            return errors

        # Step 2: Validate against schema
        validator = jsonschema.Draft7Validator(self._schema)
        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.absolute_path) or "(root)"
            errors.append(ValidationErrorDetail(
                field=path,
                message=error.message,
                severity="error",
                validator_name=self.name,
            ))

        return errors


class RegexValidator(Validator):
    """Validates that the response matches a regex pattern."""

    def __init__(self, pattern: str, description: str = "regex pattern"):
        self._pattern = re.compile(pattern, re.DOTALL)
        self._description = description

    @property
    def name(self) -> str:
        return f"regex({self._description})"

    def validate(self, response: str) -> list[ValidationErrorDetail]:
        if not self._pattern.search(response):
            return [ValidationErrorDetail(
                field=None,
                message=f"Response does not match pattern: {self._description}",
                severity="error",
                validator_name=self.name,
            )]
        return []


class LengthValidator(Validator):
    """Validates response length is within bounds."""

    def __init__(self, min_length: int = 0, max_length: int | None = None):
        self._min = min_length
        self._max = max_length

    @property
    def name(self) -> str:
        return "length"

    def validate(self, response: str) -> list[ValidationErrorDetail]:
        errors: list[ValidationErrorDetail] = []
        length = len(response.strip())

        if length < self._min:
            errors.append(ValidationErrorDetail(
                field=None,
                message=f"Response too short: {length} chars (min: {self._min})",
                severity="error",
                validator_name=self.name,
            ))
        if self._max is not None and length > self._max:
            errors.append(ValidationErrorDetail(
                field=None,
                message=f"Response too long: {length} chars (max: {self._max})",
                severity="error",
                validator_name=self.name,
            ))

        return errors


class CustomValidator(Validator):
    """Wraps any callable as a validator.

    The callable should accept a string and return a list of ValidationErrorDetail.
    """

    def __init__(self, fn: Callable[[str], list[ValidationErrorDetail]],
                 validator_name: str = "custom"):
        self._fn = fn
        self._name = validator_name

    @property
    def name(self) -> str:
        return self._name

    def validate(self, response: str) -> list[ValidationErrorDetail]:
        return self._fn(response)


class CompositeValidator(Validator):
    """Runs multiple validators and collects all errors."""

    def __init__(self, validators: list[Validator]):
        self._validators = validators

    @property
    def name(self) -> str:
        return "composite"

    def validate(self, response: str) -> list[ValidationErrorDetail]:
        all_errors: list[ValidationErrorDetail] = []
        for v in self._validators:
            all_errors.extend(v.validate(response))
        return all_errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove reasoning tags and markdown code fences from LLM responses."""
    text = text.strip()
    
    # Remove <think>...</think> blocks common in reasoning models like Qwen or DeepSeek
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
    # Match ```json\n...\n``` or ```\n...\n``` anywhere in the remaining text
    match = re.search(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def build_validators(
    response_schema: dict[str, Any] | None = None,
    validator_configs: list[dict[str, Any]] | None = None,
) -> Validator | None:
    """Build a composite validator from request parameters.

    Args:
        response_schema: JSON schema for the response.
        validator_configs: List of validator configs, each with 'type' and params.

    Returns:
        A Validator instance, or None if no validators are configured.
    """
    validators: list[Validator] = []

    if response_schema:
        validators.append(JsonSchemaValidator(response_schema))

    if validator_configs:
        for cfg in validator_configs:
            vtype = cfg.get("type", "")
            if vtype == "regex":
                validators.append(RegexValidator(
                    pattern=cfg["pattern"],
                    description=cfg.get("description", "custom regex"),
                ))
            elif vtype == "length":
                validators.append(LengthValidator(
                    min_length=cfg.get("min_length", 0),
                    max_length=cfg.get("max_length"),
                ))

    if not validators:
        return None
    if len(validators) == 1:
        return validators[0]
    return CompositeValidator(validators)
