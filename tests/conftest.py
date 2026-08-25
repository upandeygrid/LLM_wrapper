"""
Shared test fixtures and mock providers for LLM Shield tests.
"""

from __future__ import annotations

import pytest

from llm_shield.config import ShieldConfig
from llm_shield.escalation import InMemoryEscalation
from llm_shield.providers import MockProvider


@pytest.fixture
def mock_provider():
    """A MockProvider that returns a valid JSON response by default."""
    return MockProvider(responses=[
        '{"name": "Alice", "age": 28, "email": "alice@example.com"}'
    ])


@pytest.fixture
def failing_provider():
    """A MockProvider that always fails."""
    from llm_shield.exceptions import LLMCallError
    return MockProvider(responses=[
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
        LLMCallError("Connection refused", model="groq/test"),
    ])


@pytest.fixture
def malformed_then_valid_provider():
    """Returns malformed JSON first, then valid JSON."""
    return MockProvider(responses=[
        '{"name": "Alice", "age": "not_a_number", "email": "bad"}',
        '{"name": "Alice", "age": 28, "email": "alice@example.com"}',
    ])


@pytest.fixture
def test_config():
    """A ShieldConfig with fast defaults for testing."""
    return ShieldConfig(
        default_model="groq/test-model",
        fallback_models=["groq/fallback-1", "groq/fallback-2"],
        max_retries=2,
        max_repairs=1,
        timeout_seconds=5.0,
        escalation_mode="in_memory",
    )


@pytest.fixture
def escalation_handler():
    """An InMemoryEscalation handler for testing."""
    return InMemoryEscalation()


@pytest.fixture
def user_schema():
    """JSON Schema for a user profile."""
    return {
        "type": "object",
        "required": ["name", "age", "email"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string"},
        },
        "additionalProperties": False,
    }
