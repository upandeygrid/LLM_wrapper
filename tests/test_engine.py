"""
State machine engine tests — verifies every transition path through the control loop.
"""

import pytest

from llm_shield.config import ShieldConfig
from llm_shield.engine import Shield
from llm_shield.escalation import InMemoryEscalation
from llm_shield.exceptions import LLMCallError
from llm_shield.models import ShieldRequest
from llm_shield.providers import MockProvider


USER_SCHEMA = {
    "type": "object",
    "required": ["name", "age", "email"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string"},
    },
    "additionalProperties": False,
}


def _make_shield(responses, **config_overrides):
    """Helper to create a Shield with a MockProvider."""
    config_kwargs = {
        "default_model": "groq/test",
        "fallback_models": ["groq/fallback-1"],
        "max_retries": 2,
        "max_repairs": 1,
        "timeout_seconds": 5.0,
        "escalation_mode": "in_memory",
    }
    config_kwargs.update(config_overrides)
    config = ShieldConfig(**config_kwargs)
    escalation = InMemoryEscalation()
    provider = MockProvider(responses=responses)
    shield = Shield(config=config, provider=provider, escalation_handler=escalation)
    return shield, escalation, provider


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_valid_response_on_first_try(self):
        """Happy path: LLM returns valid JSON on first attempt."""
        shield, _, _ = _make_shield([
            '{"name": "Alice", "age": 28, "email": "alice@example.com"}'
        ])

        response = await shield.execute(ShieldRequest(
            prompt="Generate a user profile",
            response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"
        assert response.result["name"] == "Alice"
        assert response.result["age"] == 28
        assert response.execution_trace.total_llm_calls == 1
        assert response.execution_trace.retries_used == 0
        assert "SUCCEEDED" in response.execution_trace.states_visited

    @pytest.mark.asyncio
    async def test_no_validators(self):
        """No validators — any response is accepted."""
        shield, _, _ = _make_shield(["Hello, world!"])

        response = await shield.execute(ShieldRequest(prompt="Say hello"))

        assert response.status == "SUCCEEDED"
        assert response.result == "Hello, world!"


class TestRetryPath:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        """Invalid → retry → valid on second try."""
        shield, _, _ = _make_shield([
            '{"name": "Alice", "age": "bad", "email": "a@b.com"}',  # Invalid: age is string
            '{"name": "Alice", "age": 28, "email": "a@b.com"}',     # Valid
        ])

        response = await shield.execute(ShieldRequest(
            prompt="Generate a user profile",
            response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"
        assert response.execution_trace.retries_used >= 1
        assert response.execution_trace.total_llm_calls == 2

    @pytest.mark.asyncio
    async def test_llm_error_triggers_retry(self):
        """LLM error on first call → retry → success."""
        shield, _, _ = _make_shield([
            LLMCallError("Server error", model="groq/test", status_code=500),
            '{"name": "Alice", "age": 28, "email": "a@b.com"}',
        ])

        response = await shield.execute(ShieldRequest(
            prompt="Generate a user profile",
            response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"
        assert response.execution_trace.retries_used >= 1


class TestRepairPath:
    @pytest.mark.asyncio
    async def test_repair_fixes_response(self):
        """Retries exhausted → repair produces valid output."""
        shield, _, _ = _make_shield([
            '{"name": "Alice", "age": "bad", "email": "a@b.com"}',  # Attempt 1: invalid
            '{"name": "Alice", "age": "bad", "email": "a@b.com"}',  # Retry 1: still invalid
            '{"name": "Alice", "age": "bad", "email": "a@b.com"}',  # Retry 2: still invalid
            # Repair call:
            '{"name": "Alice", "age": 28, "email": "a@b.com"}',     # Repair succeeds
        ])

        response = await shield.execute(ShieldRequest(
            prompt="Generate a user profile",
            response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"
        assert response.execution_trace.repairs_used >= 1


class TestFallbackPath:
    @pytest.mark.asyncio
    async def test_fallback_model_succeeds(self):
        """Primary + repairs fail → fallback model produces valid output."""
        shield, _, _ = _make_shield([
            '{"name": "Alice", "age": "bad", "email": "x"}',  # Primary: invalid
            '{"name": "Alice", "age": "bad", "email": "x"}',  # Retry 1
            '{"name": "Alice", "age": "bad", "email": "x"}',  # Retry 2
            '{"name": "Alice", "age": "bad", "email": "x"}',  # Repair: still bad
            # Fallback model:
            '{"name": "Alice", "age": 28, "email": "a@b.com"}',  # Fallback succeeds
        ])

        response = await shield.execute(ShieldRequest(
            prompt="Generate a user profile",
            response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"
        assert response.execution_trace.fallbacks_used >= 1


class TestTemplateFallbackPath:
    @pytest.mark.asyncio
    async def test_template_fallback_succeeds(self):
        """All models fail → template produces valid output."""
        errors = [LLMCallError("fail", model="groq/test")] * 20
        shield, _, _ = _make_shield(errors, max_retries=1, max_repairs=1)

        response = await shield.execute(ShieldRequest(
            prompt="Generate a user profile",
            response_schema=USER_SCHEMA,
            config={"template_response": {"name": "Default", "age": 0, "email": "n/a"}},
        ))

        assert response.status == "SUCCEEDED"
        assert response.result["name"] == "Default"


class TestEscalationPath:
    @pytest.mark.asyncio
    async def test_full_escalation(self):
        """Everything fails → escalation is triggered."""
        errors = [LLMCallError("fail", model="groq/test")] * 20
        shield, escalation, _ = _make_shield(errors, max_retries=1, max_repairs=1)

        response = await shield.execute(ShieldRequest(
            prompt="Generate a user profile",
            response_schema=USER_SCHEMA,
            # No template configured → escalation
        ))

        assert response.status == "FAILED"
        assert response.escalation is not None
        assert response.escalation.original_prompt == "Generate a user profile"
        assert escalation.pending_count == 1

    @pytest.mark.asyncio
    async def test_escalation_contains_full_context(self):
        """Escalation packet contains all required context."""
        errors = [LLMCallError("fail", model="groq/test")] * 20
        shield, escalation, _ = _make_shield(errors, max_retries=1, max_repairs=1)

        response = await shield.execute(ShieldRequest(
            prompt="Generate a user profile",
            response_schema=USER_SCHEMA,
        ))

        packet = response.escalation
        assert packet is not None
        assert packet.original_prompt == "Generate a user profile"
        assert packet.original_model == "groq/test"
        assert packet.response_schema == USER_SCHEMA
        assert len(packet.model_responses) > 0
        assert len(packet.recovery_attempts) > 0


class TestTerminalStateGuarantee:
    @pytest.mark.asyncio
    async def test_always_reaches_terminal_state(self):
        """Every execution must reach either SUCCEEDED or FAILED."""
        test_cases = [
            # Happy path
            (['{"name": "A", "age": 1, "email": "a@b.com"}'], "SUCCEEDED"),
            # All errors, no template
            ([LLMCallError("fail", model="groq/test")] * 20, "FAILED"),
        ]

        for responses, expected_status in test_cases:
            shield, _, _ = _make_shield(responses, max_retries=1, max_repairs=1)
            response = await shield.execute(ShieldRequest(
                prompt="test",
                response_schema=USER_SCHEMA,
            ))
            assert response.status == expected_status
            assert response.execution_trace.total_duration_ms is not None
            assert response.execution_trace.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_trace_is_monotonically_ordered(self):
        """Trace timestamps must be in order."""
        shield, _, _ = _make_shield([
            '{"bad": true}',
            '{"name": "A", "age": 1, "email": "a@b.com"}',
        ])

        response = await shield.execute(ShieldRequest(
            prompt="test",
            response_schema=USER_SCHEMA,
        ))

        transitions = response.execution_trace.transitions
        for i in range(1, len(transitions)):
            assert transitions[i].timestamp >= transitions[i - 1].timestamp
