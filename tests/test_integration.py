"""
Integration tests — deliberate failure injection matrix.

Each test injects a specific type of failure and verifies that the engine
reaches the correct terminal state with a valid execution trace.
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


def _shield(responses, template=None, max_retries=1, max_repairs=1, fallbacks=None):
    """Helper to create a fully wired Shield with mock provider."""
    config = ShieldConfig(
        default_model="groq/test",
        fallback_models=fallbacks or ["groq/fallback-1"],
        max_retries=max_retries,
        max_repairs=max_repairs,
        timeout_seconds=5.0,
        escalation_mode="in_memory",
    )
    escalation = InMemoryEscalation()
    provider = MockProvider(responses=responses)
    shield = Shield(config=config, provider=provider, escalation_handler=escalation)
    return shield, escalation, provider


class TestMalformedJsonResponse:
    """Injected failure: LLM returns syntactically invalid JSON."""

    @pytest.mark.asyncio
    async def test_malformed_json_triggers_retry(self):
        shield, _, _ = _shield([
            '{invalid json!!!',
            '{"name": "A", "age": 1, "email": "a@b.com"}',
        ])

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"
        assert response.execution_trace.retries_used >= 1

    @pytest.mark.asyncio
    async def test_persistent_malformed_json_escalates(self):
        shield, escalation, _ = _shield(
            ['{bad}'] * 20,
            max_retries=1, max_repairs=1,
        )

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        assert response.status == "FAILED"
        assert escalation.pending_count >= 1


class TestModelServerError:
    """Injected failure: LLM provider returns 500 errors."""

    @pytest.mark.asyncio
    async def test_500_retries_then_succeeds(self):
        shield, _, _ = _shield([
            LLMCallError("Internal Server Error", model="groq/test", status_code=500),
            '{"name": "A", "age": 1, "email": "a@b.com"}',
        ])

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"

    @pytest.mark.asyncio
    async def test_persistent_500_escalates(self):
        errors = [LLMCallError("500", model="groq/test", status_code=500)] * 20
        shield, escalation, _ = _shield(errors, max_retries=1, max_repairs=1)

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        assert response.status == "FAILED"
        assert escalation.pending_count >= 1


class TestTimeout:
    """Injected failure: LLM call times out."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self):
        shield, _, _ = _shield([
            LLMCallError("Timeout after 5s", model="groq/test"),
            '{"name": "A", "age": 1, "email": "a@b.com"}',
        ])

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"
        assert response.execution_trace.retries_used >= 1


class TestRepairMakesItWorse:
    """Injected failure: Repair returns even worse output."""

    @pytest.mark.asyncio
    async def test_bad_repair_leads_to_fallback(self):
        shield, _, _ = _shield([
            '{"name": "A", "age": "bad", "email": "x"}',   # Primary: invalid
            '{"name": "A", "age": "bad", "email": "x"}',   # Retry: still invalid
            '{"age": "still bad"}',                          # Repair: even worse
            # Fallback model:
            '{"name": "B", "age": 2, "email": "b@b.com"}', # Fallback succeeds
        ])

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        assert response.status == "SUCCEEDED"
        assert response.execution_trace.fallbacks_used >= 1


class TestAllFallbackModelsDown:
    """Injected failure: Every model (primary + all fallbacks) returns errors."""

    @pytest.mark.asyncio
    async def test_all_models_down_with_template(self):
        errors = [LLMCallError("down", model="groq/test")] * 20
        shield, _, _ = _shield(
            errors,
            max_retries=0, max_repairs=0,
            fallbacks=["groq/fb1", "groq/fb2"],
        )

        response = await shield.execute(ShieldRequest(
            prompt="test",
            response_schema=USER_SCHEMA,
            config={"template_response": {"name": "Default", "age": 0, "email": "n/a"}},
        ))

        assert response.status == "SUCCEEDED"
        assert response.result["name"] == "Default"

    @pytest.mark.asyncio
    async def test_all_models_down_no_template_escalates(self):
        errors = [LLMCallError("down", model="groq/test")] * 20
        shield, escalation, _ = _shield(
            errors,
            max_retries=0, max_repairs=0,
            fallbacks=["groq/fb1"],
        )

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        assert response.status == "FAILED"
        assert escalation.pending_count >= 1


class TestTemplateFailsValidation:
    """Injected failure: Template response doesn't pass validation."""

    @pytest.mark.asyncio
    async def test_invalid_template_escalates(self):
        errors = [LLMCallError("down", model="groq/test")] * 20
        shield, escalation, _ = _shield(
            errors,
            max_retries=0, max_repairs=0,
            fallbacks=[],
        )

        # Template is missing required 'email' field
        response = await shield.execute(ShieldRequest(
            prompt="test",
            response_schema=USER_SCHEMA,
            config={"template_response": {"name": "X", "age": 0}},
        ))

        assert response.status == "FAILED"
        assert escalation.pending_count >= 1


class TestCascadingFailures:
    """Injected failure: Multiple failure types in sequence."""

    @pytest.mark.asyncio
    async def test_timeout_plus_malformed_plus_repair_fail(self):
        """Timeout → malformed → bad repair → fallback down → escalation."""
        shield, escalation, _ = _shield(
            [
                LLMCallError("timeout", model="groq/test"),     # Timeout
                '{broken json',                                   # Malformed
                '{still broken',                                  # Repair attempt: still bad
                LLMCallError("fallback down", model="groq/fb"),  # Fallback down
            ],
            max_retries=1, max_repairs=1, fallbacks=["groq/fb"],
        )

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        assert response.status == "FAILED"
        assert response.escalation is not None
        assert escalation.pending_count >= 1
        # Verify trace captured all phases
        trace = response.execution_trace
        assert trace.total_llm_calls >= 2
        assert trace.total_duration_ms > 0


class TestTraceCompleteness:
    """Verify that execution traces are complete and accurate."""

    @pytest.mark.asyncio
    async def test_succeeded_trace_structure(self):
        shield, _, _ = _shield([
            '{"name": "A", "age": 1, "email": "a@b.com"}',
        ])

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        trace = response.execution_trace
        assert trace.total_llm_calls == 1
        assert trace.total_duration_ms > 0
        assert "SUCCEEDED" in trace.states_visited
        assert len(trace.transitions) > 0
        assert all(t.timestamp is not None for t in trace.transitions)

    @pytest.mark.asyncio
    async def test_failed_trace_has_all_attempts(self):
        errors = [LLMCallError("fail", model="groq/test")] * 20
        shield, _, _ = _shield(errors, max_retries=2, max_repairs=1)

        response = await shield.execute(ShieldRequest(
            prompt="test", response_schema=USER_SCHEMA,
        ))

        trace = response.execution_trace
        assert trace.total_llm_calls >= 2  # At least primary + some retries
        assert len(trace.recovery_attempts) >= 1
        assert "FAILED" in trace.states_visited
