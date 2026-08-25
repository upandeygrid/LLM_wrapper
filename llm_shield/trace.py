"""
Execution tracer — records every state transition, LLM call, and recovery attempt.

Produces a complete ExecutionTrace that is returned with every ShieldResponse,
giving full observability into what happened and why.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from llm_shield.models import (
    ExecutionTrace,
    LLMCallRecord,
    RecoveryAttemptRecord,
    StateTransitionRecord,
    ValidationErrorDetail,
)
from llm_shield.states import ExecutionState


class ExecutionTracer:
    """Collects trace data throughout an execution run."""

    def __init__(self):
        self._trace = ExecutionTrace()
        self._start_time: float = time.monotonic()

    def record_transition(
        self,
        from_state: ExecutionState,
        to_state: ExecutionState,
        reason: str,
        metadata: dict | None = None,
    ) -> None:
        """Record a state transition."""
        record = StateTransitionRecord(
            from_state=from_state.name,
            to_state=to_state.name,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        self._trace.transitions.append(record)

        if to_state.name not in self._trace.states_visited:
            self._trace.states_visited.append(to_state.name)

    def record_llm_call(
        self,
        model: str,
        attempt: int,
        phase: str,
        response: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Record an LLM API call."""
        record = LLMCallRecord(
            model=model,
            attempt=attempt,
            phase=phase,
            response=response,
            error=error,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc),
        )
        self._trace.llm_calls.append(record)
        self._trace.total_llm_calls += 1
        self._trace.final_model = model

    def record_recovery_attempt(
        self,
        type_: str,
        attempt: int,
        result: str,
        model: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Record a recovery attempt (retry, repair, fallback, template)."""
        record = RecoveryAttemptRecord(
            type=type_,
            attempt=attempt,
            model=model,
            result=result,
            detail=detail,
        )
        self._trace.recovery_attempts.append(record)

        if type_ == "retry":
            self._trace.retries_used += 1
        elif type_ == "repair":
            self._trace.repairs_used += 1
        elif type_ == "fallback":
            self._trace.fallbacks_used += 1

    def record_validation_errors(self, errors: list[ValidationErrorDetail]) -> None:
        """Record validation errors from a validation pass."""
        self._trace.validation_errors.extend(errors)

    def finalize(self) -> ExecutionTrace:
        """Finalize the trace with total duration."""
        elapsed = time.monotonic() - self._start_time
        self._trace.total_duration_ms = round(elapsed * 1000, 2)
        return self._trace

    @property
    def trace(self) -> ExecutionTrace:
        """Access the current (possibly incomplete) trace."""
        return self._trace
