"""
Pydantic data models for LLM Shield requests, responses, traces, and escalations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationErrorDetail(BaseModel):
    """A single validation error found in an LLM response."""

    field: str | None = None
    message: str
    severity: str = "error"  # "error" | "warning"
    validator_name: str | None = None


# ---------------------------------------------------------------------------
# Execution Trace
# ---------------------------------------------------------------------------

class StateTransitionRecord(BaseModel):
    """Record of a single state transition."""

    from_state: str
    to_state: str
    reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMCallRecord(BaseModel):
    """Record of a single LLM API call."""

    model: str
    attempt: int
    phase: str  # "primary", "retry", "repair", "fallback"
    response: str | None = None
    error: str | None = None
    duration_ms: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecoveryAttemptRecord(BaseModel):
    """Record of a single recovery attempt."""

    type: str  # "retry", "repair", "fallback", "template"
    attempt: int
    model: str | None = None
    result: str  # "success", "invalid", "error", "timeout"
    detail: str | None = None


class ExecutionTrace(BaseModel):
    """Complete trace of an execution run — returned with every response."""

    states_visited: list[str] = Field(default_factory=list)
    transitions: list[StateTransitionRecord] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    recovery_attempts: list[RecoveryAttemptRecord] = Field(default_factory=list)
    validation_errors: list[ValidationErrorDetail] = Field(default_factory=list)
    total_duration_ms: float | None = None
    total_llm_calls: int = 0
    retries_used: int = 0
    repairs_used: int = 0
    fallbacks_used: int = 0
    final_model: str | None = None


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

class EscalationStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"


class EscalationPacket(BaseModel):
    """Full context packet for human escalation — contains everything needed to take action."""

    id: str = Field(default_factory=lambda: f"esc_{uuid.uuid4().hex[:12]}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: EscalationStatus = EscalationStatus.PENDING
    reason: str
    original_prompt: str
    original_model: str
    response_schema: dict[str, Any] | None = None
    validation_errors: list[ValidationErrorDetail] = Field(default_factory=list)
    model_responses: list[LLMCallRecord] = Field(default_factory=list)
    recovery_attempts: list[RecoveryAttemptRecord] = Field(default_factory=list)
    final_state: str | None = None
    resolved_at: datetime | None = None
    resolution_notes: str | None = None


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class ShieldRequestConfig(BaseModel):
    """Per-request configuration overrides."""

    max_retries: int | None = None
    max_repairs: int | None = None
    timeout_seconds: float | None = None
    fallback_models: list[str] | None = None
    template_response: dict[str, Any] | str | None = None


class ShieldRequest(BaseModel):
    """Input to the Shield engine."""

    prompt: str = Field(..., min_length=1, description="The prompt to send to the LLM")
    model: str | None = Field(
        default=None,
        description="LLM model identifier (e.g. 'groq/llama-3.1-70b-versatile'). "
                    "Falls back to config default.",
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema to validate the LLM response against.",
    )
    validators: list[dict[str, Any]] | None = Field(
        default=None,
        description="Additional validators to apply. Each dict has 'type' and type-specific params.",
    )
    config: ShieldRequestConfig = Field(default_factory=ShieldRequestConfig)
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt for the LLM call.",
    )


class ShieldResponse(BaseModel):
    """Output from the Shield engine — always returned, never raises."""

    status: str  # "SUCCEEDED" or "FAILED"
    result: Any | None = None
    error: str | None = None
    escalation: EscalationPacket | None = None
    execution_trace: ExecutionTrace = Field(default_factory=ExecutionTrace)
