"""
Custom exception hierarchy for LLM Shield.

All exceptions inherit from ShieldError so callers can catch broadly or narrowly.
"""


class ShieldError(Exception):
    """Base exception for all LLM Shield errors."""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class LLMCallError(ShieldError):
    """Raised when an LLM provider call fails (timeout, HTTP error, etc.)."""

    def __init__(self, message: str, *, model: str | None = None, status_code: int | None = None,
                 details: dict | None = None):
        super().__init__(message, details=details)
        self.model = model
        self.status_code = status_code


class ValidationFailedError(ShieldError):
    """Raised when the LLM response fails validation."""

    def __init__(self, message: str, *, errors: list | None = None,
                 details: dict | None = None):
        super().__init__(message, details=details)
        self.errors = errors or []


class RepairFailedError(ShieldError):
    """Raised when repair attempts are exhausted without producing valid output."""

    def __init__(self, message: str, *, attempt: int = 0, details: dict | None = None):
        super().__init__(message, details=details)
        self.attempt = attempt


class FallbackExhaustedError(ShieldError):
    """Raised when all fallback models have been tried and failed."""

    def __init__(self, message: str, *, models_tried: list[str] | None = None,
                 details: dict | None = None):
        super().__init__(message, details=details)
        self.models_tried = models_tried or []


class EscalationError(ShieldError):
    """Raised when the escalation mechanism itself fails."""

    def __init__(self, message: str, *, channel: str | None = None,
                 details: dict | None = None):
        super().__init__(message, details=details)
        self.channel = channel


class TemplateError(ShieldError):
    """Raised when the template fallback fails (missing template, validation error, etc.)."""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message, details=details)


class InvalidTransitionError(ShieldError):
    """Raised when the state machine attempts an invalid state transition."""

    def __init__(self, from_state, to_state, *, details: dict | None = None):
        message = f"Invalid transition: {from_state.name} → {to_state.name}"
        super().__init__(message, details=details)
        self.from_state = from_state
        self.to_state = to_state
