"""
State and event definitions for the LLM Shield state machine.

The state machine enforces a deterministic control loop:
INIT → CALLING_LLM → VALIDATING → SUCCEEDED
                   ↘ RETRYING → REPAIRING → FALLING_BACK → TEMPLATE_FALLBACK → ESCALATING → FAILED
"""

from enum import Enum, auto


class ExecutionState(Enum):
    """All possible states in the Shield execution lifecycle."""

    INIT = auto()
    CALLING_LLM = auto()
    VALIDATING = auto()
    RETRYING = auto()
    REPAIRING = auto()
    FALLING_BACK = auto()
    TEMPLATE_FALLBACK = auto()
    ESCALATING = auto()
    SUCCEEDED = auto()
    FAILED = auto()

    @property
    def is_terminal(self) -> bool:
        """Whether this state is a final state (no further transitions)."""
        return self in _TERMINAL_STATES

    @property
    def is_recovery(self) -> bool:
        """Whether this state is part of the recovery pipeline."""
        return self in _RECOVERY_STATES


_TERMINAL_STATES = frozenset({ExecutionState.SUCCEEDED, ExecutionState.FAILED})

_RECOVERY_STATES = frozenset({
    ExecutionState.RETRYING,
    ExecutionState.REPAIRING,
    ExecutionState.FALLING_BACK,
    ExecutionState.TEMPLATE_FALLBACK,
    ExecutionState.ESCALATING,
})

# Valid state transitions — used for assertion / debugging.
# Maps each state to the set of states it is allowed to transition to.
VALID_TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.INIT: frozenset({ExecutionState.CALLING_LLM}),
    ExecutionState.CALLING_LLM: frozenset({
        ExecutionState.VALIDATING,
        ExecutionState.RETRYING,
    }),
    ExecutionState.VALIDATING: frozenset({
        ExecutionState.SUCCEEDED,
        ExecutionState.RETRYING,
        ExecutionState.REPAIRING,
    }),
    ExecutionState.RETRYING: frozenset({
        ExecutionState.CALLING_LLM,
        ExecutionState.REPAIRING,
    }),
    ExecutionState.REPAIRING: frozenset({
        ExecutionState.VALIDATING,
        ExecutionState.FALLING_BACK,
    }),
    ExecutionState.FALLING_BACK: frozenset({
        ExecutionState.CALLING_LLM,
        ExecutionState.TEMPLATE_FALLBACK,
    }),
    ExecutionState.TEMPLATE_FALLBACK: frozenset({
        ExecutionState.SUCCEEDED,
        ExecutionState.ESCALATING,
    }),
    ExecutionState.ESCALATING: frozenset({ExecutionState.FAILED}),
    ExecutionState.SUCCEEDED: frozenset(),
    ExecutionState.FAILED: frozenset(),
}
