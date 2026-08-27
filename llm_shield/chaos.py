"""
LLM Shield — Chaos Fault Injection Layer.

THIS MODULE IS NEVER IMPORTED BY PRODUCTION CODE.
It is only loaded when chaos mode is explicitly requested via:
  - Python API:  Shield(provider=ChaosProvider(...))
  - CLI:         python -m tests.chaos_runner --requests 50
  - HTTP:        POST /execute with X-Chaos-Mode: true header
                 (requires SHIELD_ALLOW_CHAOS=true in env as a double lock)

The ChaosProvider wraps the real LiteLLMProvider and probabilistically
injects deliberate faults to verify that llm_shield's state machine is
resilient: every execution must still terminate cleanly in SUCCEEDED or FAILED.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Emitted once at import time so it's impossible to accidentally load chaos
# in production without seeing a log warning.
logger.warning(
    "llm_shield.chaos loaded — ChaosProvider available. "
    "NEVER use in production."
)


# ---------------------------------------------------------------------------
# ChaosConfig — the opt-in gate
# ---------------------------------------------------------------------------

@dataclass
class ChaosConfig:
    """Explicit configuration for chaos fault injection.

    The existence of this object IS the opt-in gate.
    If you never construct a ChaosConfig, no chaos code is ever executed.

    All probabilities are in [0.0, 1.0]. They must not sum to more than 1.0,
    otherwise the remainder is pass-through (no fault).

    Example::

        from llm_shield.chaos import ChaosConfig, ChaosProvider
        from llm_shield.providers import LiteLLMProvider

        chaos = ChaosConfig(
            timeout_rate=0.2,
            server_error_rate=0.2,
            malformed_json_rate=0.2,
            seed=42,           # reproducible — same seed = same fault sequence
        )
        provider = ChaosProvider(base=LiteLLMProvider(), config=chaos)
        shield = Shield(provider=provider)
    """

    # --- Fault probabilities (0.0 = never, 1.0 = always) ---
    timeout_rate: float = 0.15
    """Simulate a timeout by sleeping well past the deadline."""

    server_error_rate: float = 0.15
    """Simulate an HTTP 500 Internal Server Error from the provider."""

    rate_limit_rate: float = 0.10
    """Simulate a 429 Too Many Requests rate-limit response."""

    malformed_json_rate: float = 0.15
    """Return syntactically broken JSON that will fail validation."""

    empty_response_rate: float = 0.05
    """Return an empty string (as if the model returned nothing)."""

    truncated_json_rate: float = 0.05
    """Return a valid-looking but truncated JSON string."""

    # --- Reproducibility ---
    seed: int | None = None
    """Set a seed to make the fault sequence deterministic and replayable."""
    
    healable: bool = False
    """If true, faults are only injected on the first `heal_after_calls` calls. Subsequent calls pass-through."""

    heal_after_calls: int = 3
    """How many calls to inject faults before healing. Only relevant when `healable=True`."""

    def __post_init__(self) -> None:
        total = (
            self.timeout_rate
            + self.server_error_rate
            + self.rate_limit_rate
            + self.malformed_json_rate
            + self.empty_response_rate
            + self.truncated_json_rate
        )
        if total > 1.0:
            raise ValueError(
                f"ChaosConfig: fault rates sum to {total:.2f} > 1.0. "
                "Reduce individual rates so their sum is at most 1.0."
            )
        self._pass_through_rate = 1.0 - total

    @property
    def total_fault_rate(self) -> float:
        return 1.0 - self._pass_through_rate


# ---------------------------------------------------------------------------
# ChaosProvider — wraps the real provider and injects faults
# ---------------------------------------------------------------------------

#: Realistic malformed payloads that replicate common real-world LLM failures.
_MALFORMED_PAYLOADS = [
    "{invalid json",
    '{"passenger_name": "Alex Mercer" "flight_class": "FIRST"}',  # missing comma
    "```json\n{\"broken\": true",                 # markdown fence, not closed
    "Sure! Here is your JSON: {\"id\": 1}",       # prose prefix
    "I cannot provide that in JSON format.",       # model refusal
    '{"key": undefined}',                         # JavaScript-style undefined
]


class ChaosProvider:
    """Wraps a real LLMProvider and injects deliberate faults.

    This class is ONLY instantiated when the caller explicitly passes a
    ChaosConfig. Normal execution creates a LiteLLMProvider directly and
    never touches this class.

    Fault selection uses a seeded RNG for reproducibility — the same seed
    will produce the exact same fault sequence across runs.
    """

    def __init__(self, base: Any, config: ChaosConfig) -> None:
        self._base = base
        self._config = config
        self._rng = random.Random(config.seed)
        self._call_count = 0

        logger.warning(
            "[CHAOS] ChaosProvider active | fault_rate=%.0f%% | seed=%s | "
            "DO NOT USE IN PRODUCTION",
            config.total_fault_rate * 100,
            config.seed,
        )

    async def call(
        self,
        prompt: str,
        model: str,
        timeout: float = 30.0,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        from llm_shield.exceptions import LLMCallError

        self._call_count += 1
        fault = self._pick_fault()

        if fault == "timeout":
            logger.warning("[CHAOS] call #%d → injecting timeout", self._call_count)
            # Raise TimeoutError directly — same as asyncio.wait_for would raise.
            # This avoids actually sleeping past the deadline, keeping tests fast.
            raise asyncio.TimeoutError("CHAOS: Simulated timeout")

        elif fault == "server_error":
            logger.warning("[CHAOS] call #%d → injecting 500 server error", self._call_count)
            raise LLMCallError("CHAOS: Simulated 500 Internal Server Error", model=model)

        elif fault == "rate_limit":
            logger.warning("[CHAOS] call #%d → injecting 429 rate limit", self._call_count)
            raise LLMCallError("CHAOS: Simulated 429 Too Many Requests — retry later", model=model)

        elif fault == "malformed_json":
            payload = self._rng.choice(_MALFORMED_PAYLOADS)
            logger.warning("[CHAOS] call #%d → injecting malformed JSON: %r", self._call_count, payload[:40])
            return payload, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        elif fault == "empty_response":
            logger.warning("[CHAOS] call #%d → injecting empty response", self._call_count)
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        elif fault == "truncated_json":
            logger.warning("[CHAOS] call #%d → injecting truncated JSON", self._call_count)
            return '{"name": "Ali', {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        else:
            # No fault — delegate to the real provider
            logger.debug("[CHAOS] call #%d → pass-through (no fault)", self._call_count)
            return await self._base.call(
                prompt=prompt,
                model=model,
                timeout=timeout,
                system_prompt=system_prompt,
                **kwargs,
            )

    def _pick_fault(self) -> str:
        """Pick a fault type based on configured probabilities using the seeded RNG."""
        if getattr(self._config, 'healable', False) and self._call_count > getattr(self._config, 'heal_after_calls', 3):
            return "pass_through"
            
        r = self._rng.random()
        cfg = self._config

        cumulative = 0.0
        for rate, name in [
            (cfg.timeout_rate, "timeout"),
            (cfg.server_error_rate, "server_error"),
            (cfg.rate_limit_rate, "rate_limit"),
            (cfg.malformed_json_rate, "malformed_json"),
            (cfg.empty_response_rate, "empty_response"),
            (cfg.truncated_json_rate, "truncated_json"),
        ]:
            cumulative += rate
            if r < cumulative:
                return name

        return "pass_through"
