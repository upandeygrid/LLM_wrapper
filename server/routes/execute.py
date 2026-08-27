"""
Main execution endpoint — POST /execute.

This is the primary API for running LLM calls through the Shield control loop.

Chaos Mode (opt-in, double-locked):
  To enable chaos fault injection via the API, TWO conditions must BOTH be true:
    1. Request header: X-Chaos-Mode: true
    2. Environment:    SHIELD_ALLOW_CHAOS=true

  If either is missing, the request runs in normal mode with zero chaos code
  in the execution path. This double-lock prevents accidental activation.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, Header

from llm_shield.engine import Shield
from llm_shield.models import ShieldRequest, ShieldResponse
from server.dependencies import get_shield

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Execution"])


def _build_chaos_shield(request: ShieldRequest, chaos_mode: str) -> Shield:
    """
    Build a fresh Shield with ChaosProvider for this single request.

    Called ONLY when both the X-Chaos-Mode header AND the SHIELD_ALLOW_CHAOS
    env var are present. A fresh Shield is used (not the singleton) so that
    chaos mode never leaks into normal requests.
    """
    # Local import — chaos module is never loaded for normal requests
    from llm_shield.chaos import ChaosConfig, ChaosProvider
    from llm_shield.config import ShieldConfig
    from llm_shield.providers import LiteLLMProvider

    is_healable = chaos_mode.startswith("healable_")
    if is_healable:
        chaos_mode = chaos_mode.replace("healable_", "")

    rates = {
        "timeout": 0.0,
        "server_error": 0.0,
        "rate_limit": 0.0,
        "malformed_json": 0.0,
        "empty_response": 0.0,
        "truncated_json": 0.0,
    }

    if chaos_mode in rates:
        rates[chaos_mode] = 1.0
    else:
        # "true" / random mixed mode — inject a mix of faults on every call
        # but heal after 3 calls so Shield can eventually win
        rates["timeout"] = 0.20
        rates["server_error"] = 0.20
        rates["rate_limit"] = 0.15
        rates["malformed_json"] = 0.20
        rates["empty_response"] = 0.10
        rates["truncated_json"] = 0.05
        is_healable = True  # Always healable in mixed mode — Shield will win eventually

    chaos_config = ChaosConfig(
        timeout_rate=rates["timeout"],
        server_error_rate=rates["server_error"],
        rate_limit_rate=rates["rate_limit"],
        malformed_json_rate=rates["malformed_json"],
        empty_response_rate=rates["empty_response"],
        truncated_json_rate=rates["truncated_json"],
        healable=is_healable,
        heal_after_calls=3,
    )
    chaos_provider = ChaosProvider(base=LiteLLMProvider(), config=chaos_config)

    cfg = ShieldConfig()
    # Chaos mode needs a bigger retry/repair budget to survive multiple injected faults
    cfg = cfg.merge_request_overrides(
        max_retries=5,
        max_repairs=3,
    )
    if request.config:
        cfg = cfg.merge_request_overrides(
            max_retries=request.config.max_retries,
            max_repairs=request.config.max_repairs,
            timeout_seconds=request.config.timeout_seconds,
            fallback_models=request.config.fallback_models,
        )

    logger.warning(
        "[CHAOS] Building chaos Shield for this request — "
        "mode=%s | NOT FOR PRODUCTION",
        chaos_mode,
    )
    return Shield(config=cfg, provider=chaos_provider)


@router.post(
    "/execute",
    response_model=ShieldResponse,
    summary="Execute LLM Call with Shield",
    description=(
        "Run an LLM call through the full Shield control loop: "
        "Validate → Retry → Repair → Fallback → Human Escalation. "
        "Always returns a response with either a valid result or an escalation packet. "
        "Never raises — all errors are captured in the execution trace.\n\n"
        "**Chaos Testing Mode** (opt-in, double-locked):\n"
        "Add header `X-Chaos-Mode: <fault_type>` **and** set `SHIELD_ALLOW_CHAOS=true` in "
        "your `.env` to activate fault injection for this request only. "
        "Both conditions must be present — one alone does nothing."
    ),
)
async def execute(
    request: ShieldRequest,
    shield: Shield = Depends(get_shield),
    x_chaos_mode: str | None = Header(default=None, alias="X-Chaos-Mode"),
) -> ShieldResponse:
    """
    Execute an LLM call through the Shield control loop.

    Normal mode (default):
        ChaosProvider is never instantiated. Zero overhead.

    Chaos mode (explicit opt-in):
        Requires BOTH X-Chaos-Mode header AND SHIELD_ALLOW_CHAOS=true env var.
        A fresh Shield with ChaosProvider is created for this request only.
        The production singleton Shield is untouched.
    """
    # ── Double-lock chaos gate ──────────────────────────────────────────────
    # Lock 1: caller must explicitly set the header
    chaos_val = x_chaos_mode.lower() if x_chaos_mode else "false"
    header_present = chaos_val != "false"
    # Lock 2: operator must explicitly allow chaos in the server environment
    env_allowed = os.getenv("SHIELD_ALLOW_CHAOS", "false").lower() == "true"

    if header_present and env_allowed:
        # Both locks open — use a fresh chaos Shield for this request only
        chaos_shield = _build_chaos_shield(request, chaos_val)
        return await chaos_shield.execute(request)

    if header_present and not env_allowed:
        logger.warning(
            "X-Chaos-Mode header received but SHIELD_ALLOW_CHAOS is not set — "
            "running in NORMAL mode. Set SHIELD_ALLOW_CHAOS=true in .env to enable."
        )

    # Normal execution — zero chaos code in the call path
    return await shield.execute(request)
