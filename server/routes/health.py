"""
Health check endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_shield.config import ShieldConfig
from server.dependencies import get_config

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health Check",
    description="Check server status, configuration, and provider connectivity.",
)
async def health_check(config: ShieldConfig = Depends(get_config)):
    return {
        "status": "healthy",
        "version": "0.1.0",
        "config": {
            "default_model": config.default_model,
            "fallback_models": config.fallback_models,
            "max_retries": config.max_retries,
            "max_repairs": config.max_repairs,
            "timeout_seconds": config.timeout_seconds,
            "escalation_mode": config.escalation_mode,
        },
    }
