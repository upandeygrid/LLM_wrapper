"""
Configuration management for LLM Shield.

Priority: defaults → environment variables → per-request overrides.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env file into os.environ so providers (e.g. litellm looking for GROQ_API_KEY) can access them
load_dotenv()


class ShieldConfig(BaseSettings):
    """Central configuration for the Shield engine.

    Values are loaded from environment variables prefixed with SHIELD_.
    Per-request overrides are merged at runtime by the engine.
    """

    model_config = {"env_prefix": "SHIELD_", "env_file": ".env", "extra": "ignore"}

    # --- LLM Defaults ---
    default_model: str = Field(
        default="groq/openai/gpt-oss-20b",
        description="Default model identifier for primary LLM calls.",
    )
    fallback_models: list[str] = Field(
        default=["groq/groq/compound-mini", "groq/qwen/qwen3.6-27b"],
        description="Ordered list of fallback models to try when the primary model fails.",
    )

    from pydantic import field_validator

    @field_validator("fallback_models", mode="before")
    @classmethod
    def parse_fallback_models(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    # --- Retry / Repair Limits ---
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts.")
    max_repairs: int = Field(default=2, ge=0, le=5, description="Max repair attempts.")
    timeout_seconds: float = Field(
        default=30.0, gt=0, le=120, description="Per-call timeout in seconds."
    )

    # --- Escalation ---
    escalation_mode: str = Field(
        default="in_memory",
        description="Escalation mode: 'in_memory' (Postman-accessible) or 'webhook'.",
    )
    escalation_webhook_url: str | None = Field(
        default=None,
        description="Webhook URL for escalation (only used if mode is 'webhook').",
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Server bind host.")
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port.")
    log_level: str = Field(default="info", description="Logging level.")

    def merge_request_overrides(
        self,
        max_retries: int | None = None,
        max_repairs: int | None = None,
        timeout_seconds: float | None = None,
        fallback_models: list[str] | None = None,
    ) -> "ShieldConfig":
        """Create a new config with per-request overrides applied."""
        overrides = {}
        if max_retries is not None:
            overrides["max_retries"] = max_retries
        if max_repairs is not None:
            overrides["max_repairs"] = max_repairs
        if timeout_seconds is not None:
            overrides["timeout_seconds"] = timeout_seconds
        if fallback_models is not None:
            overrides["fallback_models"] = fallback_models

        if not overrides:
            return self

        data = self.model_dump()
        data.update(overrides)
        return ShieldConfig(**data)
