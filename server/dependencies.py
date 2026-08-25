"""
FastAPI dependency injection for the Shield engine and shared state.
"""

from __future__ import annotations

from functools import lru_cache

from llm_shield.config import ShieldConfig
from llm_shield.engine import Shield
from llm_shield.escalation import InMemoryEscalation, create_escalation_handler


@lru_cache
def get_config() -> ShieldConfig:
    """Get the global ShieldConfig (loaded once from env / .env)."""
    return ShieldConfig()


# Singleton instances — shared across all requests
_shield: Shield | None = None
_escalation_handler: InMemoryEscalation | None = None


def get_escalation_handler() -> InMemoryEscalation:
    """Get the shared in-memory escalation handler."""
    global _escalation_handler
    if _escalation_handler is None:
        _escalation_handler = InMemoryEscalation()
    return _escalation_handler


def get_shield() -> Shield:
    """Get the shared Shield engine instance."""
    global _shield
    if _shield is None:
        config = get_config()
        handler = get_escalation_handler()
        _shield = Shield(config=config, escalation_handler=handler)
    return _shield


def reset_singletons() -> None:
    """Reset singletons (useful for testing)."""
    global _shield, _escalation_handler
    _shield = None
    _escalation_handler = None
    get_config.cache_clear()
