"""
LLM Shield — Production-ready library for reliable LLM calls.

Wraps any LLM call in a deterministic control loop:
  Validate → Retry → Repair → Fallback → Human Escalation

Quick Start:
    from llm_shield import Shield, ShieldRequest

    shield = Shield()
    response = await shield.execute(ShieldRequest(
        prompt="Generate a JSON user profile",
        response_schema={"type": "object", "required": ["name", "age"]},
    ))
    print(response.status)  # "SUCCEEDED" or "FAILED"
"""

from llm_shield.config import ShieldConfig
from llm_shield.engine import Shield
from llm_shield.escalation import (
    EscalationHandler,
    InMemoryEscalation,
    LogEscalation,
    WebhookEscalation,
)
from llm_shield.exceptions import (
    EscalationError,
    FallbackExhaustedError,
    LLMCallError,
    RepairFailedError,
    ShieldError,
    TemplateError,
    ValidationFailedError,
)
from llm_shield.models import (
    EscalationPacket,
    ExecutionTrace,
    ShieldRequest,
    ShieldResponse,
    ValidationErrorDetail,
)
from llm_shield.providers import LiteLLMProvider, LLMProvider, MockProvider
from llm_shield.repair import ChainedRepairStrategy, LLMRepairStrategy, RegexRepairStrategy

# Chaos testing — imported explicitly. NOT auto-imported in production.
# Usage: from llm_shield.chaos import ChaosConfig, ChaosProvider
from llm_shield.states import ExecutionState
from llm_shield.validators import (
    CompositeValidator,
    CustomValidator,
    JsonSchemaValidator,
    LengthValidator,
    RegexValidator,
    Validator,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Shield",
    "ShieldConfig",
    "ShieldRequest",
    "ShieldResponse",
    # Models
    "ExecutionTrace",
    "EscalationPacket",
    "ValidationErrorDetail",
    "ExecutionState",
    # Providers
    "LLMProvider",
    "LiteLLMProvider",
    "MockProvider",
    # Chaos (opt-in only — import explicitly from llm_shield.chaos)
    # "ChaosConfig",
    # "ChaosProvider",
    # Validators
    "Validator",
    "JsonSchemaValidator",
    "RegexValidator",
    "LengthValidator",
    "CustomValidator",
    "CompositeValidator",
    # Repair
    "ChainedRepairStrategy",
    "LLMRepairStrategy",
    "RegexRepairStrategy",
    # Escalation
    "EscalationHandler",
    "InMemoryEscalation",
    "WebhookEscalation",
    "LogEscalation",
    # Exceptions
    "ShieldError",
    "LLMCallError",
    "ValidationFailedError",
    "RepairFailedError",
    "FallbackExhaustedError",
    "EscalationError",
    "TemplateError",
]
