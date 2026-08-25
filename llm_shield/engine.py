"""
Core state machine engine — the heart of LLM Shield.

Orchestrates the full control loop:
  INIT → CALLING_LLM → VALIDATING → SUCCEEDED
                     ↘ RETRYING → REPAIRING → FALLING_BACK → TEMPLATE_FALLBACK → ESCALATING → FAILED

Guarantees:
- Every execution reaches exactly one terminal state (SUCCEEDED or FAILED)
- No infinite loops: fixed counters for retries, repairs, fallbacks
- No silent failures: every path is traced and errors are collected
- Full execution trace returned with every response
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from llm_shield.config import ShieldConfig
from llm_shield.escalation import EscalationHandler, InMemoryEscalation, create_escalation_handler
from llm_shield.exceptions import (
    EscalationError,
    FallbackExhaustedError,
    InvalidTransitionError,
    LLMCallError,
    ShieldError,
    TemplateError,
)
from llm_shield.fallback import FallbackChain, TemplateFallback
from llm_shield.models import (
    EscalationPacket,
    ExecutionTrace,
    ShieldRequest,
    ShieldResponse,
    ValidationErrorDetail,
)
from llm_shield.providers import LLMProvider, LiteLLMProvider
from llm_shield.repair import ChainedRepairStrategy, RepairStrategy
from llm_shield.states import VALID_TRANSITIONS, ExecutionState
from llm_shield.trace import ExecutionTracer
from llm_shield.validators import Validator, build_validators

logger = logging.getLogger(__name__)


class Shield:
    """The main orchestrator — wraps any LLM call in a reliable control loop.

    Usage:
        shield = Shield()
        response = await shield.execute(ShieldRequest(
            prompt="Generate a JSON user profile",
            response_schema={"type": "object", "required": ["name", "age"]},
        ))
        print(response.status)  # "SUCCEEDED" or "FAILED"
        print(response.execution_trace)  # Full trace of what happened
    """

    def __init__(
        self,
        config: ShieldConfig | None = None,
        provider: LLMProvider | None = None,
        repair_strategy: RepairStrategy | None = None,
        escalation_handler: EscalationHandler | None = None,
    ):
        self.config = config or ShieldConfig()
        self._provider = provider or LiteLLMProvider()
        self._repair_strategy = repair_strategy
        self._escalation_handler = escalation_handler or create_escalation_handler(
            mode=self.config.escalation_mode,
            webhook_url=self.config.escalation_webhook_url,
        )
        self._template_fallback = TemplateFallback()

    @property
    def escalation_handler(self) -> EscalationHandler:
        """Access the escalation handler (useful for the API layer)."""
        return self._escalation_handler

    async def execute(self, request: ShieldRequest) -> ShieldResponse:
        """Execute the full control loop. Always returns — never raises.

        This is the primary API. It takes a request, runs it through the
        state machine, and returns a response with either the result or
        an escalation packet.
        """
        # Merge per-request config overrides
        config = self.config.merge_request_overrides(
            max_retries=request.config.max_retries,
            max_repairs=request.config.max_repairs,
            timeout_seconds=request.config.timeout_seconds,
            fallback_models=request.config.fallback_models,
        )

        # Resolve model
        model = request.model or config.default_model

        # Build validators
        validator = build_validators(
            response_schema=request.response_schema,
            validator_configs=request.validators,
        )

        # Initialize repair strategy (lazy — only created if needed)
        repair = self._repair_strategy or ChainedRepairStrategy(
            self._provider, timeout=config.timeout_seconds
        )

        # Build system prompt for structured output
        system_prompt = request.system_prompt
        if request.response_schema and not system_prompt:
            system_prompt = (
                "You must respond with valid JSON that conforms to the requested schema. "
                "Do not include any explanation, markdown formatting, or extra text. "
                "Output ONLY the JSON object."
            )

        # Initialize execution context
        ctx = _ExecutionContext(
            request=request,
            config=config,
            model=model,
            validator=validator,
            repair=repair,
            provider=self._provider,
            escalation_handler=self._escalation_handler,
            template_fallback=self._template_fallback,
            system_prompt=system_prompt,
        )

        # Run the state machine
        return await self._run_state_machine(ctx)

    async def _run_state_machine(self, ctx: _ExecutionContext) -> ShieldResponse:
        """Execute the state machine loop until a terminal state is reached."""
        ctx.transition(ExecutionState.INIT, ExecutionState.CALLING_LLM, "starting execution")

        # Safety valve: absolute maximum iterations to prevent infinite loops.
        # Each fallback model can get its own retry + repair cycle, so we need to
        # account for: primary (call + validate + retries + repairs) +
        # per-fallback (fallback + call + validate + retries + repairs) +
        # template + escalation + generous buffer.
        num_models = 1 + len(ctx.config.fallback_models)
        per_model_steps = (
            2  # CALLING_LLM + VALIDATING
            + ctx.config.max_retries * 2  # RETRYING + CALLING_LLM per retry
            + ctx.config.max_repairs * 3  # REPAIRING + VALIDATING + possible retry
        )
        max_iterations = (
            num_models * per_model_steps
            + len(ctx.config.fallback_models) * 2  # FALLING_BACK transitions
            + 5  # TEMPLATE_FALLBACK + ESCALATING + buffer
        )

        iteration = 0
        while not ctx.state.is_terminal and iteration < max_iterations:
            iteration += 1

            if ctx.state == ExecutionState.CALLING_LLM:
                await self._handle_calling_llm(ctx)
            elif ctx.state == ExecutionState.VALIDATING:
                self._handle_validating(ctx)
            elif ctx.state == ExecutionState.RETRYING:
                self._handle_retrying(ctx)
            elif ctx.state == ExecutionState.REPAIRING:
                await self._handle_repairing(ctx)
            elif ctx.state == ExecutionState.FALLING_BACK:
                self._handle_falling_back(ctx)
            elif ctx.state == ExecutionState.TEMPLATE_FALLBACK:
                self._handle_template_fallback(ctx)
            elif ctx.state == ExecutionState.ESCALATING:
                await self._handle_escalating(ctx)

        # Safety: if we somehow didn't reach a terminal state, force a valid path to FAILED.
        # We cannot jump directly to FAILED from arbitrary states — we must go through
        # ESCALATING first (which is a valid terminal path from many recovery states).
        if not ctx.state.is_terminal:
            logger.error("State machine exceeded max iterations (%d)", max_iterations)
            ctx.force_terminal(
                f"Safety valve: exceeded {max_iterations} iterations"
            )

        return ctx.build_response()

    # ----- State handlers -----

    async def _handle_calling_llm(self, ctx: _ExecutionContext) -> None:
        """CALLING_LLM: Make the LLM API call."""
        try:
            import time
            start = time.monotonic()
            response = await ctx.provider.call(
                prompt=ctx.request.prompt,
                model=ctx.current_model,
                timeout=ctx.config.timeout_seconds,
                system_prompt=ctx.system_prompt,
            )
            duration = (time.monotonic() - start) * 1000

            ctx.last_response = response
            ctx.tracer.record_llm_call(
                model=ctx.current_model,
                attempt=ctx.total_llm_calls + 1,
                phase=ctx.current_phase,
                response=response[:500],  # Truncate for trace
                duration_ms=duration,
            )
            ctx.total_llm_calls += 1
            ctx.transition(
                ExecutionState.CALLING_LLM, ExecutionState.VALIDATING,
                "LLM response received",
            )

        except (LLMCallError, asyncio.TimeoutError) as e:
            ctx.tracer.record_llm_call(
                model=ctx.current_model,
                attempt=ctx.total_llm_calls + 1,
                phase=ctx.current_phase,
                error=str(e),
            )
            ctx.total_llm_calls += 1
            ctx.last_error = str(e)
            ctx.transition(
                ExecutionState.CALLING_LLM, ExecutionState.RETRYING,
                f"LLM call failed: {e}",
            )

    def _handle_validating(self, ctx: _ExecutionContext) -> None:
        """VALIDATING: Check the response against validators."""
        if ctx.validator is None:
            # No validators configured — accept the response as-is
            ctx.result = ctx.last_response
            ctx.transition(
                ExecutionState.VALIDATING, ExecutionState.SUCCEEDED,
                "No validators configured — accepting response",
            )
            return

        errors = ctx.validator.validate(ctx.last_response)

        if not errors:
            # Parse JSON if schema was provided
            if ctx.request.response_schema:
                import json
                from llm_shield.validators import _strip_markdown_fences
                try:
                    ctx.result = json.loads(_strip_markdown_fences(ctx.last_response))
                except json.JSONDecodeError:
                    ctx.result = ctx.last_response
            else:
                ctx.result = ctx.last_response

            ctx.transition(
                ExecutionState.VALIDATING, ExecutionState.SUCCEEDED,
                "All validators passed",
            )
            return

        # Validation failed
        ctx.tracer.record_validation_errors(errors)
        ctx.last_validation_errors = errors
        error_summary = "; ".join(e.message for e in errors[:3])

        if ctx.retry_count < ctx.config.max_retries:
            ctx.transition(
                ExecutionState.VALIDATING, ExecutionState.RETRYING,
                f"Validation failed ({len(errors)} errors): {error_summary}",
            )
        else:
            ctx.transition(
                ExecutionState.VALIDATING, ExecutionState.REPAIRING,
                f"Validation failed, retries exhausted ({ctx.retry_count}/{ctx.config.max_retries}): {error_summary}",
            )

    def _handle_retrying(self, ctx: _ExecutionContext) -> None:
        """RETRYING: Decide whether to retry or move to repair."""
        ctx.retry_count += 1
        ctx.tracer.record_recovery_attempt(
            type_="retry",
            attempt=ctx.retry_count,
            model=ctx.current_model,
            result="retrying",
        )

        if ctx.retry_count <= ctx.config.max_retries:
            ctx.current_phase = "retry"
            ctx.transition(
                ExecutionState.RETRYING, ExecutionState.CALLING_LLM,
                f"Retry attempt {ctx.retry_count}/{ctx.config.max_retries}",
            )
        else:
            ctx.transition(
                ExecutionState.RETRYING, ExecutionState.REPAIRING,
                f"Max retries reached ({ctx.config.max_retries})",
            )

    async def _handle_repairing(self, ctx: _ExecutionContext) -> None:
        """REPAIRING: Attempt to fix the bad response."""
        ctx.repair_count += 1

        if ctx.repair_count > ctx.config.max_repairs:
            ctx.tracer.record_recovery_attempt(
                type_="repair",
                attempt=ctx.repair_count,
                model=ctx.current_model,
                result="max_repairs_reached",
            )
            ctx.transition(
                ExecutionState.REPAIRING, ExecutionState.FALLING_BACK,
                f"Max repairs reached ({ctx.config.max_repairs})",
            )
            return

        try:
            repaired = await ctx.repair.repair(
                original_prompt=ctx.request.prompt,
                bad_response=ctx.last_response or "",
                errors=ctx.last_validation_errors,
                model=ctx.current_model,
            )

            ctx.last_response = repaired
            ctx.tracer.record_recovery_attempt(
                type_="repair",
                attempt=ctx.repair_count,
                model=ctx.current_model,
                result="attempted",
                detail=repaired[:200],
            )
            ctx.current_phase = "repair"
            ctx.transition(
                ExecutionState.REPAIRING, ExecutionState.VALIDATING,
                f"Repair attempt {ctx.repair_count}/{ctx.config.max_repairs}",
            )

        except (LLMCallError, ShieldError, asyncio.TimeoutError) as e:
            ctx.tracer.record_recovery_attempt(
                type_="repair",
                attempt=ctx.repair_count,
                model=ctx.current_model,
                result="error",
                detail=str(e),
            )
            ctx.transition(
                ExecutionState.REPAIRING, ExecutionState.FALLING_BACK,
                f"Repair failed: {e}",
            )

    def _handle_falling_back(self, ctx: _ExecutionContext) -> None:
        """FALLING_BACK: Try the next fallback model or move to template."""
        if ctx.fallback_chain is None:
            ctx.fallback_chain = FallbackChain(ctx.config.fallback_models)

        if ctx.fallback_chain.has_next:
            next_model = ctx.fallback_chain.current_model
            ctx.fallback_chain.advance()

            ctx.current_model = next_model
            ctx.current_phase = "fallback"
            ctx.retry_count = 0  # Reset retries for new model
            ctx.repair_count = 0  # Reset repairs for new model

            ctx.tracer.record_recovery_attempt(
                type_="fallback",
                attempt=ctx.fallback_chain._current_index,
                model=next_model,
                result="trying",
            )
            ctx.transition(
                ExecutionState.FALLING_BACK, ExecutionState.CALLING_LLM,
                f"Trying fallback model: {next_model}",
            )
        else:
            ctx.transition(
                ExecutionState.FALLING_BACK, ExecutionState.TEMPLATE_FALLBACK,
                f"All fallback models exhausted ({ctx.fallback_chain.total_models} tried)",
            )

    def _handle_template_fallback(self, ctx: _ExecutionContext) -> None:
        """TEMPLATE_FALLBACK: Use a predefined template response."""
        template = ctx.request.config.template_response

        try:
            template_response = ctx.template_fallback.get_response(
                template=template,
                prompt=ctx.request.prompt,
            )

            # Validate the template response if we have a validator
            if ctx.validator:
                errors = ctx.validator.validate(template_response)
                if errors:
                    ctx.tracer.record_recovery_attempt(
                        type_="template",
                        attempt=1,
                        result="invalid",
                        detail=f"Template failed validation: {errors[0].message}",
                    )
                    ctx.transition(
                        ExecutionState.TEMPLATE_FALLBACK, ExecutionState.ESCALATING,
                        f"Template response failed validation: {errors[0].message}",
                    )
                    return

            # Template is valid
            import json
            try:
                ctx.result = json.loads(template_response)
            except (json.JSONDecodeError, TypeError):
                ctx.result = template_response

            ctx.tracer.record_recovery_attempt(
                type_="template",
                attempt=1,
                result="success",
            )
            ctx.transition(
                ExecutionState.TEMPLATE_FALLBACK, ExecutionState.SUCCEEDED,
                "Template fallback succeeded",
            )

        except TemplateError as e:
            ctx.tracer.record_recovery_attempt(
                type_="template",
                attempt=1,
                result="error",
                detail=str(e),
            )
            ctx.transition(
                ExecutionState.TEMPLATE_FALLBACK, ExecutionState.ESCALATING,
                f"Template fallback failed: {e}",
            )

    async def _handle_escalating(self, ctx: _ExecutionContext) -> None:
        """ESCALATING: Send context to human review."""
        packet = EscalationPacket(
            reason="All automated recovery strategies exhausted",
            original_prompt=ctx.request.prompt,
            original_model=ctx.request.model or ctx.config.default_model,
            response_schema=ctx.request.response_schema,
            validation_errors=ctx.tracer.trace.validation_errors,
            model_responses=ctx.tracer.trace.llm_calls,
            recovery_attempts=ctx.tracer.trace.recovery_attempts,
            final_state=ctx.state.name,
        )

        try:
            await ctx.escalation_handler.escalate(packet)
            ctx.escalation_packet = packet
            ctx.transition(
                ExecutionState.ESCALATING, ExecutionState.FAILED,
                "Escalation delivered to human review",
            )
        except EscalationError as e:
            logger.error("Escalation itself failed: %s", e)
            packet.reason = f"Escalation delivery failed: {e}. Original: {packet.reason}"
            ctx.escalation_packet = packet
            ctx.transition(
                ExecutionState.ESCALATING, ExecutionState.FAILED,
                f"Escalation delivery failed: {e}",
            )


class _ExecutionContext:
    """Mutable execution context that holds all state during a single run."""

    def __init__(
        self,
        request: ShieldRequest,
        config: ShieldConfig,
        model: str,
        validator: Validator | None,
        repair: RepairStrategy,
        provider: LLMProvider,
        escalation_handler: EscalationHandler,
        template_fallback: TemplateFallback,
        system_prompt: str | None,
    ):
        self.request = request
        self.config = config
        self.current_model = model
        self.validator = validator
        self.repair = repair
        self.provider = provider
        self.escalation_handler = escalation_handler
        self.template_fallback = template_fallback
        self.system_prompt = system_prompt

        # State
        self.state = ExecutionState.INIT
        self.tracer = ExecutionTracer()

        # Counters
        self.retry_count = 0
        self.repair_count = 0
        self.total_llm_calls = 0
        self.current_phase = "primary"

        # Results
        self.last_response: str | None = None
        self.last_error: str | None = None
        self.last_validation_errors: list[ValidationErrorDetail] = []
        self.result: Any = None
        self.escalation_packet: EscalationPacket | None = None
        self.fallback_chain: FallbackChain | None = None

    def transition(self, from_state: ExecutionState, to_state: ExecutionState, reason: str) -> None:
        """Transition to a new state with validation."""
        # Validate transition
        valid_targets = VALID_TRANSITIONS.get(from_state, frozenset())
        if to_state not in valid_targets:
            raise InvalidTransitionError(from_state, to_state)

        self.tracer.record_transition(from_state, to_state, reason)
        self.state = to_state
        logger.debug("State: %s → %s (%s)", from_state.name, to_state.name, reason)

    def force_terminal(self, reason: str) -> None:
        """Force the state machine to FAILED — used only by the safety valve.

        This bypasses normal transition validation because we need to terminate
        from any arbitrary state when the iteration limit is hit.
        """
        prev = self.state
        self.tracer.record_transition(prev, ExecutionState.FAILED, f"FORCED: {reason}")
        self.state = ExecutionState.FAILED
        self.last_error = reason
        logger.error("Forced terminal: %s → FAILED (%s)", prev.name, reason)

    def build_response(self) -> ShieldResponse:
        """Build the final response from the execution context."""
        trace = self.tracer.finalize()
        trace.final_model = self.current_model

        if self.state == ExecutionState.SUCCEEDED:
            return ShieldResponse(
                status="SUCCEEDED",
                result=self.result,
                execution_trace=trace,
            )
        else:
            return ShieldResponse(
                status="FAILED",
                result=None,
                error=self.last_error or "All recovery strategies exhausted",
                escalation=self.escalation_packet,
                execution_trace=trace,
            )
