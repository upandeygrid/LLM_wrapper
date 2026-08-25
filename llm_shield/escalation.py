"""
Human escalation handlers.

When all automated recovery (retry, repair, fallback, template) fails,
the system escalates to a human with full context.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from llm_shield.exceptions import EscalationError
from llm_shield.models import EscalationPacket, EscalationStatus

logger = logging.getLogger(__name__)


class EscalationHandler:
    """Abstract base for escalation handlers."""

    async def escalate(self, packet: EscalationPacket) -> bool:
        """Send an escalation. Returns True if successfully delivered."""
        raise NotImplementedError


class InMemoryEscalation(EscalationHandler):
    """Stores escalations in an in-memory queue for Postman retrieval.

    This is the default mode — escalations are accessible via the /escalations API.
    """

    def __init__(self):
        self._queue: dict[str, EscalationPacket] = {}

    async def escalate(self, packet: EscalationPacket) -> bool:
        """Store escalation in memory."""
        try:
            self._queue[packet.id] = packet
            logger.warning(
                "ESCALATION [%s]: %s — prompt: %s",
                packet.id, packet.reason, packet.original_prompt[:100],
            )
            return True
        except Exception as e:
            raise EscalationError(
                f"In-memory escalation failed: {e}",
                channel="in_memory",
            )

    def list_pending(self) -> list[EscalationPacket]:
        """List all pending escalations."""
        return [
            p for p in self._queue.values()
            if p.status == EscalationStatus.PENDING
        ]

    def list_all(self) -> list[EscalationPacket]:
        """List all escalations."""
        return list(self._queue.values())

    def get(self, escalation_id: str) -> EscalationPacket | None:
        """Get a specific escalation by ID."""
        return self._queue.get(escalation_id)

    def resolve(self, escalation_id: str, notes: str | None = None) -> EscalationPacket | None:
        """Mark an escalation as resolved."""
        from datetime import datetime, timezone

        packet = self._queue.get(escalation_id)
        if packet is None:
            return None

        packet.status = EscalationStatus.RESOLVED
        packet.resolved_at = datetime.now(timezone.utc)
        packet.resolution_notes = notes
        return packet

    @property
    def count(self) -> int:
        return len(self._queue)

    @property
    def pending_count(self) -> int:
        return sum(
            1 for p in self._queue.values()
            if p.status == EscalationStatus.PENDING
        )


class WebhookEscalation(EscalationHandler):
    """Sends escalation context to a webhook URL via HTTP POST."""

    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self._url = webhook_url
        self._timeout = timeout

    async def escalate(self, packet: EscalationPacket) -> bool:
        """POST the escalation packet to the webhook URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._url,
                    json=packet.model_dump(mode="json"),
                    timeout=self._timeout,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                logger.info(
                    "Escalation [%s] delivered to webhook: %s (status=%d)",
                    packet.id, self._url, response.status_code,
                )
                return True
        except httpx.TimeoutException:
            raise EscalationError(
                f"Webhook timeout after {self._timeout}s: {self._url}",
                channel="webhook",
                details={"url": self._url},
            )
        except httpx.HTTPStatusError as e:
            raise EscalationError(
                f"Webhook returned {e.response.status_code}: {self._url}",
                channel="webhook",
                details={"url": self._url, "status_code": e.response.status_code},
            )
        except Exception as e:
            raise EscalationError(
                f"Webhook delivery failed: {e}",
                channel="webhook",
                details={"url": self._url, "error": str(e)},
            )


class LogEscalation(EscalationHandler):
    """Logs escalations to structured logging (for development)."""

    async def escalate(self, packet: EscalationPacket) -> bool:
        logger.critical(
            "HUMAN ESCALATION REQUIRED\n"
            "  ID: %s\n"
            "  Reason: %s\n"
            "  Prompt: %s\n"
            "  Errors: %d validation errors\n"
            "  Attempts: %d model responses, %d recovery attempts\n",
            packet.id,
            packet.reason,
            packet.original_prompt[:200],
            len(packet.validation_errors),
            len(packet.model_responses),
            len(packet.recovery_attempts),
        )
        return True


def create_escalation_handler(
    mode: str = "in_memory",
    webhook_url: str | None = None,
) -> EscalationHandler:
    """Factory function for creating escalation handlers."""
    if mode == "in_memory":
        return InMemoryEscalation()
    elif mode == "webhook":
        if not webhook_url:
            raise ValueError("webhook_url is required when escalation_mode is 'webhook'")
        return WebhookEscalation(webhook_url)
    elif mode == "log":
        return LogEscalation()
    else:
        raise ValueError(f"Unknown escalation mode: {mode}")
