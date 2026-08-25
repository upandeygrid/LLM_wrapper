"""
Escalation handler tests.
"""

import pytest

from llm_shield.escalation import InMemoryEscalation, create_escalation_handler
from llm_shield.models import EscalationPacket, EscalationStatus


def _make_packet(**overrides) -> EscalationPacket:
    defaults = {
        "reason": "test escalation",
        "original_prompt": "Generate something",
        "original_model": "groq/test",
    }
    defaults.update(overrides)
    return EscalationPacket(**defaults)


class TestInMemoryEscalation:
    @pytest.mark.asyncio
    async def test_escalate_stores_packet(self):
        handler = InMemoryEscalation()
        packet = _make_packet()

        result = await handler.escalate(packet)

        assert result is True
        assert handler.count == 1
        assert handler.pending_count == 1

    @pytest.mark.asyncio
    async def test_list_pending(self):
        handler = InMemoryEscalation()
        p1 = _make_packet(reason="first")
        p2 = _make_packet(reason="second")

        await handler.escalate(p1)
        await handler.escalate(p2)

        pending = handler.list_pending()
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_resolve(self):
        handler = InMemoryEscalation()
        packet = _make_packet()
        await handler.escalate(packet)

        resolved = handler.resolve(packet.id, notes="Fixed manually")

        assert resolved is not None
        assert resolved.status == EscalationStatus.RESOLVED
        assert resolved.resolution_notes == "Fixed manually"
        assert handler.pending_count == 0

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        handler = InMemoryEscalation()
        packet = _make_packet()
        await handler.escalate(packet)

        retrieved = handler.get(packet.id)
        assert retrieved is not None
        assert retrieved.reason == "test escalation"

    def test_get_nonexistent(self):
        handler = InMemoryEscalation()
        assert handler.get("nonexistent") is None

    def test_resolve_nonexistent(self):
        handler = InMemoryEscalation()
        assert handler.resolve("nonexistent") is None


class TestCreateEscalationHandler:
    def test_in_memory(self):
        handler = create_escalation_handler("in_memory")
        assert isinstance(handler, InMemoryEscalation)

    def test_webhook_requires_url(self):
        with pytest.raises(ValueError, match="webhook_url"):
            create_escalation_handler("webhook")

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_escalation_handler("unknown")
