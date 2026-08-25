"""
Escalation management endpoints — human review queue accessible via Postman.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from llm_shield.escalation import InMemoryEscalation
from llm_shield.models import EscalationPacket
from server.dependencies import get_escalation_handler

router = APIRouter(prefix="/escalations", tags=["Escalations"])


class ResolveRequest(BaseModel):
    notes: str | None = None


@router.get(
    "",
    response_model=list[EscalationPacket],
    summary="List All Escalations",
    description="Get all escalations (pending and resolved) from the in-memory queue.",
)
async def list_escalations(
    status: str | None = None,
    handler: InMemoryEscalation = Depends(get_escalation_handler),
) -> list[EscalationPacket]:
    if status == "pending":
        return handler.list_pending()
    return handler.list_all()


@router.get(
    "/stats",
    summary="Escalation Statistics",
    description="Get counts of total and pending escalations.",
)
async def escalation_stats(
    handler: InMemoryEscalation = Depends(get_escalation_handler),
):
    return {
        "total": handler.count,
        "pending": handler.pending_count,
        "resolved": handler.count - handler.pending_count,
    }


@router.get(
    "/{escalation_id}",
    response_model=EscalationPacket,
    summary="Get Escalation Details",
    description=(
        "Get full details of a specific escalation, including the original request, "
        "all model responses, validation errors, and recovery attempts."
    ),
)
async def get_escalation(
    escalation_id: str,
    handler: InMemoryEscalation = Depends(get_escalation_handler),
) -> EscalationPacket:
    packet = handler.get(escalation_id)
    if packet is None:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found")
    return packet


@router.post(
    "/{escalation_id}/resolve",
    response_model=EscalationPacket,
    summary="Resolve Escalation",
    description="Mark an escalation as resolved with optional notes.",
)
async def resolve_escalation(
    escalation_id: str,
    body: ResolveRequest,
    handler: InMemoryEscalation = Depends(get_escalation_handler),
) -> EscalationPacket:
    packet = handler.resolve(escalation_id, notes=body.notes)
    if packet is None:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found")
    return packet
