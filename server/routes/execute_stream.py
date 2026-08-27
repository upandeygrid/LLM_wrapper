"""
Streaming execution endpoint — GET /execute-stream.

Uses Server-Sent Events (SSE) to stream each state machine transition
live to the frontend as they happen, instead of returning the full trace
at the end.

EventSource must use GET, so prompt + schema are passed as query params.

Event types:
    event: state  — fired on every state transition
    event: result — fired once at the end with the full ShieldResponse JSON
    event: error  — fired if an unexpected exception occurs
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from llm_shield.config import ShieldConfig
from llm_shield.engine import Shield
from llm_shield.models import ShieldRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Execution (Streaming)"])


@router.get("/execute-stream")
async def execute_stream(prompt: str, schema: str | None = None):
    """
    Stream Shield state machine transitions live via Server-Sent Events.

    Args:
        prompt: The user prompt to run through the Shield control loop.
        schema: Optional JSON Schema string. If provided, Shield will validate
                the LLM response against it.

    Event types emitted:
        - `state`  -> {"state": "CALLING_LLM", "reason": "starting execution"}
        - `result` -> full ShieldResponse JSON (final event)
        - `error`  -> {"message": "..."} (only on unexpected failure)
    """
    # Parse schema if provided
    response_schema = None
    if schema:
        try:
            response_schema = json.loads(schema)
        except json.JSONDecodeError:
            async def error_gen():
                yield {"event": "error", "data": json.dumps({"message": "Invalid JSON schema"})}
            return EventSourceResponse(error_gen())

    # Use asyncio.Queue to bridge the sync callback with the async SSE generator
    queue: asyncio.Queue = asyncio.Queue()

    def on_transition(event: dict) -> None:
        """Called synchronously by the engine on every state transition."""
        try:
            queue.put_nowait(event)
        except Exception:
            pass

    async def run_shield() -> None:
        """Run the Shield engine in the background, feeding events into the queue."""
        try:
            shield = Shield(
                config=ShieldConfig(),
                event_callback=on_transition,
            )
            request = ShieldRequest(
                prompt=prompt,
                response_schema=response_schema,
            )
            response = await shield.execute(request)
            queue.put_nowait({"__result__": response.model_dump()})
        except Exception as e:
            logger.error("Streaming execute failed: %s", e)
            queue.put_nowait({"__error__": str(e)})

    async def event_generator():
        """Async generator that yields SSE events as they arrive from the queue."""
        task = asyncio.create_task(run_shield())
        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=60.0)
                if "__result__" in item:
                    yield {"event": "result", "data": json.dumps(item["__result__"])}
                    break
                elif "__error__" in item:
                    yield {"event": "error", "data": json.dumps({"message": item["__error__"]})}
                    break
                else:
                    yield {"event": "state", "data": json.dumps(item)}
        except asyncio.TimeoutError:
            yield {"event": "error", "data": json.dumps({"message": "Request timed out after 60 seconds"})}
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(event_generator())
