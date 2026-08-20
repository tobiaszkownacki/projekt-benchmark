"""Server-sent events carrying task state changes.

SSE rather than WebSocket because the traffic is one-directional, it works over
ordinary HTTP, and the browser reconnects on its own.

The chain is poller -> UPDATE tasks -> trigger -> pg_notify -> LISTEN -> here.
The status in the interface changes in the same second the poller wrote it,
without anything polling the API.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app import db
from app.security import CurrentUser, optional_user
from app.services.authz import can_read_run

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["events"])

HEARTBEAT_SECONDS = 15


@router.get("/events")
async def events(
    request: Request, user: Optional[CurrentUser] = Depends(optional_user)
) -> StreamingResponse:
    queue = await db.broker.subscribe()

    async def stream():
        try:
            yield b": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    # Without a heartbeat an idle connection gets closed by
                    # whatever sits in front of the application.
                    yield b": ping\n\n"
                    continue

                # Filtered here, using the same policy the REST endpoints use.
                # Two copies of an authorisation rule is one copy too many.
                owner = payload.get("submitted_by")
                if owner and not can_read_run(user, owner):
                    continue

                body = json.dumps(payload, default=str)
                yield f"event: task_changed\ndata: {body}\n\n".encode()
        finally:
            await db.broker.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # nginx buffers streamed responses by default, which turns live
            # status into status that arrives in batches when the buffer fills.
            "X-Accel-Buffering": "no",
        },
    )
