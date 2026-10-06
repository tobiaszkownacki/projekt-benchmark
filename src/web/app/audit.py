"""Minimal, private audit trail for public visits and authentication.

The application records only events that are useful to the operator: a page
view or an authentication attempt. Request bodies, cookies and credentials are
never written. The real client address is accepted only when the immediate
peer is a configured reverse proxy; direct requests cannot spoof the forwarded
headers.
"""

import ipaddress
import logging
from typing import Literal
from uuid import UUID

from fastapi import Request

from app import db
from app.settings import settings

logger = logging.getLogger(__name__)

AuditEvent = Literal["page_view", "login_success", "login_failure"]


def _trusted_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    networks = []
    for value in settings.trusted_proxy_cidrs:
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid trusted proxy network %r", value)
    return tuple(networks)


_TRUSTED_NETWORKS = _trusted_networks()


def _parse_ip(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def _is_trusted_proxy(value: str | None) -> bool:
    parsed = _parse_ip(value)
    return parsed is not None and any(ipaddress.ip_address(parsed) in network for network in _TRUSTED_NETWORKS)


def client_ip(request: Request) -> str | None:
    """Return the public client address, trusting forwarding headers safely."""
    peer = request.client.host if request.client else None
    if _is_trusted_proxy(peer):
        forwarded = request.headers.get("x-forwarded-for", "")
        for candidate in forwarded.split(","):
            parsed = _parse_ip(candidate)
            if parsed:
                return parsed
        parsed = _parse_ip(request.headers.get("x-real-ip"))
        if parsed:
            return parsed
    return _parse_ip(peer)


def is_page_view(request: Request) -> bool:
    """Count browser document loads, excluding API, health and asset traffic."""
    if request.method != "GET" or request.url.path.startswith(("/api/", "/assets/")):
        return False
    if request.url.path == "/healthz":
        return False
    return "text/html" in request.headers.get("accept", "").lower()


async def record_event(
    event_type: AuditEvent,
    request: Request,
    *,
    user_id: UUID | None = None,
    status_code: int | None = None,
) -> None:
    """Persist one event without making the user request fail if audit storage is down."""
    try:
        await db.execute(
            """
            INSERT INTO web_audit_events
                (event_type, user_id, client_ip, path, method, status_code, user_agent)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                event_type,
                user_id,
                client_ip(request),
                request.url.path[:512],
                request.method[:16],
                status_code,
                request.headers.get("user-agent", "")[:512],
            ),
        )
    except Exception:
        # Observability must never turn a successful login or page load into a
        # 500. The exception remains visible to the operator in web logs.
        logger.exception("Could not record web audit event %s", event_type)


async def record_page_view(request: Request, status_code: int) -> None:
    if is_page_view(request) and status_code < 400:
        await record_event("page_view", request, status_code=status_code)
