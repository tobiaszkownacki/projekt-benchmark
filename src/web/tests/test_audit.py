from starlette.requests import Request

from app.audit import client_ip, is_page_view


def request_for(
    *,
    path: str = "/",
    method: str = "GET",
    accept: str = "text/html",
    client: tuple[str, int] = ("172.22.0.1", 12345),
    forwarded_for: str | None = None,
) -> Request:
    headers = [(b"accept", accept.encode())]
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "query_string": b"",
            "headers": headers,
            "client": client,
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


def test_client_ip_uses_forwarded_address_from_trusted_proxy():
    request = request_for(client=("127.0.0.1", 12345), forwarded_for="198.51.100.42, 127.0.0.1")

    assert client_ip(request) == "198.51.100.42"


def test_client_ip_does_not_trust_forwarding_from_direct_client():
    request = request_for(client=("198.51.100.9", 12345), forwarded_for="203.0.113.7")

    assert client_ip(request) == "198.51.100.9"


def test_client_ip_does_not_count_a_proxy_without_forwarded_address():
    assert client_ip(request_for(client=("127.0.0.1", 12345))) is None


def test_page_view_requires_html_document_and_excludes_internal_traffic():
    assert is_page_view(request_for()) is True
    assert is_page_view(request_for(path="/api/auth/me")) is False
    assert is_page_view(request_for(path="/assets/app.js", accept="*/*")) is False
    assert is_page_view(request_for(path="/healthz")) is False
    assert is_page_view(request_for(accept="application/json")) is False
    assert is_page_view(request_for(method="POST")) is False
