"""Runtime configuration, read once from the environment."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def find_source_root() -> Path:
    """Locate the directory that contains ``src/benchmark_core``.

    The package sits at ``src/backend/`` in the repository and at ``/app/backend/``
    in the image, so a fixed number of parent hops is wrong in one of the two.
    ``SOURCE_ROOT`` overrides the search.
    """
    override = os.environ.get("SOURCE_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "benchmark_core").is_dir():
            return parent
    return here.parents[-1]


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))
    artifact_root: Path = field(default_factory=lambda: Path(os.environ.get("ARTIFACT_ROOT", "/downloads")))
    static_root: Path = field(default_factory=lambda: Path(os.environ.get("STATIC_ROOT", "/app/static")))
    migrations_dir: Path = field(default_factory=lambda: Path(os.environ.get("MIGRATIONS_DIR", "/app/migrations")))
    session_secret: str = field(default_factory=lambda: os.environ.get("SESSION_SECRET", ""))
    run_migrations: bool = field(default_factory=lambda: _bool("RUN_MIGRATIONS", True))
    session_cookie: str = "benchmark_session"
    # Scopes the session cookie to the deployment prefix. At the origin root
    # this is "/" and nothing changes; behind a prefix it stops the cookie from
    # being sent to unrelated applications sharing the same host.
    session_cookie_path: str = field(default_factory=lambda: "/" + os.environ.get("ROOT_PATH", "").strip("/"))
    session_max_age: int = field(default_factory=lambda: _int("SESSION_MAX_AGE", 60 * 60 * 12))
    secure_cookies: bool = field(default_factory=lambda: _bool("SECURE_COOKIES", False))

    # Read in exactly one place (services/authz.can_read_run), so switching
    # between public and owner-only results is a configuration change rather
    # than an audit of every endpoint.
    public_results: bool = field(default_factory=lambda: _bool("PUBLIC_RESULTS", True))

    daily_submission_limit: int = field(default_factory=lambda: _int("DAILY_SUBMISSION_LIMIT", 3))

    preview_limit_bytes: int = field(default_factory=lambda: _int("PREVIEW_LIMIT_BYTES", 2 * 1024 * 1024))
    archive_limit_bytes: int = field(default_factory=lambda: _int("ARCHIVE_LIMIT_BYTES", 512 * 1024 * 1024))
    archive_max_entries: int = field(default_factory=lambda: _int("ARCHIVE_MAX_ENTRIES", 5000))

    validator_enabled: bool = field(default_factory=lambda: _bool("VALIDATOR_ENABLED", True))
    validator_image: str = field(default_factory=lambda: os.environ.get("VALIDATOR_IMAGE", "python:3.12-slim"))
    validator_timeout: int = field(default_factory=lambda: _int("VALIDATOR_TIMEOUT", 30))

    rabbitmq_management_url: str = field(default_factory=lambda: os.environ.get("RABBITMQ_MANAGEMENT_URL", ""))
    rabbitmq_user: str = field(default_factory=lambda: os.environ.get("RABBITMQ_USER", ""))
    rabbitmq_password: str = field(default_factory=lambda: os.environ.get("RABBITMQ_PASSWORD", ""))
    # Must match src/config/rabbitmq/definitions.json (and QueueTopology("athena")
    # in src/shared), which is the exchange/queue src/pipeline consumes.
    worker_queue: str = field(default_factory=lambda: os.environ.get("ATHENA_WORKER_QUEUE", "athena_worker_queue"))
    main_exchange: str = field(default_factory=lambda: os.environ.get("MAIN_EXCHANGE", "main_exchange"))

    google_client_id: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLIENT_ID", ""))
    google_client_secret: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLIENT_SECRET", ""))
    microsoft_client_id: str = field(default_factory=lambda: os.environ.get("MICROSOFT_CLIENT_ID", ""))
    microsoft_client_secret: str = field(default_factory=lambda: os.environ.get("MICROSOFT_CLIENT_SECRET", ""))
    public_base_url: str = field(default_factory=lambda: os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000"))


settings = Settings()
