"""Configuration lookup for the frontend and the control plane.

Values are resolved from the environment first and from Streamlit secrets only
as a fallback. Keeping that order lets the same modules run inside the Streamlit
app (where ``secrets.toml`` is the natural home for configuration) and inside the
FastAPI control plane (which is configured by the environment and must not
depend on Streamlit being installed).

``streamlit`` is imported lazily, inside the fallback path, for exactly that
reason: importing this module must not require Streamlit.
"""

import os
from typing import Any, Optional


def _secret(section: str, key: str) -> Optional[Any]:
    """Read ``[section] key`` from Streamlit secrets, or None when unavailable."""
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        return st.secrets[section][key]
    except Exception:
        return None


def _lookup(env_var: str, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
    value = os.environ.get(env_var)
    if value not in (None, ""):
        return value
    value = _secret(section, key)
    if value not in (None, ""):
        return value
    return default


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    host = _lookup("POSTGRES_HOST", "database", "host", "localhost")
    port = _lookup("POSTGRES_PORT", "database", "port", "5432")
    dbname = _lookup("POSTGRES_DB", "database", "dbname")
    user = _lookup("POSTGRES_USER", "database", "user")
    password = _lookup("POSTGRES_PASSWORD", "database", "password")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_rabbitmq_connection_params():
    import pika

    host = _lookup("RABBITMQ_HOST", "rabbitmq", "host", "localhost")
    port = int(_lookup("RABBITMQ_PORT", "rabbitmq", "port", 5672))
    user = _lookup("RABBITMQ_USER", "rabbitmq", "user")
    password = _lookup("RABBITMQ_PASSWORD", "rabbitmq", "password")

    credentials = pika.PlainCredentials(user, password)
    return pika.ConnectionParameters(host=host, port=port, credentials=credentials)


def get_recaptcha_site_key() -> str:
    return _lookup("RECAPTCHA_SITE_KEY", "recaptcha", "site_key", "") or ""


def get_recaptcha_secret_key() -> str:
    return _lookup("RECAPTCHA_SECRET_KEY", "recaptcha", "secret_key", "") or ""


def get_recaptcha_min_score() -> float:
    return float(_lookup("RECAPTCHA_MIN_SCORE", "recaptcha", "min_score", 0.5))
