"""Synchronous connection pool shared by the Streamlit app and the auth layer.

The pool is a module-level singleton rather than a ``st.cache_resource``: the
semantics are the same (one pool per process) but the data layer no longer
depends on Streamlit, which is what makes ``auth/repository.py`` reusable from
the control plane.
"""

import atexit
import threading
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg
from psycopg_pool import ConnectionPool

from core.config import get_database_url

_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=get_database_url(),
                    min_size=1,
                    max_size=10,
                    open=True,
                )
                atexit.register(close_pool)
    return _pool


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    with get_pool().connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
