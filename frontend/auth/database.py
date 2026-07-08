from contextlib import contextmanager
from typing import Generator

import psycopg
import streamlit as st
from psycopg_pool import ConnectionPool

from auth.config import get_database_url


@st.cache_resource(on_release=lambda pool: pool.close())
def get_pool() -> ConnectionPool:
    return ConnectionPool(
        conninfo=get_database_url(),
        min_size=1,
        max_size=10,
        open=True,
    )


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    with get_pool().connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
