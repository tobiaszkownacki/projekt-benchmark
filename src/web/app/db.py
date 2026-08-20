"""Async connection pool plus the dedicated LISTEN connection used by SSE."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.settings import settings

logger = logging.getLogger(__name__)

_pool: Optional[AsyncConnectionPool] = None


async def open_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            open=False,
            kwargs={"row_factory": dict_row},
        )
        await _pool.open(wait=True, timeout=30)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("Connection pool is not open")
    return _pool


@asynccontextmanager
async def connection() -> AsyncIterator[psycopg.AsyncConnection]:
    async with pool().connection() as conn:
        yield conn


async def fetch_all(sql: str, params: Sequence[Any] = ()) -> list[dict]:
    async with connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


async def fetch_one(sql: str, params: Sequence[Any] = ()) -> Optional[dict]:
    async with connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchone()


async def execute(sql: str, params: Sequence[Any] = ()) -> None:
    async with connection() as conn:
        await conn.execute(sql, params)


class TaskChangeBroker:
    """Fans one Postgres LISTEN connection out to every open SSE stream.

    The connection is deliberately outside the pool. A pooled connection is
    returned after the request that borrowed it, and the subscription goes with
    it -- the failure mode being a stream that stays open and silently stops
    delivering.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._listen_forever())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def _publish(self, payload: dict) -> None:
        async with self._lock:
            targets = list(self._subscribers)
        for queue in targets:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # A client too slow to drain its own queue loses events rather
                # than stalling the listener for everybody else.
                logger.warning("Dropping task change for a saturated subscriber")

    async def _listen_forever(self) -> None:
        while True:
            try:
                conn = await psycopg.AsyncConnection.connect(
                    settings.database_url, autocommit=True
                )
            except Exception as exc:
                logger.warning("LISTEN connection failed (%s); retrying in 5s", exc)
                await asyncio.sleep(5)
                continue
            try:
                await conn.execute("LISTEN task_changed")
                logger.info("Listening for task_changed notifications")
                async for notify in conn.notifies():
                    try:
                        await self._publish(json.loads(notify.payload))
                    except json.JSONDecodeError:
                        logger.warning("Unparseable notification payload")
            except asyncio.CancelledError:
                await conn.close()
                raise
            except Exception as exc:
                logger.warning("LISTEN connection lost (%s); reconnecting", exc)
            finally:
                try:
                    await conn.close()
                except Exception:
                    pass
            await asyncio.sleep(1)


broker = TaskChangeBroker()
