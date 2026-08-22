from typing import Iterable
from uuid import UUID

from psycopg.types.json import Jsonb

from shared.connectors.postgres_connector import PostGresConnector

QUEUE_NAME = "ATHENA_WORKER_QUEUE"
EXECUTOR_NAME = "Athena"


def create_task(dataset: str, run_name: str, optimizers: Iterable[str], submitted_by: UUID) -> dict:
    optimizer_params = {"optimizers": list(optimizers)}
    with PostGresConnector() as db:
        cur = db.execute(
            """
            INSERT INTO tasks (queue_name, executor_name, submitted_by, dataset, run_name, optimizer_params)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (QUEUE_NAME, EXECUTOR_NAME, submitted_by, dataset, run_name, Jsonb(optimizer_params)),
        )
        columns = [col.name for col in cur.description]
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to create task")
        return dict(zip(columns, row))
