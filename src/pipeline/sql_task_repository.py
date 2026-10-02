from collections.abc import Iterator
from contextlib import contextmanager

from pipeline.task_repository import TaskRepository, TaskStatus
from shared.connectors.base import DatabaseConnector


class SqlTaskRepository(TaskRepository):
    def __init__(self, db_connector_cls: type[DatabaseConnector]):
        self._db_cls = db_connector_cls

    @contextmanager
    def reserve_submission(self, task_id: str) -> Iterator[bool]:
        """Holds the task for one submitter until the block exits; True if it is still PENDING with no job.

        A transaction-scoped advisory lock rather than a status written to the
        row: it ends with the transaction, so an exception, a killed worker or a
        dropped connection releases it, and no task can be left reserved but
        never submitted. A second submitter waits for the first and then finds
        the job id it recorded.
        """
        with self._db_cls() as db:
            db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"submit:{task_id}",))
            row = db.execute(
                "SELECT 1 FROM tasks WHERE task_id = %s AND task_status = 'PENDING' AND executor_task_id IS NULL",
                (task_id,),
            ).fetchone()
            yield row is not None

    def mark_submitted(self, task_id: str, executor_task_id: str) -> bool:
        """Records the cluster's job id, once. False means someone got there first."""
        with self._db_cls() as db:
            cursor = db.execute(
                "UPDATE tasks SET executor_task_id = %s, task_status = 'SUBMITTED', updated_at = NOW() "
                "WHERE task_id = %s AND executor_task_id IS NULL",
                (executor_task_id, task_id),
            )
            return cursor.rowcount > 0

    def mark_failed(self, task_id: str, error_message: str) -> None:
        with self._db_cls() as db:
            db.execute(
                "UPDATE tasks SET task_status = 'FAILED', updated_at = NOW(), error_message = %s WHERE task_id = %s",
                (error_message, task_id),
            )

    def set_error(self, task_id: str, error_message: str) -> None:
        with self._db_cls() as db:
            db.execute(
                "UPDATE tasks SET error_message = %s, updated_at = NOW() WHERE task_id = %s",
                (error_message, task_id),
            )

    def mark_completed_by_executor_id(self, executor_task_id: str) -> bool:
        with self._db_cls() as db:
            cursor = db.execute(
                "UPDATE tasks SET task_status = 'COMPLETED', updated_at = NOW(), completed_at = NOW() "
                "WHERE executor_task_id = %s AND task_status != 'COMPLETED'",
                (executor_task_id,),
            )
            return cursor.rowcount > 0

    def get_by_executor_id(self, executor_task_id: str) -> TaskStatus | None:
        return self._fetch(
            "SELECT task_id, task_status, executor_task_id FROM tasks WHERE executor_task_id = %s",
            executor_task_id,
        )

    def get_by_task_id(self, task_id: str) -> TaskStatus | None:
        return self._fetch(
            "SELECT task_id, task_status, executor_task_id FROM tasks WHERE task_id = %s",
            task_id,
        )

    def _fetch(self, query: str, value: str) -> TaskStatus | None:
        with self._db_cls() as db:
            row = db.execute(query, (value,)).fetchone()
        if row is None:
            return None
        task_id, task_status, executor_task_id = row
        return TaskStatus(
            task_id=str(task_id),
            task_status=str(task_status),
            executor_task_id=str(executor_task_id) if executor_task_id else None,
        )
