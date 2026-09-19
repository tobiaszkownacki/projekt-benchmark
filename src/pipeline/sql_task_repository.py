from pipeline.task_repository import TaskRepository, TaskStatus
from shared.connectors.base import DatabaseConnector
from shared.run_result import RunResult


class SqlTaskRepository(TaskRepository):
    def __init__(self, db_connector_cls: type[DatabaseConnector]):
        self._db_cls = db_connector_cls

    def mark_submitted(self, task_id: str, executor_task_id: str) -> bool:
        """Records the cluster's job id, once. False means someone got there first."""
        with self._db_cls() as db:
            cursor = db.execute(
                "UPDATE tasks SET executor_task_id = %s, task_status = 'SUBMITTED', updated_at = NOW() "
                "WHERE task_id = %s AND executor_task_id IS NULL",
                (executor_task_id, task_id),
            )
            return cursor.rowcount > 0

    def mark_running_by_executor_id(self, executor_task_id: str) -> bool:
        """SUBMITTED means queued at the scheduler, RUNNING means burning compute."""
        with self._db_cls() as db:
            cursor = db.execute(
                "UPDATE tasks SET task_status = 'RUNNING', started_at = COALESCE(started_at, NOW()), "
                "updated_at = NOW() "
                "WHERE executor_task_id = %s AND task_status = 'SUBMITTED'",
                (executor_task_id,),
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
                "UPDATE tasks SET task_status = 'COMPLETED', artifact_status = 'downloading', "
                "updated_at = NOW(), completed_at = NOW() "
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

    def store_result(self, task_id: str, result: RunResult) -> None:
        """Idempotent on purpose: delivery to the downloader is at-least-once."""
        series = result.series
        with self._db_cls() as db:
            db.execute(
                "INSERT INTO results (task_id, final_loss, final_accuracy, gradient_count, database_reaches, "
                "total_steps, total_epochs, wall_time_seconds, stop_reason) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (task_id) DO NOTHING",
                (
                    task_id,
                    result.final_loss,
                    result.final_accuracy,
                    result.gradient_count,
                    result.database_reaches,
                    result.total_steps,
                    result.total_epochs,
                    result.wall_time_seconds,
                    result.stop_reason,
                ),
            )
            db.execute(
                "INSERT INTO result_series (task_id, epochs, loss, accuracy, gradient_count, database_reaches, "
                "wall_time_seconds) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (task_id) DO NOTHING",
                (
                    task_id,
                    series.epochs,
                    series.loss,
                    series.accuracy,
                    series.gradient_count,
                    series.database_reaches,
                    series.wall_time_seconds,
                ),
            )

    def mark_artifacts(self, task_id: str, files: int, total_bytes: int) -> None:
        with self._db_cls() as db:
            db.execute(
                "UPDATE tasks SET artifact_status = %s, artifact_files = %s, artifact_bytes = %s, "
                "updated_at = NOW() WHERE task_id = %s",
                ("ready" if files else "empty", files, total_bytes, task_id),
            )
