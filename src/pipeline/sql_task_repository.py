from pipeline.task_repository import TaskRepository, TaskStatus
from shared.interfaces.database_connector import DatabaseConnector


class SqlTaskRepository(TaskRepository):
    def __init__(self, db_connector_cls: type[DatabaseConnector]):
        self._db_cls = db_connector_cls
    def mark_submitted(self, task_id: str, executor_task_id: str) -> None:
        with self._db_cls() as db:
            db.execute(
                "UPDATE tasks SET executor_task_id = %s, task_status = 'running', updated_at = NOW() "
                "WHERE task_id = %s",
                (executor_task_id, task_id),
            )
    def mark_failed(self, task_id: str, error_message: str) -> None:
        with self._db_cls() as db:
            db.execute(
                "UPDATE tasks SET task_status = 'failed', updated_at = NOW(), error_message = %s WHERE task_id = %s",
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
                "UPDATE tasks SET task_status = 'completed', updated_at = NOW(), completed_at = NOW() "
                "WHERE executor_task_id = %s AND task_status != 'completed'",
                (executor_task_id,),
            )
            return cursor.rowcount > 0
    def get_by_executor_id(self, executor_task_id: str) -> TaskStatus | None:
        with self._db_cls() as db:
            cursor = db.execute(
                "SELECT task_id, task_status FROM tasks WHERE executor_task_id = %s",
                (executor_task_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        task_id, task_status = row
        return TaskStatus(task_id=str(task_id), task_status=str(task_status))
