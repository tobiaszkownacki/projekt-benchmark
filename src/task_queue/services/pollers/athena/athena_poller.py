import logging
import os
import sys
from collections import Counter

from task_queue.services.shared.interfaces.abstract_poller import Poller
from task_queue.services.pollers.logging_config import configure_logging

from task_queue.services.shared.connectors.postgres_connector import PostGresConnector
from task_queue.services.shared.connectors.athena_connector import AthenaConnector
from task_queue.services.shared.connectors.rabbitmq_connector import RabbitMQConnector

configure_logging()
logger = logging.getLogger(__name__)

_FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

SLURM_PARTITION = "plgrid-gpu-a100"
ATHENA_REMOTE_PATH = os.environ.get("ATHENA_REMOTE_PATH")
PROJECT_DIR = f"{ATHENA_REMOTE_PATH}/projekt-benchmark"


class AthenaPoller(Poller):

    def is_ready(self, athena_conn: AthenaConnector) -> bool:
        partition_state, _, _ = athena_conn.ssh_capture(f"sinfo -p {SLURM_PARTITION} -h -o '%a'")
        if partition_state.strip() != "up":
            logger.warning(f"Partition {SLURM_PARTITION} is not up (state={partition_state!r})")
            return False

        assoc, _, _ = athena_conn.ssh_capture(
            f"sacctmgr show assoc user={athena_conn.user} account={athena_conn.account} format=Account -n"
        )
        if not assoc.strip():
            logger.warning(f"No Slurm association for user={athena_conn.user} account={athena_conn.account}")
            return False

        _, _, exit_code = athena_conn.ssh_capture(f"mkdir -p {PROJECT_DIR}/reports && test -w {PROJECT_DIR}/reports")
        if exit_code != 0:
            logger.warning(f"{PROJECT_DIR}/reports is not writable")
            return False

        return True

    def _parse_slurm_output(self, slurm_output: str) -> list[dict]:
        lines = [line for line in slurm_output.strip().splitlines() if line]
        if not lines:
            return []

        header, *rows = lines
        columns = header.split("|")
        return [dict(zip(columns, row.split("|"))) for row in rows]

    def _fetch_error_tail(self, athena_conn: AthenaConnector, job_name: str, job_id: str, lines: int = 30) -> str:
        remote_out = f"{PROJECT_DIR}/reports/{job_name}/{job_id}.out"
        return athena_conn.ssh(f"tail -n {lines} {remote_out} 2>/dev/null || true")

    def _update_tasks_status_db(self, athena_conn: AthenaConnector, jobs: list[dict]) -> list[str]:
        newly_completed_task_ids: list[str] = []

        with PostGresConnector() as db_conn:
            for job in jobs:
                state = job.get("State", "")
                job_id = job.get("JobID")
                job_name = job.get("JobName")

                if state == "COMPLETED":
                    row = db_conn.execute(
                        "UPDATE tasks SET task_status = %s, updated_at = NOW(), completed_at = NOW() "
                        "WHERE executor_task_id = %s AND task_status != 'completed' "
                        "RETURNING task_id",
                        ("completed", job_id),
                    ).fetchone()
                    if row:
                        newly_completed_task_ids.append(str(row[0]))
                elif state in _FAILURE_STATES:
                    row = db_conn.execute(
                        "SELECT task_id, task_status FROM tasks WHERE executor_task_id = %s",
                        (job_id,),
                    ).fetchone()
                    if row is None:
                        logger.warning(f"job_id={job_id} is {state} but has no matching task in the database")
                        continue

                    task_id, task_status = row
                    if task_status == "failed":
                        continue

                    error_tail = self._fetch_error_tail(athena_conn, job_name, job_id)
                    db_conn.execute(
                        "UPDATE tasks SET task_status = %s, updated_at = NOW(), error_message = %s WHERE task_id = %s",
                        ("failed", error_tail, task_id),
                    )
                    logger.error(f"task_id={task_id} failed (job_id={job_id}, state={state}):\n{error_tail}")

        return newly_completed_task_ids

    def check_jobs_status(self, athena_conn: AthenaConnector) -> list[dict]:
        cmd = [
            "sacct",
            "--parsable2",
            "--allocations",
            "--format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,End",
        ]
        raw_output = athena_conn.ssh(" ".join(cmd))
        return self._parse_slurm_output(raw_output)

    def publish_finished_taks2downloader(self, list_of_tasks2download: list[str]) -> None:
        with RabbitMQConnector(exchange="main-exchange", routing_key="ATHENA_DOWNLOADER_QUEUE") as publisher:
            for task_id in list_of_tasks2download:
                try:
                    publisher.publish({"task_id": task_id})
                    logger.info(f"Task with task_id={task_id} finished and was sent to the download queue")
                except Exception:
                    logger.exception(f"Failed to publish task_id={task_id} to downloader queue")

    def run_poll_cycle(self) -> None:
        with AthenaConnector() as athena_conn:
            if not self.is_ready(athena_conn):
                logger.info("Athena is not ready to run tasks, skipping this poll cycle")
                return

            jobs = self.check_jobs_status(athena_conn)
            status_counts = Counter(job.get("State", "UNKNOWN") for job in jobs)
            breakdown = ", ".join(f"{state}={count}" for state, count in sorted(status_counts.items()))
            logger.info(f"Found {len(jobs)} jobs ({breakdown})")

            finished_task_ids = self._update_tasks_status_db(athena_conn, jobs)

        if finished_task_ids:
            self.publish_finished_taks2downloader(finished_task_ids)


if __name__=="__main__":
    logger.info("Started polling")
    poller = AthenaPoller(executor_name="Athena")
    try:
        poller.run_poll_cycle()
    except Exception:
        logger.exception("Polling failed. Retrying in 60 seconds")
        sys.exit(1)
