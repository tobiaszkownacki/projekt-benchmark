import time
import uuid

import pytest

from shared.connectors.rabbitmq import RabbitMQConnector
from shared.queue_topology import QueueTopology


@pytest.mark.integration
def test_task_has_status_running_after_sending_to_db(db_connector, test_user_id):

    topology = QueueTopology("fake")

    task_id = str(uuid.uuid4())

    db_connector.execute(
        "INSERT INTO tasks (task_id, queue_name, executor_name, submitted_by, dataset, run_name) "
        "VALUES (%s, %s, 'fake', %s, 'wine quality', 'smoke-test')",
        (task_id, topology.worker_queue, test_user_id),
    )

    with RabbitMQConnector(exchange=topology.main_exchange, routing_key=topology.worker_queue) as publisher:
        publisher.publish(
            {
                "task_id": task_id,
                "dataset": "test_dataset",
                "optimizer": "adam",
                "run_name": "test_run",
            }
        )

    max_time = time.monotonic() + 5
    row = None
    while time.monotonic() < max_time:
        row = db_connector.execute(
            "SELECT task_status, executor_task_id FROM tasks WHERE task_id = %s", (task_id,)
        ).fetchone()
        if row and row[0] == "SUBMITTED":
            break
        time.sleep(0.2)

    assert row is not None
    assert row[0] == "SUBMITTED"
    assert row[1] is not None
