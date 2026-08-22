import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import FastAPI, HTTPException

from task_queue.services.api.logging_config import configure_logging
from task_queue.services.api.fastapi import repository
from task_queue.services.api.fastapi.schemas import TaskCreateRequest, TaskResponse
from shared.connectors.rabbitmq_connector import RabbitMQConnector

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/tasks", response_model=TaskResponse, status_code=201)
def submit_task(payload: TaskCreateRequest) -> dict:
    try:
        task = repository.create_task(
            dataset=payload.dataset,
            run_name=payload.run_name,
            optimizers=payload.optimizers,
            submitted_by=payload.submitted_by,
        )
    except Exception:
        logger.exception("Failed to create task in the database.")
        raise HTTPException(status_code=500, detail="Failed to create task.")

    message = {
        "task_id": str(task["task_id"]),
        "run_name": task["run_name"],
        "dataset": task["dataset"],
        "optimizer": ",".join(payload.optimizers),
        "submitted_by": str(task["submitted_by"]),
        "created_at": task["created_at"].isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": task["task_status"],
    }
    try:
        with RabbitMQConnector(exchange="main-exchange", routing_key=repository.QUEUE_NAME) as publisher:
            publisher.publish(message)
    except Exception:
        logger.exception(f"Task {task['task_id']} was created but failed to publish to {repository.QUEUE_NAME}")
        raise HTTPException(status_code=502, detail="Task created but failed to enqueue it for execution.")

    return task
