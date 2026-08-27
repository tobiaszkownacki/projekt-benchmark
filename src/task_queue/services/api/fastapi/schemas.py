from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    dataset: str
    optimizers: list[str] = Field(min_length=1)
    run_name: str
    submitted_by: UUID


class TaskResponse(BaseModel):
    task_id: UUID
    queue_name: str
    executor_name: str
    submitted_by: UUID
    task_status: str
    created_at: datetime
    updated_at: datetime
    dataset: str | None
    run_name: str | None
    optimizer_params: dict
    completed_at: datetime | None
    error_message: str | None
    executor_task_id: str | None
