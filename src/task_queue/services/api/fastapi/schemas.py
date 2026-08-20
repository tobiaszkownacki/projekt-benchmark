from datetime import datetime
from typing import Optional
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
    dataset: Optional[str]
    run_name: Optional[str]
    optimizer_params: dict
    completed_at: Optional[datetime]
    error_message: Optional[str]
    executor_task_id: Optional[str]
