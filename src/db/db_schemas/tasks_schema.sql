CREATE TYPE task_status AS ENUM ('pending', 'running', 'completed', 'failed');

CREATE TABLE tasks (
    task_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_name          VARCHAR(255) NOT NULL,
    executor_name         TEXT NOT NULL,
    submitted_by        UUID NOT NULL REFERENCES users (id),
    task_status         task_status NOT NULL DEFAULT 'pending',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dataset             TEXT,
    run_name            TEXT,
    optimizer_params    JSONB NOT NULL DEFAULT '{}'::jsonb,
    completed_at        TIMESTAMPTZ,
    error_message       TEXT,
    executor_task_id    TEXT
);
