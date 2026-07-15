CREATE TYPE task_status AS ENUM (
    'PENDING_APPROVAL',
    'REJECTED',
    'QUEUED',
    'RUNNING',
    'COMPLETED',
    'FAILED'
);

CREATE TABLE tasks (
    id              SERIAL PRIMARY KEY,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    user_email      VARCHAR(255) NOT NULL,
    status          task_status NOT NULL DEFAULT 'PENDING_APPROVAL',
    queue_name      VARCHAR(50) NOT NULL DEFAULT 'default',
    priority        INTEGER NOT NULL DEFAULT 100,
    backend         VARCHAR(20) NOT NULL DEFAULT 'athena',
    config          JSONB NOT NULL,
    workdir         TEXT,
    slurm_job_id    VARCHAR(64),
    exit_code       INTEGER,
    result          JSONB,
    error_message   TEXT,
    reviewed_by     UUID REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

CREATE INDEX idx_tasks_status ON tasks (status);
