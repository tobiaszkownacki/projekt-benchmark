-- Every state change, with who caused it.
--
-- §18 names the alternative: answering "why has my run been stuck for two
-- hours" by reading five services' logs. This table is the cheap version of
-- that answer, and it is only cheap if it exists from the start.

CREATE TABLE IF NOT EXISTS task_state_transitions (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id     UUID NOT NULL REFERENCES tasks (task_id) ON DELETE CASCADE,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    actor       TEXT NOT NULL,
    detail      JSONB,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transitions_task
    ON task_state_transitions (task_id, occurred_at);
