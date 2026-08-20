-- Columns tasks needs before a run page or a leaderboard can be built.
--
-- seed / runner_version / gpu_model address the reproducibility risk in §18:
-- cuDNN is non-deterministic on A100 by default, so a result without its seed
-- and versions cannot be defended on a competition leaderboard.
--
-- queued_at / started_at exist so /runs can show time spent waiting, which §11.2
-- asks for and which cannot be derived from created_at alone.

DO $$ BEGIN
    CREATE TYPE benchmark_suite_t AS ENUM ('test', 'final');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE artifact_status_t AS ENUM ('absent', 'downloading', 'ready', 'empty', 'error');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

ALTER TABLE tasks
    ADD COLUMN IF NOT EXISTS submission_id   UUID REFERENCES submissions (submission_id),
    ADD COLUMN IF NOT EXISTS seed            BIGINT,
    ADD COLUMN IF NOT EXISTS suite           benchmark_suite_t NOT NULL DEFAULT 'test',
    ADD COLUMN IF NOT EXISTS model_name      TEXT,
    ADD COLUMN IF NOT EXISTS optimizer_name  TEXT,
    ADD COLUMN IF NOT EXISTS family          optimizer_family_t,
    ADD COLUMN IF NOT EXISTS stop_condition  JSONB,
    ADD COLUMN IF NOT EXISTS artifact_root   TEXT,
    ADD COLUMN IF NOT EXISTS artifact_status artifact_status_t NOT NULL DEFAULT 'absent',
    ADD COLUMN IF NOT EXISTS artifact_bytes  BIGINT,
    ADD COLUMN IF NOT EXISTS artifact_files  INTEGER,
    ADD COLUMN IF NOT EXISTS queued_at       TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS started_at      TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS runner_version  TEXT,
    ADD COLUMN IF NOT EXISTS gpu_model       TEXT;

CREATE INDEX IF NOT EXISTS idx_tasks_submitter ON tasks (submitted_by, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_status    ON tasks (task_status);
CREATE INDEX IF NOT EXISTS idx_tasks_suite     ON tasks (suite, dataset, model_name);
