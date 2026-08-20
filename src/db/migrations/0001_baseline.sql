-- Baseline: the schema as it exists in src/db/db_schemas/.
--
-- Written idempotently on purpose. A fresh volume gets these objects from
-- docker-entrypoint-initdb.d; an existing volume never runs that directory at
-- all. Making the baseline a no-op when the objects are already present lets the
-- migration runner be the single source of truth in both cases.

DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('unverified', 'verified', 'admin');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE auth_provider AS ENUM ('email', 'google', 'microsoft');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE task_status AS ENUM ('pending', 'running', 'completed', 'failed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS users (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email                   VARCHAR(255) NOT NULL UNIQUE,
    password_hash           VARCHAR(255),
    role                    user_role NOT NULL DEFAULT 'unverified',
    auth_provider           auth_provider NOT NULL,
    oauth_sub               VARCHAR(255),
    display_name            VARCHAR(255),
    associated_organisation VARCHAR(255),
    associated_org_email    VARCHAR(255),
    join_reason             TEXT,
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at           TIMESTAMPTZ,

    CONSTRAINT uq_oauth_identity UNIQUE (auth_provider, oauth_sub),
    CONSTRAINT chk_email_password CHECK (
        auth_provider != 'email' OR password_hash IS NOT NULL
    )
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_role  ON users (role);

CREATE TABLE IF NOT EXISTS tasks (
    task_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_name       VARCHAR(255) NOT NULL,
    executor_name    TEXT NOT NULL,
    submitted_by     UUID NOT NULL REFERENCES users (id),
    task_status      task_status NOT NULL DEFAULT 'pending',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dataset          TEXT,
    run_name         TEXT,
    optimizer_params JSONB NOT NULL DEFAULT '{}'::jsonb,
    completed_at     TIMESTAMPTZ,
    error_message    TEXT,
    executor_task_id TEXT
);
