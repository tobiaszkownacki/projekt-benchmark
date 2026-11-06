-- A job on the cluster reports its own end, and the only thing it can prove is
-- knowledge of a secret handed to it at submission time. The plaintext lives in
-- the queue message and in the batch script on the cluster's scratch; only the
-- digest lives here, as with api_tokens.
--
-- Its own table rather than a column on tasks: a secret has a lifecycle of its
-- own -- expiry, a use count, revocation -- and half the web layer reads tasks,
-- some of it with SELECT *.
CREATE TABLE IF NOT EXISTS task_webhook_tokens (
    task_id      UUID PRIMARY KEY REFERENCES tasks (task_id) ON DELETE CASCADE,
    token_sha256 CHAR(64) NOT NULL UNIQUE,
    prefix       TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ NOT NULL,
    uses         INTEGER NOT NULL DEFAULT 0,
    last_used_at TIMESTAMPTZ
);

-- The exit status of the run, as the callback reports it. A column rather than
-- a phrase inside error_message, so "how many runs died on OOM" is a query.
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS exit_code SMALLINT;
