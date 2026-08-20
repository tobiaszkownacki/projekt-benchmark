-- Transactional outbox for queue publication.
--
-- Writing the task and publishing to RabbitMQ are two operations with no shared
-- transaction. If the broker dies between them the submission exists but never
-- runs; if the process dies after publishing but before committing, a job runs
-- that no row describes. Inserting the message in the same transaction as the
-- task removes both cases.
--
-- It also removes the async trap in §15: a separate drain process publishes
-- with ordinary blocking pika, outside any event loop, so the question of
-- pika's async safety never arises. And the API keeps no broker credentials.

CREATE TABLE IF NOT EXISTS queue_outbox (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    exchange     TEXT NOT NULL,
    routing_key  TEXT NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT
);

CREATE INDEX IF NOT EXISTS idx_outbox_unpublished
    ON queue_outbox (id) WHERE published_at IS NULL;
