-- Scalars and convergence series for a finished run.
--
-- stop_reason mirrors the StopReason enum actually emitted by BenchmarkResult
-- (optimization_engine/runner.py:28-33). Note that metrics/stop_metrics.py
-- declares a second, differently-named StopReason; the two disagree. This
-- schema follows the one that reaches a result, and the divergence is reported
-- as a defect rather than silently reconciled here.

DO $$ BEGIN
    CREATE TYPE stop_reason_t AS ENUM (
        'GRADIENT_LIMIT', 'DATABASE_LIMIT', 'EPOCH_LIMIT',
        'OPTIMIZER_CONVERGED', 'MAX_STEPS'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS results (
    task_id           UUID PRIMARY KEY REFERENCES tasks (task_id) ON DELETE CASCADE,
    final_loss        DOUBLE PRECISION,
    final_accuracy    DOUBLE PRECISION,
    gradient_count    BIGINT NOT NULL,
    database_reaches  BIGINT NOT NULL,
    total_steps       BIGINT,
    total_epochs      INTEGER,
    wall_time_seconds DOUBLE PRECISION,
    stop_reason       stop_reason_t NOT NULL,
    recorded_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_results_loss ON results (final_loss);

-- Parallel arrays, one row per run.
--
-- A narrow (task, epoch, metric, value) table would allow percentile_cont in
-- SQL, but that is the wrong trade here: the X axis of every chart is budget,
-- not epoch (§13.2), and different optimizers spend wildly different budget per
-- epoch. Measurements are therefore not aligned across runs on the budget axis,
-- so quantiles require interpolating each run onto a common grid first --
-- something SQL will not do usefully. Since aggregation happens in Python
-- anyway, arrays cost one row instead of 5 x epochs and read in a single fetch.
CREATE TABLE IF NOT EXISTS result_series (
    task_id           UUID PRIMARY KEY REFERENCES tasks (task_id) ON DELETE CASCADE,
    epochs            INTEGER[]          NOT NULL,
    loss              DOUBLE PRECISION[] NOT NULL,
    accuracy          DOUBLE PRECISION[],
    gradient_count    BIGINT[]           NOT NULL,
    database_reaches  BIGINT[]           NOT NULL,
    wall_time_seconds DOUBLE PRECISION[],

    CONSTRAINT series_same_length CHECK (
        cardinality(loss)             = cardinality(epochs) AND
        cardinality(gradient_count)   = cardinality(epochs) AND
        cardinality(database_reaches) = cardinality(epochs)
    )
);
