-- executor_task_id is the only link between a row in tasks and a job on the
-- cluster, and delivery out of the outbox is at-least-once. A message consumed
-- twice submitted a second sbatch and overwrote the identifier of the first,
-- which left a job burning grant hours that nothing in this system tracks.
--
-- The index makes the second write fail rather than succeed quietly. Creation
-- fails on a database that already holds duplicates, which is the intended
-- outcome: they have to be resolved by hand before the rule can hold.
CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_executor_task_id
    ON tasks (executor_task_id)
    WHERE executor_task_id IS NOT NULL;
