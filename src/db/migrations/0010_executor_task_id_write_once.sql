-- executor_task_id is the only link between a row in tasks and a job on the
-- cluster, and delivery out of the outbox is at-least-once. A message consumed
-- twice submitted a second sbatch and overwrote the identifier of the first,
-- which left a job burning grant hours that nothing in this system tracks.
--
-- The index stops two tasks from claiming the same job id. It does not stop a
-- task's id from being replaced by a new one; that is the trigger added in
-- 0014. Creating the index fails on a database that already holds duplicates
-- (the demo seed once produced them); resolve those rows first. They are
-- listed by SELECT executor_task_id, count(*) FROM tasks WHERE executor_task_id
-- IS NOT NULL GROUP BY executor_task_id HAVING count(*) > 1.
CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_executor_task_id
    ON tasks (executor_task_id)
    WHERE executor_task_id IS NOT NULL;
