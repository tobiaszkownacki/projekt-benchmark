-- executor_task_id is the only link between a task and its job on the
-- cluster. Once it is set, replacing it would orphan a running job that
-- nothing in this system tracks any more, so the database refuses that write.
-- Setting it for the first time and writing the same value again stay allowed.
CREATE OR REPLACE FUNCTION tasks_executor_task_id_immutable() RETURNS trigger AS $$
BEGIN
    IF OLD.executor_task_id IS NOT NULL
       AND NEW.executor_task_id IS DISTINCT FROM OLD.executor_task_id THEN
        RAISE EXCEPTION 'executor_task_id of task % is already %, refusing to change it to %',
            OLD.task_id, OLD.executor_task_id, NEW.executor_task_id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_tasks_executor_task_id_immutable ON tasks;
CREATE TRIGGER trg_tasks_executor_task_id_immutable
    BEFORE UPDATE OF executor_task_id ON tasks
    FOR EACH ROW EXECUTE FUNCTION tasks_executor_task_id_immutable();
