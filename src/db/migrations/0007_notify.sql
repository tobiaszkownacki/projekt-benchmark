-- Status changes pushed to the browser: trigger -> pg_notify -> LISTEN -> SSE.
--
-- The payload carries identifiers and state only. pg_notify caps payloads at
-- roughly 8000 bytes, and a notification that grows with the data it describes
-- works right up until the day it does not; the client refetches the detail.

CREATE OR REPLACE FUNCTION notify_task_change() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('task_changed', json_build_object(
        'task_id',      NEW.task_id,
        'status',       NEW.task_status,
        'artifact',     NEW.artifact_status,
        'submitted_by', NEW.submitted_by
    )::text);
    RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tasks_notify ON tasks;
CREATE TRIGGER tasks_notify
    AFTER INSERT OR UPDATE OF task_status, artifact_status, executor_task_id ON tasks
    FOR EACH ROW EXECUTE FUNCTION notify_task_change();

-- Record transitions automatically, so no service can forget to.
CREATE OR REPLACE FUNCTION record_task_transition() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO task_state_transitions (task_id, from_status, to_status, actor, detail)
        VALUES (NEW.task_id, NULL, NEW.task_status::text, 'api',
                jsonb_build_object('run_name', NEW.run_name));
    ELSIF NEW.task_status IS DISTINCT FROM OLD.task_status
       OR NEW.artifact_status IS DISTINCT FROM OLD.artifact_status THEN
        INSERT INTO task_state_transitions (task_id, from_status, to_status, actor, detail)
        VALUES (NEW.task_id, OLD.task_status::text, NEW.task_status::text, 'system',
                jsonb_build_object(
                    'artifact_status',  NEW.artifact_status,
                    'executor_task_id', NEW.executor_task_id,
                    'error_message',    NEW.error_message));
    END IF;
    RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tasks_transition ON tasks;
CREATE TRIGGER tasks_transition
    AFTER INSERT OR UPDATE OF task_status, artifact_status ON tasks
    FOR EACH ROW EXECUTE FUNCTION record_task_transition();
