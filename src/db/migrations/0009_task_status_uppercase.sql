-- Adopt the uppercase task_status vocabulary and the SUBMITTED state.
--
-- SUBMITTED separates "handed to the cluster's scheduler" from "occupying a
-- node"; both were 'running', so a job queued for an hour and a job burning an
-- hour of compute were indistinguishable.
--
-- Written as a type swap rather than RENAME VALUE plus ADD VALUE: ADD VALUE
-- cannot run inside a transaction block before PostgreSQL 12 and the runner
-- wraps every migration in one, and the swap places SUBMITTED in sort order
-- instead of appending it after FAILED.

-- Both triggers name task_status in UPDATE OF, which makes them depend on the
-- column, and PostgreSQL refuses to retype a column a trigger depends on. They
-- are recreated below verbatim from 0007; the functions they call are untouched.
DROP TRIGGER IF EXISTS tasks_notify ON tasks;
DROP TRIGGER IF EXISTS tasks_transition ON tasks;

ALTER TYPE task_status RENAME TO task_status_lowercase;

CREATE TYPE task_status AS ENUM ('PENDING', 'SUBMITTED', 'RUNNING', 'COMPLETED', 'FAILED');

ALTER TABLE tasks ALTER COLUMN task_status DROP DEFAULT;
ALTER TABLE tasks
    ALTER COLUMN task_status TYPE task_status
    USING UPPER(task_status::text)::task_status;
ALTER TABLE tasks ALTER COLUMN task_status SET DEFAULT 'PENDING';

DROP TYPE task_status_lowercase;

CREATE TRIGGER tasks_notify
    AFTER INSERT OR UPDATE OF task_status, artifact_status, executor_task_id ON tasks
    FOR EACH ROW EXECUTE FUNCTION notify_task_change();

CREATE TRIGGER tasks_transition
    AFTER INSERT OR UPDATE OF task_status, artifact_status ON tasks
    FOR EACH ROW EXECUTE FUNCTION record_task_transition();

-- The transition log stores the same labels as free text and the interface
-- renders them verbatim, so leaving it alone would spell one state two ways.
UPDATE task_state_transitions
   SET from_status = UPPER(from_status)
 WHERE from_status IN ('pending', 'running', 'completed', 'failed');

UPDATE task_state_transitions
   SET to_status = UPPER(to_status)
 WHERE to_status IN ('pending', 'running', 'completed', 'failed');
