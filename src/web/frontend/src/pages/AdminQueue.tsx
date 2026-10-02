import { Link } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { ErrorNote, Loading } from '../components/Empty';
import { duration, integer, shortId, timestamp } from '../components/format';
import type { User } from '../types';

interface Queue {
  name: string; messages: number; ready: number;
  unacknowledged: number; consumers: number; is_dlq: boolean;
}
interface OutboxRow {
  id: number; exchange: string; routing_key: string;
  payload: Record<string, unknown>; created_at: string;
  published_at: string | null; attempts: number; last_error: string | null;
}
interface SlurmJob {
  task_id: string; run_name: string | null; executor_task_id: string | null;
  task_status: string; created_at: string; queued_at: string | null;
  started_at: string | null; elapsed_seconds: number;
}
interface Orphan {
  task_id: string; run_name: string | null; task_status: string;
  executor_task_id: string | null; updated_at: string; stale_seconds: number;
}
interface Payload {
  rabbitmq: { available: boolean; reason?: string; queues?: Queue[] };
  outbox: { pending: number; abandoned: number; recent: OutboxRow[] };
  slurm: { jobs: SlurmJob[]; cluster_probe: { available: boolean; reason: string } };
  orphans: Orphan[];
  states: { status: string; artifact: string | null; n: number }[];
}

export function AdminQueue({ user, revision }: { user: User | null; revision: number }) {
  const data = useApi<Payload>(user?.is_admin ? '/api/admin/queue' : null, [revision]);

  if (!user?.is_admin) {
    return (
      <div className="page">
        <div className="note note-warning">This page is available to administrators only.</div>
      </div>
    );
  }
  if (data.loading) return <div className="page"><Loading what="queue status" /></div>;
  if (data.error) return <div className="page"><ErrorNote error={data.error} /></div>;
  if (!data.data) return null;

  const broker = data.data.rabbitmq;

  return (
    <div className="page page-wide">
      <div className="page-head">
        <div className="spread">
          <div>
            <h1>Queue and cluster</h1>
            <p>Broker, outbox, SLURM tasks, and runs that have stopped progressing.</p>
          </div>
          <Link className="button" to="/admin">← Admin</Link>
        </div>
      </div>

      <div className="section">
        <div className="section-head"><h2>RabbitMQ</h2></div>
        {broker.available && broker.queues ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Queue</th><th className="num">Messages</th>
                  <th className="num">Ready</th><th className="num">Unacknowledged</th>
                  <th className="num">Consumers</th><th>Type</th>
                </tr>
              </thead>
              <tbody>
                {broker.queues.map((queue) => (
                  <tr key={queue.name}>
                    <td className="mono small">{queue.name}</td>
                    <td className="num">{integer(queue.messages)}</td>
                    <td className="num">{integer(queue.ready)}</td>
                    <td className="num">{integer(queue.unacknowledged)}</td>
                    <td className="num">
                      {queue.consumers === 0
                        ? <span style={{ color: 'var(--error)' }} title="No consumers are reading this queue">0</span>
                        : queue.consumers}
                    </td>
                    <td className="small">
                      {queue.is_dlq
                        ? <span style={{ color: 'var(--error)' }}>dead-letter queue</span>
                        : 'work queue'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="note note-warning">
            Could not read broker status.
            <div className="small mono" style={{ marginTop: 4 }}>{broker.reason}</div>
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-head">
          <h2>Outbox</h2>
          <span className="small muted">
            pending: <span className="mono">{data.data.outbox.pending}</span>
          </span>
        </div>
        <p className="small muted">
          The submission and queue message are written in one transaction, while
          a separate process publishes them. This keeps a broker outage from losing the submission
          or returning an error to the user, and the API needs no queue credentials.
        </p>
        {data.data.outbox.abandoned > 0 && (
          <div className="note note-error">
            <strong className="mono">{data.data.outbox.abandoned}</strong> messages
            exhausted their retry budget and will not be published. Each is a
            submission that exists in the database and will never reach the cluster. Rows
            at the maximum attempt count appear below with the last error.
          </div>
        )}
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th className="num">#</th><th>Exchange</th><th>Routing key</th>
                <th>Created</th><th>Published</th>
                <th className="num">Attempts</th><th>Last error</th>
              </tr>
            </thead>
            <tbody>
              {data.data.outbox.recent.map((row) => (
                <tr key={row.id}>
                  <td className="num">{row.id}</td>
                  <td className="mono small">{row.exchange}</td>
                  <td className="mono small">{row.routing_key}</td>
                  <td className="mono small">{timestamp(row.created_at)}</td>
                  <td className="mono small">
                    {row.published_at
                      ? timestamp(row.published_at)
                      : <span style={{ color: 'var(--warning)' }}>pending</span>}
                  </td>
                  <td className="num">{row.attempts}</td>
                  <td className="small mono">{row.last_error ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {data.data.outbox.recent.length === 0 && <div className="empty">Outbox is empty.</div>}
      </div>

      <div className="section">
        <div className="section-head"><h2>Queued and running tasks</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Name</th><th>Status</th><th>SLURM task ID</th>
                <th className="num">Time since submission</th><th>Created</th>
              </tr>
            </thead>
            <tbody>
              {data.data.slurm.jobs.map((job) => (
                <tr key={job.task_id}>
                  <td><Link className="mono small" to={`/runs/${job.task_id}`}>
                    {job.run_name ?? shortId(job.task_id)}
                  </Link></td>
                  <td className="small">{job.task_status}</td>
                  <td className="mono small">{job.executor_task_id ?? '—'}</td>
                  <td className="num small">{duration(job.elapsed_seconds)}</td>
                  <td className="mono small">{timestamp(job.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {data.data.slurm.jobs.length === 0 && (
          <div className="empty">Nothing is queued or running.</div>
        )}
        <div className="note" style={{ marginTop: 'var(--space-3)' }}>
          <strong>Athena node status is not shown here.</strong>
          <p className="small" style={{ marginTop: 4, marginBottom: 0 }}>
            {data.data.slurm.cluster_probe.reason}
          </p>
        </div>
      </div>

      <div className="section">
        <div className="section-head">
          <h2>Runs with no progress</h2>
          <span className="small muted mono">{data.data.orphans.length}</span>
        </div>
        <p className="small muted">
          Tasks in queued or running state whose state has not changed
          for more than two hours. A typical cause is a worker that died after submitting the
          task but before writing its job number — the task remains on the cluster,
          consumes grant time, and nobody is watching it.
        </p>
        {data.data.orphans.length === 0 ? (
          <div className="empty">No orphaned runs.</div>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Name</th><th>Status</th><th>SLURM task ID</th>
                  <th className="num">Unchanged for</th><th>Last change</th>
                </tr>
              </thead>
              <tbody>
                {data.data.orphans.map((row) => (
                  <tr key={row.task_id}>
                    <td><Link className="mono small" to={`/runs/${row.task_id}`}>
                      {row.run_name ?? shortId(row.task_id)}
                    </Link></td>
                    <td className="small">{row.task_status}</td>
                    <td className="mono small">{row.executor_task_id ?? '—'}</td>
                    <td className="num small" style={{ color: 'var(--warning)' }}>
                      {duration(row.stale_seconds)}
                    </td>
                    <td className="mono small">{timestamp(row.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-head"><h2>State distribution</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr><th>task_status</th><th>artifact_status</th><th className="num">Count</th></tr>
            </thead>
            <tbody>
              {data.data.states.map((row, i) => (
                <tr key={i}>
                  <td className="mono small">{row.status}</td>
                  <td className="mono small">{row.artifact ?? '—'}</td>
                  <td className="num">{integer(row.n)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
