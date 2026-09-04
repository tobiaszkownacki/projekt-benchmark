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
  outbox: { pending: number; recent: OutboxRow[] };
  slurm: { jobs: SlurmJob[]; cluster_probe: { available: boolean; reason: string } };
  orphans: Orphan[];
  states: { status: string; artifact: string | null; n: number }[];
}

export function AdminQueue({ user, revision }: { user: User | null; revision: number }) {
  const data = useApi<Payload>(user?.is_admin ? '/api/admin/queue' : null, [revision]);

  if (!user?.is_admin) {
    return (
      <div className="page">
        <div className="note note-warning">Ta strona jest dostępna tylko dla administratorów.</div>
      </div>
    );
  }
  if (data.loading) return <div className="page"><Loading what="stan kolejki" /></div>;
  if (data.error) return <div className="page"><ErrorNote error={data.error} /></div>;
  if (!data.data) return null;

  const broker = data.data.rabbitmq;

  return (
    <div className="page page-wide">
      <div className="page-head">
        <div className="spread">
          <div>
            <h1>Kolejka i klaster</h1>
            <p>Stan brokera, outboxu, zadań w SLURM i uruchomień, które przestały się ruszać.</p>
          </div>
          <Link className="button" to="/admin">← Panel</Link>
        </div>
      </div>

      <div className="section">
        <div className="section-head"><h2>RabbitMQ</h2></div>
        {broker.available && broker.queues ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Kolejka</th><th className="num">Wiadomości</th>
                  <th className="num">Gotowe</th><th className="num">Niepotwierdzone</th>
                  <th className="num">Konsumenci</th><th>Rodzaj</th>
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
                        ? <span style={{ color: 'var(--error)' }} title="Nikt nie czyta z tej kolejki">0</span>
                        : queue.consumers}
                    </td>
                    <td className="small">
                      {queue.is_dlq
                        ? <span style={{ color: 'var(--error)' }}>martwe listy</span>
                        : 'robocza'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="note note-warning">
            Nie udało się odczytać stanu brokera.
            <div className="small mono" style={{ marginTop: 4 }}>{broker.reason}</div>
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-head">
          <h2>Outbox</h2>
          <span className="small muted">
            oczekujących: <span className="mono">{data.data.outbox.pending}</span>
          </span>
        </div>
        <p className="small muted">
          Zgłoszenie i wiadomość na kolejkę są zapisywane w jednej transakcji, a
          publikuje je osobny proces. Dzięki temu awaria brokera nie gubi zgłoszenia
          i nie zwraca użytkownikowi błędu, a API nie ma poświadczeń do kolejki.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th className="num">#</th><th>Exchange</th><th>Routing key</th>
                <th>Utworzona</th><th>Opublikowana</th>
                <th className="num">Prób</th><th>Ostatni błąd</th>
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
                      : <span style={{ color: 'var(--warning)' }}>oczekuje</span>}
                  </td>
                  <td className="num">{row.attempts}</td>
                  <td className="small mono">{row.last_error ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {data.data.outbox.recent.length === 0 && <div className="empty">Outbox jest pusty.</div>}
      </div>

      <div className="section">
        <div className="section-head"><h2>Zadania w kolejce i w obliczeniach</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Nazwa</th><th>Stan</th><th>ID zadania SLURM</th>
                <th className="num">Czas od zgłoszenia</th><th>Utworzone</th>
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
          <div className="empty">Nic nie czeka i nic się nie liczy.</div>
        )}
        <div className="note" style={{ marginTop: 'var(--space-3)' }}>
          <strong>Stan węzłów Ateny nie jest tu pokazany.</strong>
          <p className="small" style={{ marginTop: 4, marginBottom: 0 }}>
            {data.data.slurm.cluster_probe.reason}
          </p>
        </div>
      </div>

      <div className="section">
        <div className="section-head">
          <h2>Uruchomienia bez ruchu</h2>
          <span className="small muted mono">{data.data.orphans.length}</span>
        </div>
        <p className="small muted">
          Zadania w stanie „w kolejce” albo „liczy się”, których stan nie zmienił się
          od ponad dwóch godzin. Typowa przyczyna to worker, który padł po wysłaniu
          zadania, a przed zapisaniem jego numeru — zadanie zostaje na klastrze,
          pali grant, a nikt go nie obserwuje.
        </p>
        {data.data.orphans.length === 0 ? (
          <div className="empty">Nie ma osieroconych uruchomień.</div>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Nazwa</th><th>Stan</th><th>ID zadania SLURM</th>
                  <th className="num">Bez zmiany od</th><th>Ostatnia zmiana</th>
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
        <div className="section-head"><h2>Rozkład stanów</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr><th>task_status</th><th>artifact_status</th><th className="num">Liczba</th></tr>
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
