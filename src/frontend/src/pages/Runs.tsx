import { Link, useSearchParams } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { ErrorNote, Loading } from '../components/Empty';
import { StatusPill } from '../components/StatusPill';
import { decimal, integer, shortId, since, timestamp } from '../components/format';
import { query } from '../api';
import type { Run, User } from '../types';

interface Payload { runs: Run[]; total: number; limit: number; offset: number }
interface Filters {
  datasets: string[]; models: string[]; optimizers: string[];
  families: string[]; suites: string[]; statuses: string[];
}

export function Runs({ user, revision }: { user: User | null; revision: number }) {
  const [params, setParams] = useSearchParams();
  const mine = params.get('mine') === '1';
  const status = params.get('status') ?? '';
  const dataset = params.get('dataset') ?? '';
  const optimizer = params.get('optimizer') ?? '';
  const search = params.get('q') ?? '';
  const offset = Number(params.get('offset') ?? 0);

  const filters = useApi<Filters>('/api/runs/filters');
  const runs = useApi<Payload>(
    `/api/runs${query({ mine: mine || undefined, status, dataset, optimizer, search, offset, limit: 50 })}`,
    [revision],
  );

  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value); else next.delete(key);
    next.delete('offset');
    setParams(next, { replace: true });
  }

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1>Uruchomienia</h1>
        <p>
          Zadanie czeka w cudzym schedulerze, więc ma więcej stanów niż „trwa”
          i „gotowe”. Każdy ma własny komunikat — pozycję w kolejce, numer zadania
          SLURM, czas oczekiwania albo powód niepowodzenia.
        </p>
      </div>

      <div className="controls section">
        {user && (
          <div className="segmented">
            <button className={!mine ? 'active' : ''} onClick={() => update('mine', '')}>
              Wszystkie
            </button>
            <button className={mine ? 'active' : ''} onClick={() => update('mine', '1')}>
              Moje
            </button>
          </div>
        )}
        <div>
          <label htmlFor="r-status">Stan</label>
          <select id="r-status" value={status} onChange={(e) => update('status', e.target.value)}>
            <option value="">wszystkie</option>
            {filters.data?.statuses.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="r-dataset">Zbiór</label>
          <select id="r-dataset" value={dataset} onChange={(e) => update('dataset', e.target.value)}>
            <option value="">wszystkie</option>
            {filters.data?.datasets.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="r-opt">Optymalizator</label>
          <select id="r-opt" value={optimizer} onChange={(e) => update('optimizer', e.target.value)}>
            <option value="">wszystkie</option>
            {filters.data?.optimizers.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="r-q">Nazwa lub identyfikator</label>
          <input id="r-q" defaultValue={search} placeholder="cma-es-wine…"
                 onKeyDown={(e) => {
                   if (e.key === 'Enter') update('q', (e.target as HTMLInputElement).value);
                 }} />
        </div>
      </div>

      {runs.loading && <Loading what="uruchomienia" />}
      {runs.error && <ErrorNote error={runs.error} />}

      {runs.data && (
        <>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Stan</th>
                  <th>Nazwa</th>
                  <th>Optymalizator</th>
                  <th>Zbiór / model</th>
                  <th className="num">Ziarno</th>
                  <th className="num">Strata</th>
                  <th className="num">Gradienty</th>
                  <th className="num">Próbki</th>
                  <th>Powód stopu</th>
                  <th className="num">W kolejce</th>
                  <th>Utworzone</th>
                  <th>ID</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((run) => (
                  <tr key={run.task_id}>
                    <td>
                      <StatusPill label={run.state_label} tone={run.state_tone}
                                  title={run.state_detail} />
                    </td>
                    <td>
                      <Link className="mono" to={`/runs/${run.task_id}`}>
                        {run.run_name ?? shortId(run.task_id)}
                      </Link>
                    </td>
                    <td className="mono small">{run.optimizer ?? '—'}</td>
                    <td className="mono small">{run.dataset} / {run.model}</td>
                    <td className="num small">{run.seed ?? '—'}</td>
                    <td className="num">{decimal(run.metrics?.final_loss ?? null)}</td>
                    <td className="num">{integer(run.metrics?.gradient_count ?? null)}</td>
                    <td className="num">{integer(run.metrics?.database_reaches ?? null)}</td>
                    <td className="small mono">{run.stop_reason ?? '—'}</td>
                    <td className="num small">
                      {/* Time between queueing and starting: the number a user
                          actually wants when a run "has not done anything". */}
                      {run.queued_at ? since(run.queued_at, run.started_at) : '—'}
                    </td>
                    <td className="small mono">{timestamp(run.created_at)}</td>
                    <td className="tiny mono dim">{shortId(run.task_id)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {runs.data.runs.length === 0 && (
            <div className="empty">Brak uruchomień spełniających kryteria.</div>
          )}

          <div className="spread" style={{ marginTop: 'var(--space-3)' }}>
            <span className="small muted mono">
              {runs.data.runs.length} z {runs.data.total}
            </span>
            <div className="row">
              <button disabled={offset === 0}
                      onClick={() => update('offset', String(Math.max(0, offset - 50)))}>
                Poprzednie
              </button>
              <button disabled={offset + 50 >= runs.data.total}
                      onClick={() => update('offset', String(offset + 50))}>
                Następne
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
