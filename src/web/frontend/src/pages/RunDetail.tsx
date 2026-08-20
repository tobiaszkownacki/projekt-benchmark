import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { Chart } from '../components/Chart';
import { Metric } from '../components/Metric';
import { StatusPill } from '../components/StatusPill';
import { Loading } from '../components/Empty';
import {
  bytes, decimal, duration, familyLabel, integer, percent, since, suiteLabel, timestamp,
} from '../components/format';
import { query } from '../api';
import type { AggregatedSeries, Run, User } from '../types';

const AXES = [
  { id: 'gradient_count', label: 'gradienty' },
  { id: 'database_reaches', label: 'próbki' },
  { id: 'epoch', label: 'epoki' },
];

interface SeriesPayload {
  points: [number, number][];
  original_points: number;
  truncated: boolean;
  downsample: string;
}
interface Transition {
  from_status: string | null; to_status: string; actor: string;
  detail: Record<string, unknown> | null; occurred_at: string;
}

export function RunDetail({ user, revision }: { user: User | null; revision: number }) {
  const { taskId } = useParams();
  const [axis, setAxis] = useState('gradient_count');
  const [metric, setMetric] = useState('loss');
  const [logY, setLogY] = useState(false);

  const run = useApi<Run>(`/api/runs/${taskId}`, [revision]);
  const series = useApi<SeriesPayload>(
    `/api/runs/${taskId}/series${query({ x: axis, metric, points: 1000 })}`,
    [revision],
  );
  const history = useApi<{ transitions: Transition[] }>(
    `/api/runs/${taskId}/transitions`, [revision],
  );

  if (run.loading) return <div className="page"><Loading what="uruchomienie" /></div>;
  if (run.error) {
    return (
      <div className="page">
        <div className="note note-error">
          <strong>Nie ma takiego uruchomienia albo nie masz do niego dostępu.</strong>
          <p className="small" style={{ marginTop: 6, marginBottom: 0 }}>
            Wyniki widzi właściciel i administratorzy. Zwracamy 404 zamiast 403,
            żeby nie potwierdzać istnienia cudzego zasobu.
          </p>
        </div>
      </div>
    );
  }
  if (!run.data) return null;

  const item = run.data;
  const budget = item.stop_condition ?? {};
  const asSeries: AggregatedSeries[] = series.data
    ? [{
        label: item.optimizer ?? 'przebieg',
        family: item.family,
        n_runs: 1,
        x: series.data.points.map((p) => p[0]),
        median: series.data.points.map((p) => p[1]),
        q1: series.data.points.map(() => null),
        q3: series.data.points.map(() => null),
        n_at_x: series.data.points.map(() => 1),
        full_until_index: series.data.points.length - 1,
      }]
    : [];

  return (
    <div className="page page-wide">
      <div className="breadcrumb" style={{ marginBottom: 'var(--space-3)' }}>
        <Link to="/runs">Uruchomienia</Link>
        <span className="sep">›</span>
        <span className="mono">{item.run_name ?? item.task_id}</span>
      </div>

      <div className="page-head">
        <div className="spread" style={{ alignItems: 'flex-start' }}>
          <div>
            <h1 style={{ fontFamily: 'var(--font-mono)', fontSize: 22 }}>
              {item.run_name ?? item.task_id}
            </h1>
            <div className="row" style={{ marginTop: 8 }}>
              <StatusPill label={item.state_label} tone={item.state_tone} />
              <span className="small muted">{item.state_detail}</span>
            </div>
          </div>
          <div className="row">
            <Link className="button" to={`/runs/${item.task_id}/files`}>Pliki</Link>
            <a className="button" href={`/api/runs/${item.task_id}/archive.zip`}>Pobierz .zip</a>
          </div>
        </div>
      </div>

      {/* Failures put the reason on the page, not behind a disclosure control.
          §11.3 is explicit about this, and it is the moment a user most needs
          the information. */}
      {item.error_message && (
        <div className="note note-error section">
          <strong>Komunikat błędu</strong>
          <pre className="log" style={{ marginTop: 8, maxHeight: 200 }}>{item.error_message}</pre>
        </div>
      )}

      {item.metrics && (
        <div className="section">
          <div className="metrics">
            <Metric label="Strata końcowa" value={decimal(item.metrics.final_loss)}
                    context={item.stop_reason_label ?? undefined} />
            <Metric label="Dokładność" value={percent(item.metrics.final_accuracy)} />
            <Metric label="Wyliczone gradienty" value={integer(item.metrics.gradient_count)}
                    hint="Rośnie o 1 przy każdym evaluate_with_grad() i grad()."
                    context={budget.max_gradient_count
                      ? `limit ${integer(budget.max_gradient_count)}` : 'bez limitu'} />
            <Metric label="Przetworzone próbki" value={integer(item.metrics.database_reaches)}
                    hint="Rośnie o batch_size przy każdym przejściu w przód."
                    context={budget.max_database_reaches
                      ? `limit ${integer(budget.max_database_reaches)}` : 'bez limitu'} />
            <Metric label="Epoki" value={integer(item.metrics.total_epochs)}
                    context={budget.max_epochs ? `limit ${integer(budget.max_epochs)}` : 'bez limitu'} />
            <Metric label="Czas obliczeń" value={duration(item.metrics.wall_time_seconds)}
                    context="nie jest metryką rankingu" />
          </div>
          {item.converged !== null && (
            <div className={`note ${item.converged ? 'note-accent' : ''}`}
                 style={{ marginTop: 'var(--space-3)' }}>
              <strong>{item.stop_reason_label}</strong>{' '}
              <span className="small muted">
                {item.converged
                  ? 'Optymalizator sam zgłosił zbieżność przed wyczerpaniem budżetu.'
                  : 'Budżet się skończył — to nie to samo, co zbieżność.'}
              </span>
            </div>
          )}
        </div>
      )}

      <div className="section grid-2">
        <div className="panel">
          <div className="panel-head"><h3>Metadane</h3></div>
          <div className="panel-body">
            <dl className="kv">
              <dt>Identyfikator</dt><dd>{item.task_id}</dd>
              <dt>Optymalizator</dt><dd>{item.optimizer ?? '—'}</dd>
              <dt>Rodzina metody</dt><dd>{familyLabel(item.family)}</dd>
              <dt>Zbiór danych</dt><dd>{item.dataset ?? '—'}</dd>
              <dt>Model</dt><dd>{item.model ?? '—'}</dd>
              <dt>Zestaw</dt><dd>{suiteLabel(item.suite)}</dd>
              <dt>Ziarno</dt><dd>{item.seed ?? '—'}</dd>
              <dt>ID zadania SLURM</dt><dd>{item.slurm_job_id ?? '—'}</dd>
              <dt>Wykonawca</dt><dd>{item.executor ?? '—'}</dd>
              <dt>Wersja runnera</dt><dd>{item.runner_version ?? '—'}</dd>
              <dt>Sprzęt</dt><dd>{item.gpu_model ?? '—'}</dd>
              <dt>Zgłaszający</dt><dd>{item.submitter_name ?? '—'}</dd>
              <dt>Artefakty</dt>
              <dd>{item.artifact_files ?? 0} plików · {bytes(item.artifact_bytes)}</dd>
            </dl>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head"><h3>Historia stanów</h3></div>
          <div className="panel-body">
            {history.data && history.data.transitions.length > 0 ? (
              <table>
                <thead>
                  <tr><th>Kiedy</th><th>Przejście</th><th>Kto</th></tr>
                </thead>
                <tbody>
                  {history.data.transitions.map((t, index) => (
                    <tr key={index}>
                      <td className="mono small">{timestamp(t.occurred_at)}</td>
                      <td className="mono small">
                        {t.from_status ? `${t.from_status} → ` : ''}{t.to_status}
                      </td>
                      <td className="small">{t.actor}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="small muted" style={{ marginBottom: 0 }}>Brak zapisanych przejść.</p>
            )}
            <div className="small muted" style={{ marginTop: 'var(--space-3)' }}>
              Czas w kolejce: <span className="mono">
                {item.queued_at ? since(item.queued_at, item.started_at) : '—'}
              </span>
              {' · '}Czas liczenia: <span className="mono">
                {item.started_at ? since(item.started_at, item.completed_at) : '—'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {asSeries.length > 0 && asSeries[0].x.length > 0 && (
        <div className="section">
          <div className="section-head">
            <h2>Krzywa zbieżności</h2>
            <div className="row">
              <div className="segmented">
                {AXES.map((option) => (
                  <button key={option.id} className={axis === option.id ? 'active' : ''}
                          onClick={() => setAxis(option.id)}>
                    {option.label}
                  </button>
                ))}
              </div>
              <div className="segmented">
                <button className={metric === 'loss' ? 'active' : ''}
                        onClick={() => setMetric('loss')}>strata</button>
                <button className={metric === 'accuracy' ? 'active' : ''}
                        onClick={() => setMetric('accuracy')}>dokładność</button>
              </div>
              <button className={logY ? 'active' : ''} onClick={() => setLogY(!logY)}>
                skala log
              </button>
            </div>
          </div>
          <div className="panel"><div className="panel-body">
            <Chart series={asSeries} showBand={false} logY={logY}
                   xLabel={axis === 'epoch' ? 'Epoka'
                     : axis === 'gradient_count' ? 'Wyliczone gradienty' : 'Przetworzone próbki'}
                   yLabel={metric === 'loss' ? 'Strata' : 'Dokładność [%]'} />
            {series.data?.truncated && (
              <div className="tiny dim" style={{ marginTop: 6 }}>
                {/* Stated rather than hidden: the reader is looking at an
                    approximation and is entitled to know. */}
                Wykres pokazuje {series.data.points.length} z{' '}
                {integer(series.data.original_points)} punktów (downsampling {series.data.downsample}).
              </div>
            )}
          </div></div>
          <div className="small muted" style={{ marginTop: 'var(--space-2)' }}>
            Oś X to budżet, nie czas zegarowy — czas zależy od tego, jaki węzeł
            przydzielił scheduler, a nie od jakości optymalizatora.
          </div>
        </div>
      )}

      {!item.metrics && (
        <div className="empty">
          Ten przebieg nie ma jeszcze wyników. {item.state_detail}
        </div>
      )}

      {user?.is_admin && (
        <div className="small dim" style={{ marginTop: 'var(--space-5)' }}>
          Widok administratora: masz dostęp do wszystkich uruchomień.
        </div>
      )}
    </div>
  );
}
