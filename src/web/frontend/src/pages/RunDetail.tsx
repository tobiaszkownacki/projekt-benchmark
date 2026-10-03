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
  { id: 'gradient_count', label: 'gradient evaluations' },
  { id: 'database_reaches', label: 'processed samples' },
  { id: 'epoch', label: 'epochs' },
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

  if (run.loading) return <div className="page"><Loading what="run" /></div>;
  if (run.error) {
    return (
      <div className="page">
        <div className="note note-error">
          <strong>Run not found or you do not have access.</strong>
          <p className="small" style={{ marginTop: 6, marginBottom: 0 }}>
            Results are visible to the owner and administrators. We return 404 instead of 403
            so we do not confirm that someone else’s resource exists.
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
        label: item.optimizer ?? 'run',
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
        <Link to="/runs">Runs</Link>
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
            <Link className="button" to={`/runs/${item.task_id}/files`}>Files</Link>
            <a className="button" href={`/api/runs/${item.task_id}/archive.zip`}>Download .zip</a>
          </div>
        </div>
      </div>

      {/* Failures put the reason directly on the page, not behind a disclosure
          control, since this is the moment a user most needs the information. */}
      {item.error_message && (
        <div className="note note-error section">
          <strong>Error message</strong>
          <pre className="log" style={{ marginTop: 8, maxHeight: 200 }}>{item.error_message}</pre>
        </div>
      )}

      {item.metrics && (
        <div className="section">
          <div className="metrics">
            <Metric label="Final loss" value={decimal(item.metrics.final_loss)}
                    context={item.stop_reason_label ?? undefined} />
            <Metric label="Accuracy" value={percent(item.metrics.final_accuracy)} />
            <Metric label="Gradient evaluations" value={integer(item.metrics.gradient_count)}
                    hint="Increases by 1 for each evaluate_with_grad() and grad()."
                    context={budget.max_gradient_count
                      ? `limit ${integer(budget.max_gradient_count)}` : 'unlimited'} />
            <Metric label="Processed samples" value={integer(item.metrics.database_reaches)}
                    hint="Increases by batch_size on each forward pass."
                    context={budget.max_database_reaches
                      ? `limit ${integer(budget.max_database_reaches)}` : 'unlimited'} />
            <Metric label="Epochs" value={integer(item.metrics.total_epochs)}
                    context={budget.max_epochs ? `limit ${integer(budget.max_epochs)}` : 'unlimited'} />
            <Metric label="Compute time" value={duration(item.metrics.wall_time_seconds)}
                    context="not a leaderboard metric" />
          </div>
          {item.converged !== null && (
            <div className={`note ${item.converged ? 'note-accent' : ''}`}
                 style={{ marginTop: 'var(--space-3)' }}>
              <strong>{item.stop_reason_label}</strong>{' '}
              <span className="small muted">
                {item.converged
                  ? 'The optimizer reported convergence before the budget was exhausted.'
                  : 'The budget was exhausted. That is not the same as convergence.'}
              </span>
            </div>
          )}
        </div>
      )}

      <div className="section grid-2">
        <div className="panel">
          <div className="panel-head"><h3>Metadata</h3></div>
          <div className="panel-body">
            <dl className="kv">
              <dt>Identifier</dt><dd>{item.task_id}</dd>
              <dt>Optimizer</dt><dd>{item.optimizer ?? 'N/A'}</dd>
              <dt>Method family</dt><dd>{familyLabel(item.family)}</dd>
              <dt>Dataset</dt><dd>{item.dataset ?? 'N/A'}</dd>
              <dt>Model</dt><dd>{item.model ?? 'N/A'}</dd>
              <dt>Suite</dt><dd>{suiteLabel(item.suite)}</dd>
              <dt>Seed</dt><dd>{item.seed ?? 'N/A'}</dd>
              <dt>SLURM task ID</dt><dd>{item.slurm_job_id ?? 'N/A'}</dd>
              <dt>Execution environment</dt><dd>{item.executor ?? 'N/A'}</dd>
              <dt>Runner version</dt><dd>{item.runner_version ?? 'N/A'}</dd>
              <dt>Hardware</dt><dd>{item.gpu_model ?? 'N/A'}</dd>
              <dt>Submitter</dt><dd>{item.submitter_name ?? 'N/A'}</dd>
              <dt>Artifacts</dt>
              <dd>{item.artifact_files ?? 0} files · {bytes(item.artifact_bytes)}</dd>
            </dl>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head"><h3>State history</h3></div>
          <div className="panel-body">
            {history.data && history.data.transitions.length > 0 ? (
              <table>
                <thead>
                  <tr><th>When</th><th>Transition</th><th>By</th></tr>
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
              <p className="small muted" style={{ marginBottom: 0 }}>No recorded transitions.</p>
            )}
            <div className="small muted" style={{ marginTop: 'var(--space-3)' }}>
              Queue time: <span className="mono">
                {item.queued_at ? since(item.queued_at, item.started_at) : 'N/A'}
              </span>
              {' · '}Compute time: <span className="mono">
                {item.started_at ? since(item.started_at, item.completed_at) : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {asSeries.length > 0 && asSeries[0].x.length > 0 && (
        <div className="section">
          <div className="section-head">
            <h2>Convergence curve</h2>
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
                        onClick={() => setMetric('accuracy')}>accuracy</button>
              </div>
              <button className={logY ? 'active' : ''} onClick={() => setLogY(!logY)}>
                log scale
              </button>
            </div>
          </div>
          <div className="panel"><div className="panel-body">
            <Chart series={asSeries} showBand={false} logY={logY}
                   xLabel={axis === 'epoch' ? 'Epoch'
                     : axis === 'gradient_count' ? 'Gradient evaluations' : 'Processed samples'}
                   yLabel={metric === 'loss' ? 'Loss' : 'Accuracy [%]'} />
            {series.data?.truncated && (
              <div className="tiny dim" style={{ marginTop: 6 }}>
                {/* Stated rather than hidden: the reader is looking at an
                    approximation and is entitled to know. */}
                Chart shows {series.data.points.length} of{' '}
                {integer(series.data.original_points)} points (downsampling {series.data.downsample}).
              </div>
            )}
          </div></div>
          <div className="small muted" style={{ marginTop: 'var(--space-2)' }}>
            The x-axis is budget, not wall-clock time. The time depends on the node
            assigned by the scheduler, not on optimizer quality.
          </div>
        </div>
      )}

      {!item.metrics && (
        <div className="empty">
          This run has no results yet. {item.state_detail}
        </div>
      )}

      {user?.is_admin && (
        <div className="small dim" style={{ marginTop: 'var(--space-5)' }}>
          Administrator view: you have access to all runs.
        </div>
      )}
    </div>
  );
}
