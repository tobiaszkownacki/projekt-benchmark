import { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { Chart } from './Chart';
import { ErrorNote, Loading } from './Empty';
import { query } from '../api';
import type { AggregatedSeries } from '../types';

interface Cell { dataset: string; model: string; suite: string; n_optimizers: number; n_runs: number }
interface Payload {
  cells: Cell[];
  selected: { dataset: string; model: string; suite: string } | null;
  min_runs: number;
  series: AggregatedSeries[];
  withheld: { label: string; n_runs: number }[];
}

const AXES = [
  { id: 'database_reaches', label: 'Processed samples' },
  { id: 'gradient_count', label: 'Gradient evaluations' },
  { id: 'epoch', label: 'Epochs' },
];

const cellKey = (c: { dataset: string; model: string; suite: string }) => `${c.dataset}|${c.model}|${c.suite}`;

/** Convergence per optimizer, open to every visitor.
 *
 *  The server sends only medians and quartiles over groups of at least
 *  `min_runs` runs, so the chart shows how methods behave without exposing
 *  any participant's run. */
export function PublicCurves({ height = 380, compact = false }: { height?: number; compact?: boolean }) {
  const [cell, setCell] = useState('');
  const [axis, setAxis] = useState('database_reaches');
  const [metric, setMetric] = useState('loss');
  const [logY, setLogY] = useState(false);

  const [dataset, model, suite] = cell ? cell.split('|') : [];
  const data = useApi<Payload>(
    `/api/leaderboard/curves${query({ dataset, model, suite, x: axis, metric })}`,
  );
  const selected = data.data?.selected ? cellKey(data.data.selected) : cell;

  if (data.data && data.data.cells.length === 0) return null;

  return (
    <div className="section">
      <div className="section-head"><h2>Convergence by method</h2></div>
      {!compact && (
        <p className="small muted">
          Median with interquartile band across runs of each method. Groups with fewer
          than {data.data?.min_runs ?? 3} runs are not drawn.
        </p>
      )}
      <div className="controls">
        <div>
          <label htmlFor="pc-cell">Problem</label>
          <select id="pc-cell" value={selected} onChange={(e) => setCell(e.target.value)}>
            {data.data?.cells.map((c) => (
              <option key={cellKey(c)} value={cellKey(c)}>
                {c.dataset} · {c.model}{c.suite !== 'test' ? ` · ${c.suite}` : ''} ({c.n_optimizers} methods)
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="pc-axis">Budget axis</label>
          <select id="pc-axis" value={axis} onChange={(e) => setAxis(e.target.value)}>
            {AXES.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="pc-metric">Metric</label>
          <select id="pc-metric" value={metric} onChange={(e) => setMetric(e.target.value)}>
            <option value="loss">Loss</option>
            <option value="accuracy">Accuracy</option>
          </select>
        </div>
        <div>
          <label>Scale</label>
          <button className={logY ? 'active' : ''} onClick={() => setLogY((v) => !v)}>log Y</button>
        </div>
      </div>

      {data.loading && !data.data && <Loading what="curves" />}
      {data.error && <ErrorNote error={data.error} />}
      {data.data && data.data.series.length > 0 && (
        <div className="panel" style={{ marginTop: 'var(--space-3)' }}>
          <div className="panel-body">
            <Chart
              series={data.data.series}
              logY={logY}
              height={height}
              xLabel={AXES.find((a) => a.id === axis)?.label ?? 'Budget'}
              yLabel={metric === 'loss' ? 'Loss' : 'Accuracy'}
            />
          </div>
        </div>
      )}
      {data.data && data.data.withheld.length > 0 && (
        <p className="small muted" style={{ marginTop: 'var(--space-2)' }}>
          Not drawn (fewer than {data.data.min_runs} runs):{' '}
          {data.data.withheld.map((w) => `${w.label} (${w.n_runs})`).join(', ')}.
        </p>
      )}
      {axis === 'gradient_count' && (
        <p className="small muted">Gradient-free methods do not evaluate gradients, so their curves collapse to a point on this axis.</p>
      )}
    </div>
  );
}
