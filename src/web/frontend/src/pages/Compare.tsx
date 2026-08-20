import { useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { Chart } from '../components/Chart';
import { ErrorNote, Loading } from '../components/Empty';
import { decimal, familyLabel, integer } from '../components/format';
import { query } from '../api';
import type { AggregatedSeries, Run } from '../types';

interface Difference {
  a: string; b: string; median_a: number; median_b: number;
  delta: number; relative: number | null; n_a: number; n_b: number;
}
interface Payload {
  series: AggregatedSeries[];
  differences: Difference[];
  runs: Run[];
  missing: number;
  statistical_test: { available: boolean; note: string };
}

const AXES = [
  { id: 'gradient_count', label: 'gradienty', axis: 'Wyliczone gradienty' },
  { id: 'database_reaches', label: 'próbki', axis: 'Przetworzone próbki' },
  { id: 'epoch', label: 'epoki', axis: 'Epoka' },
];
const GROUPINGS = [
  { id: 'optimizer', label: 'optymalizator' },
  { id: 'optimizer_dataset', label: 'optymalizator × zbiór' },
  { id: 'optimizer_model', label: 'optymalizator × model' },
  { id: 'run', label: 'pojedyncze przebiegi' },
];

export function Compare() {
  const [params, setParams] = useSearchParams();
  const runs = params.get('runs') ?? '';
  const axis = params.get('x') ?? 'gradient_count';
  const metric = params.get('metric') ?? 'loss';
  const groupBy = params.get('group_by') ?? 'optimizer';
  const logY = params.get('logy') === '1';
  const logX = params.get('logx') === '1';

  const [picker, setPicker] = useState(false);
  const available = useApi<{ runs: Run[] }>('/api/runs?status=completed&limit=200');
  const data = useApi<Payload>(
    runs ? `/api/compare${query({ runs, x: axis, metric, group_by: groupBy, logx: logX, points: 240 })}` : null,
  );

  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value); else next.delete(key);
    setParams(next, { replace: true });
  }

  const selected = useMemo(() => new Set(runs.split(',').filter(Boolean)), [runs]);

  function toggle(taskId: string) {
    const next = new Set(selected);
    if (next.has(taskId)) next.delete(taskId); else next.add(taskId);
    update('runs', [...next].join(','));
  }

  const exportHref = `/api/compare/export.csv${query({
    runs, x: axis, metric, group_by: groupBy, logx: logX, points: 240,
  })}`;

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1>Porównanie</h1>
        <p>
          Nałożone krzywe zbieżności z medianą i wstęgą międzykwartylową po
          ziarnach. Zestaw porównywanych przebiegów jest zakodowany w adresie,
          więc ten widok da się wysłać linkiem.
        </p>
      </div>

      <div className="controls section">
        <div>
          <label htmlFor="c-axis">Oś X — budżet</label>
          <div className="segmented" id="c-axis">
            {AXES.map((option) => (
              <button key={option.id} className={axis === option.id ? 'active' : ''}
                      onClick={() => update('x', option.id)}>{option.label}</button>
            ))}
          </div>
        </div>
        <div>
          <label htmlFor="c-metric">Metryka</label>
          <div className="segmented" id="c-metric">
            <button className={metric === 'loss' ? 'active' : ''}
                    onClick={() => update('metric', 'loss')}>strata</button>
            <button className={metric === 'accuracy' ? 'active' : ''}
                    onClick={() => update('metric', 'accuracy')}>dokładność</button>
          </div>
        </div>
        <div>
          <label htmlFor="c-group">Grupowanie</label>
          <select id="c-group" value={groupBy} onChange={(e) => update('group_by', e.target.value)}>
            {GROUPINGS.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
          </select>
        </div>
        <div>
          <label>Skala</label>
          <div className="row" style={{ gap: 'var(--space-2)' }}>
            <button className={logY ? 'active' : ''} onClick={() => update('logy', logY ? '' : '1')}>
              log Y
            </button>
            <button className={logX ? 'active' : ''} onClick={() => update('logx', logX ? '' : '1')}>
              log X
            </button>
          </div>
        </div>
        <button onClick={() => setPicker(!picker)}>
          {picker ? 'Ukryj wybór' : `Wybierz przebiegi (${selected.size})`}
        </button>
        {runs && <a className="button" href={exportHref}>Eksport CSV</a>}
      </div>

      {picker && (
        <div className="panel section">
          <div className="panel-head">
            <h3>Zakończone uruchomienia</h3>
            <button className="subtle" onClick={() => update('runs', '')}>Wyczyść wybór</button>
          </div>
          <div className="table-scroll" style={{ maxHeight: 320, overflowY: 'auto', border: 'none' }}>
            <table>
              <thead>
                <tr>
                  <th /><th>Nazwa</th><th>Optymalizator</th><th>Rodzina</th>
                  <th>Zbiór / model</th><th className="num">Ziarno</th><th className="num">Strata</th>
                </tr>
              </thead>
              <tbody>
                {available.data?.runs.map((run) => (
                  <tr key={run.task_id}
                      className={selected.has(run.task_id) ? 'selected' : ''}
                      onClick={() => toggle(run.task_id)}
                      style={{ cursor: 'pointer' }}>
                    <td><input type="checkbox" readOnly checked={selected.has(run.task_id)} /></td>
                    <td className="mono small">{run.run_name}</td>
                    <td className="mono small">{run.optimizer}</td>
                    <td className="small">{familyLabel(run.family)}</td>
                    <td className="mono small">{run.dataset} / {run.model}</td>
                    <td className="num small">{run.seed}</td>
                    <td className="num">{decimal(run.metrics?.final_loss ?? null)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!runs && (
        <div className="empty">
          Nie wybrano żadnych przebiegów. Użyj przycisku „Wybierz przebiegi”
          albo przejdź tu z <Link to="/leaderboard">rankingu</Link>.
        </div>
      )}

      {data.loading && <Loading what="serie" />}
      {data.error && <ErrorNote error={data.error} />}

      {data.data && data.data.series.length > 0 && (
        <>
          <div className="panel section">
            <div className="panel-body">
              <Chart
                series={data.data.series}
                logY={logY}
                logX={logX}
                height={430}
                xLabel={AXES.find((a) => a.id === axis)?.axis ?? 'Budżet'}
                yLabel={metric === 'loss' ? 'Strata' : 'Dokładność [%]'}
              />
            </div>
          </div>

          {data.data.missing > 0 && (
            <div className="note note-warning section">
              {data.data.missing} z wybranych przebiegów nie jest dostępnych —
              albo nie mają jeszcze wyników, albo nie masz do nich dostępu.
            </div>
          )}

          <div className="section">
            <div className="section-head"><h2>Tabela różnic</h2></div>
            {data.data.differences.length === 0 ? (
              <div className="empty">Potrzeba co najmniej dwóch serii.</div>
            ) : (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Seria A</th><th>Seria B</th>
                      <th className="num">Mediana A</th><th className="num">Mediana B</th>
                      <th className="num">Różnica</th><th className="num">Względnie</th>
                      <th className="num">n(A)</th><th className="num">n(B)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.data.differences.map((d, i) => (
                      <tr key={i}>
                        <td className="mono small">{d.a}</td>
                        <td className="mono small">{d.b}</td>
                        <td className="num">{decimal(d.median_a)}</td>
                        <td className="num">{decimal(d.median_b)}</td>
                        <td className="num" style={{
                          color: d.delta < 0 ? 'var(--success)' : 'var(--error)',
                        }}>
                          {d.delta > 0 ? '+' : ''}{decimal(d.delta)}
                        </td>
                        <td className="num">
                          {d.relative === null ? '—'
                            : `${d.relative > 0 ? '+' : ''}${(d.relative * 100).toFixed(1)}%`}
                        </td>
                        <td className="num">{d.n_a}</td>
                        <td className="num">{d.n_b}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="note section" style={{ marginTop: 'var(--space-3)' }}>
              <strong>Bez p-wartości — celowo.</strong>
              <p className="small" style={{ marginTop: 6, marginBottom: 0 }}>
                {data.data.statistical_test.note}
              </p>
            </div>
          </div>

          <div className="section">
            <div className="section-head"><h2>Przebiegi w porównaniu</h2></div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Nazwa</th><th>Optymalizator</th><th>Zbiór / model</th>
                    <th className="num">Ziarno</th><th className="num">Strata</th>
                    <th className="num">Gradienty</th><th className="num">Próbki</th>
                    <th>Powód stopu</th>
                  </tr>
                </thead>
                <tbody>
                  {data.data.runs.map((run) => (
                    <tr key={run.task_id}>
                      <td><Link className="mono small" to={`/runs/${run.task_id}`}>
                        {run.run_name}
                      </Link></td>
                      <td className="mono small">{run.optimizer}</td>
                      <td className="mono small">{run.dataset} / {run.model}</td>
                      <td className="num small">{run.seed}</td>
                      <td className="num">{decimal(run.metrics?.final_loss ?? null)}</td>
                      <td className="num">{integer(run.metrics?.gradient_count ?? null)}</td>
                      <td className="num">{integer(run.metrics?.database_reaches ?? null)}</td>
                      <td className="mono small">{run.stop_reason ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
