import { useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { ErrorNote, Loading } from '../components/Empty';
import { decimal, familyLabel, integer, suiteLabel } from '../components/format';
import { query } from '../api';
import type { LeaderboardRow, ScoreFormula } from '../types';

interface Filters {
  datasets: string[]; models: string[]; families: string[]; suites: string[];
}
interface Payload {
  rows: LeaderboardRow[];
  score_formula: ScoreFormula;
  available_formulas: ScoreFormula[];
}

type SortKey = 'rank' | 'loss' | 'accuracy' | 'gradients' | 'samples' | 'n';

export function Leaderboard() {
  // Filters live in the URL so a filtered view is a link somebody can send.
  const [params, setParams] = useSearchParams();
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'rank', desc: false });

  const dataset = params.get('dataset') ?? '';
  const model = params.get('model') ?? '';
  const family = params.get('family') ?? '';
  const suite = params.get('suite') ?? '';
  const score = params.get('score') ?? 'loss_v1';

  const filters = useApi<Filters>('/api/leaderboard/filters');
  const board = useApi<Payload>(
    `/api/leaderboard${query({ dataset, model, family, suite, score })}`,
  );

  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value); else next.delete(key);
    setParams(next, { replace: true });
  }

  const rows = useMemo(() => {
    if (!board.data) return [];
    const pick: Record<SortKey, (r: LeaderboardRow) => number> = {
      rank: (r) => r.rank,
      loss: (r) => r.final_loss.median ?? Infinity,
      accuracy: (r) => r.final_accuracy.median ?? -Infinity,
      gradients: (r) => r.gradient_count.median ?? 0,
      samples: (r) => r.database_reaches.median ?? 0,
      n: (r) => r.n_runs,
    };
    const sorted = [...board.data.rows].sort((a, b) => pick[sort.key](a) - pick[sort.key](b));
    return sort.desc ? sorted.reverse() : sorted;
  }, [board.data, sort]);

  const compareLink = useMemo(() => {
    const ids = rows.slice(0, 6).flatMap((r) => r.task_ids.slice(0, 8));
    return ids.length ? `/compare?runs=${ids.join(',')}&group_by=optimizer` : null;
  }, [rows]);

  function header(key: SortKey, label: string, numeric = true) {
    return (
      <th className={`sortable ${numeric ? 'num' : ''}`}
          onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : false }))}>
        {label}{sort.key === key ? (sort.desc ? ' ↓' : ' ↑') : ''}
      </th>
    );
  }

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1>Ranking</h1>
        <p>
          Wynik pojedynczego przebiegu jest anegdotą — metody ewolucyjne są
          stochastyczne. Tabela pokazuje medianę z kwartylami po ziarnach, a liczba
          przebiegów stoi w osobnej kolumnie, żeby było widać, na czym mediana stoi.
        </p>
      </div>

      <div className="controls section">
        <div>
          <label htmlFor="f-dataset">Zbiór danych</label>
          <select id="f-dataset" value={dataset} onChange={(e) => update('dataset', e.target.value)}>
            <option value="">wszystkie</option>
            {filters.data?.datasets.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="f-model">Model</label>
          <select id="f-model" value={model} onChange={(e) => update('model', e.target.value)}>
            <option value="">wszystkie</option>
            {filters.data?.models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="f-family">Rodzina metody</label>
          <select id="f-family" value={family} onChange={(e) => update('family', e.target.value)}>
            <option value="">wszystkie</option>
            <option value="gradient">gradientowa</option>
            <option value="gradient_free">bezgradientowa</option>
          </select>
        </div>
        <div>
          <label htmlFor="f-suite">Zestaw</label>
          <select id="f-suite" value={suite} onChange={(e) => update('suite', e.target.value)}>
            <option value="">wszystkie</option>
            <option value="test">testowy</option>
            <option value="final">finałowy</option>
          </select>
        </div>
        <div>
          <label htmlFor="f-score">Agregat</label>
          <select id="f-score" value={score} onChange={(e) => update('score', e.target.value)}>
            {board.data?.available_formulas.map((f) => (
              <option key={f.id} value={f.id}>{f.label}</option>
            ))}
          </select>
        </div>
        {compareLink && (
          <Link className="button" to={compareLink}>Porównaj czołówkę</Link>
        )}
      </div>

      {board.loading && <Loading what="ranking" />}
      {board.error && <ErrorNote error={board.error} />}

      {board.data && (
        <>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  {header('rank', '#')}
                  <th>Optymalizator</th>
                  <th>Rodzina</th>
                  <th>Zbiór</th>
                  <th>Model</th>
                  <th>Zestaw</th>
                  {header('loss', 'Strata — mediana')}
                  <th className="num">IQR straty</th>
                  {header('accuracy', 'Dokładność')}
                  {header('gradients', 'Gradienty')}
                  {header('samples', 'Próbki')}
                  <th>Powód stopu</th>
                  {header('n', 'n')}
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={`${row.optimizer}-${row.dataset}-${row.model}-${row.suite}`}>
                    <td className="num">{row.rank}</td>
                    <td className="mono">{row.optimizer}</td>
                    <td className="small">{familyLabel(row.family)}</td>
                    <td className="mono small">{row.dataset}</td>
                    <td className="mono small">{row.model}</td>
                    <td className="small">{suiteLabel(row.suite)}</td>
                    <td className="num">{decimal(row.final_loss.median)}</td>
                    <td className="num small muted">
                      {decimal(row.final_loss.q1, 3)}–{decimal(row.final_loss.q3, 3)}
                    </td>
                    <td className="num">{decimal(row.final_accuracy.median, 2)}%</td>
                    <td className="num">{integer(row.gradient_count.median)}</td>
                    <td className="num">{integer(row.database_reaches.median)}</td>
                    <td className="small mono">{row.stop_reason_mode ?? '—'}</td>
                    <td className="num">
                      {/* A median over a single run is flagged rather than
                          presented as if it were a distribution. */}
                      {row.n_runs < 3
                        ? <span title="Za mało powtórzeń, żeby mediana coś znaczyła"
                                style={{ color: 'var(--warning)' }}>{row.n_runs}!</span>
                        : row.n_runs}
                    </td>
                    <td>
                      <Link className="small"
                            to={`/compare?runs=${row.task_ids.slice(0, 12).join(',')}&group_by=run`}>
                        krzywe
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {rows.length === 0 && (
            <div className="empty">Brak wyników dla wybranych filtrów.</div>
          )}

          <div className="note note-accent" style={{ marginTop: 'var(--space-4)' }}>
            <p>
              <strong>Agregat jest wymienialny, a nie zaszyty.</strong>{' '}
              Aktywna formuła: {board.data.score_formula.label}. {board.data.score_formula.note}
            </p>
            <p className="small" style={{ marginBottom: 0 }}>
              Wspólna waluta budżetu dla metod gradientowych i bezgradientowych
              (decyzja D2) pozostaje nierozstrzygnięta i została świadomie
              oddelegowana. Do czasu jej zamknięcia żadna z dostępnych formuł nie
              miesza gradientów z próbkami — kolumny wymiarów stoją obok siebie,
              a nie są sprowadzane do jednej liczby.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
