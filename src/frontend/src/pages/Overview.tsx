import { Link } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { Metric } from '../components/Metric';
import { ErrorNote, Loading } from '../components/Empty';
import { decimal, duration, familyLabel, integer } from '../components/format';
import type { LeaderboardRow, ScoreFormula } from '../types';

interface Counters {
  submissions: number; runs: number; completed: number; failed: number;
  active: number; participants: number; datasets: number; optimizers: number;
  compute_seconds: number; gradients: number; samples: number;
}

export function Overview() {
  const counters = useApi<Counters>('/api/overview');
  const board = useApi<{ rows: LeaderboardRow[]; score_formula: ScoreFormula }>(
    '/api/leaderboard?limit=10',
  );

  return (
    <div className="page">
      <div className="page-head">
        <h1>Benchmark Czarnej Skrzynki</h1>
        <p>
          Uczestnik zgłasza wyłącznie matematyczną definicję optymalizatora i nigdy
          nie widzi ani modelu, ani zbioru danych. Sieć jest sprowadzona do płaskiego
          wektora parametrów i funkcji celu, a zużycie budżetu liczy ewaluator po
          stronie systemu — nie zgłoszenie.
        </p>
        <p>
          Dzięki temu metoda gradientowa i ewolucyjna podłączają się do tego samego
          interfejsu i dają się porównać na wspólnej osi budżetu.{' '}
          <Link to="/docs">Zobacz kontrakt optymalizatora</Link>.
        </p>
      </div>

      {counters.error && <ErrorNote error={counters.error} />}
      {counters.data && (
        <div className="section">
          <div className="metrics">
            <Metric label="Zgłoszenia" value={integer(counters.data.submissions)}
                    context={`${integer(counters.data.optimizers)} optymalizatorów`} />
            <Metric label="Uruchomienia" value={integer(counters.data.runs)}
                    context={`${integer(counters.data.completed)} zakończonych, ${integer(counters.data.failed)} nieudanych`} />
            <Metric label="W toku" value={integer(counters.data.active)}
                    context="w kolejce lub liczące się" />
            <Metric label="Wyliczone gradienty" value={integer(counters.data.gradients)}
                    hint="Liczba obliczeń gradientu. Rośnie o 1 przy każdym evaluate_with_grad() i grad()." />
            <Metric label="Przetworzone próbki" value={integer(counters.data.samples)}
                    hint="Liczba próbek pobranych ze zbioru danych. Rośnie o batch_size przy każdym przejściu w przód." />
            <Metric label="Czas obliczeń" value={duration(counters.data.compute_seconds)}
                    context="świadomie nie jest metryką rankingu" />
          </div>
        </div>
      )}

      <div className="section">
        <div className="section-head">
          <h2>Ranking — pierwsza dziesiątka</h2>
          <Link to="/leaderboard" className="small">Pełna tabela i filtry →</Link>
        </div>
        {board.loading && <Loading what="ranking" />}
        {board.error && <ErrorNote error={board.error} />}
        {board.data && board.data.rows.length === 0 && (
          <div className="empty">Nie ma jeszcze zakończonych uruchomień.</div>
        )}
        {board.data && board.data.rows.length > 0 && (
          <>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th className="num">#</th>
                    <th>Optymalizator</th>
                    <th>Rodzina</th>
                    <th>Zbiór</th>
                    <th>Model</th>
                    <th className="num">Strata (mediana)</th>
                    <th className="num">Dokładność</th>
                    <th className="num">Gradienty</th>
                    <th className="num">Próbki</th>
                    <th className="num">n</th>
                  </tr>
                </thead>
                <tbody>
                  {board.data.rows.map((row) => (
                    <tr key={`${row.optimizer}-${row.dataset}-${row.model}-${row.suite}`}>
                      <td className="num">{row.rank}</td>
                      <td className="mono">{row.optimizer}</td>
                      <td className="small">{familyLabel(row.family)}</td>
                      <td className="mono small">{row.dataset}</td>
                      <td className="mono small">{row.model}</td>
                      <td className="num">{decimal(row.final_loss.median)}</td>
                      <td className="num">{decimal(row.final_accuracy.median, 2)}%</td>
                      <td className="num">{integer(row.gradient_count.median)}</td>
                      <td className="num">{integer(row.database_reaches.median)}</td>
                      <td className="num">{row.n_runs}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="note" style={{ marginTop: 'var(--space-3)' }}>
              Agregat: <strong>{board.data.score_formula.label}</strong>.{' '}
              {board.data.score_formula.note}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
