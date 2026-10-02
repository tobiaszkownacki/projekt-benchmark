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
        <h1>Black-Box Benchmark</h1>
        <p>
          Participants submit only a mathematical definition of an optimizer and never
          see the model or dataset. The network is reduced to a flat
          parameter vector and objective function, while the evaluator tracks budget use on
          the system side rather than the submission.
        </p>
        <p>
          This lets gradient-based and evolutionary methods use the same
          interface and be compared on a shared budget axis.{' '}
          <Link to="/docs">See the optimizer contract</Link>.
        </p>
      </div>

      {counters.error && <ErrorNote error={counters.error} />}
      {counters.data && (
        <div className="section">
          <div className="metrics">
            <Metric label="Submissions" value={integer(counters.data.submissions)}
                    context={`${integer(counters.data.optimizers)} optimizers`} />
            <Metric label="Runs" value={integer(counters.data.runs)}
                    context={`${integer(counters.data.completed)} completed, ${integer(counters.data.failed)} failed`} />
            <Metric label="Active" value={integer(counters.data.active)}
                    context="queued or running" />
            <Metric label="Gradient evaluations" value={integer(counters.data.gradients)}
                    hint="Number of gradient evaluations. Increases by 1 for each evaluate_with_grad() and grad()." />
            <Metric label="Processed samples" value={integer(counters.data.samples)}
                    hint="Number of samples read from the dataset. Increases by batch_size on each forward pass." />
            <Metric label="Compute time" value={duration(counters.data.compute_seconds)}
                    context="intentionally excluded from leaderboard metrics" />
          </div>
        </div>
      )}

      <div className="section">
        <div className="section-head">
          <h2>Leaderboard — top ten</h2>
          <Link to="/leaderboard" className="small">Full table and filters →</Link>
        </div>
        {board.loading && <Loading what="leaderboard" />}
        {board.error && <ErrorNote error={board.error} />}
        {board.data && board.data.rows.length === 0 && (
          <div className="empty">No completed runs yet.</div>
        )}
        {board.data && board.data.rows.length > 0 && (
          <>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th className="num">#</th>
                    <th>Optimizer</th>
                    <th>Family</th>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th className="num">Final loss (median)</th>
                    <th className="num">Accuracy</th>
                    <th className="num">Gradient evaluations</th>
                    <th className="num">Processed samples</th>
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
              Aggregate: <strong>{board.data.score_formula.label}</strong>.{' '}
              {board.data.score_formula.note}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
