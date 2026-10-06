import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { suiteLabel } from '../components/format';
import { api, ApiError, withBase } from '../api';
import type { User } from '../types';

const BUILTINS = [
  { name: 'adam', family: 'gradient-based' },
  { name: 'adamw', family: 'gradient-based' },
  { name: 'lion', family: 'gradient-based' },
  { name: 'rmsprop', family: 'gradient-based' },
  { name: 'sgd', family: 'gradient-based' },
  { name: 'sgd_momentum', family: 'gradient-based' },
  { name: 'cma-es', family: 'gradient-free' },
  { name: 'de', family: 'gradient-free' },
  { name: 'des', family: 'gradient-free' },
];

interface Quota { limit: number; remaining: number; used: number }
interface Accepted {
  submission_id: string; status: string; validator_log: string;
  family: string; task_ids: string[]; remaining_today: number;
}

function parseSeeds(raw: string): number[] {
  return raw.split(',').map((s) => Number(s.trim())).filter((n) => Number.isFinite(n));
}

export function Submit({ user }: { user: User | null }) {
  const navigate = useNavigate();
  const quota = useApi<Quota>(user?.is_verified ? '/api/submissions/quota' : null);
  const filters = useApi<{ datasets: string[]; models: string[] }>('/api/runs/filters');

  const [kind, setKind] = useState<'builtin' | 'uploaded'>('builtin');
  const [builtin, setBuiltin] = useState('adam');
  const [source, setSource] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [datasetChoice, setDataset] = useState('wine_quality');
  const [modelChoice, setModel] = useState('mlp-1x16');
  const [suite, setSuite] = useState('test');
  const [seeds, setSeeds] = useState('11,23,42');
  const [maxEpochs, setMaxEpochs] = useState('12');
  const [maxGradients, setMaxGradients] = useState('');
  const [maxSamples, setMaxSamples] = useState('');

  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Accepted | null>(null);
  const [rejection, setRejection] = useState<{ log: string; message: string } | null>(null);

  // The vocabulary comes from the runs that exist, so a value chosen before it
  // arrives may be absent from it. A select whose value is not among its
  // options paints the first one and submits the other.
  const datasets = filters.data?.datasets?.length ? filters.data.datasets : ['wine_quality'];
  const models = filters.data?.models?.length ? filters.data.models : ['mlp-1x16'];
  const dataset = datasets.includes(datasetChoice) ? datasetChoice : datasets[0];
  const model = models.includes(modelChoice) ? modelChoice : models[0];

  if (!user) {
    return (
      <div className="page">
        <div className="page-head"><h1>Optimizer submission</h1></div>
        <div className="note note-accent">
          You need an approved account to submit an optimizer.{' '}
          <Link to="/login">Log in or create an account</Link>.
        </div>
      </div>
    );
  }
  if (!user.is_verified) {
    return (
      <div className="page">
        <div className="page-head"><h1>Optimizer submission</h1></div>
        <div className="note note-warning">
          Your account is awaiting administrator approval. Until then,
          you can browse the leaderboard and results but cannot submit tasks.
        </div>
      </div>
    );
  }

  const stopConditions = [
    maxEpochs && `epochs ${maxEpochs}`,
    maxGradients && `gradient evaluations ${maxGradients}`,
    maxSamples && `processed samples ${maxSamples}`,
  ].filter(Boolean).join(', ');

  const advancedSummary = [
    `tasks: ${parseSeeds(seeds).length}`,
    `suite ${suiteLabel(suite)}`,
    `stop: ${stopConditions || 'none'}`,
  ].join(' · ');

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true); setResult(null); setRejection(null);
    try {
      const payload = {
        display_name: displayName || builtin,
        kind,
        builtin_name: kind === 'builtin' ? builtin : null,
        source_code: kind === 'uploaded' ? source : null,
        dataset, model, suite,
        seeds: parseSeeds(seeds),
        max_epochs: maxEpochs ? Number(maxEpochs) : null,
        max_gradient_count: maxGradients ? Number(maxGradients) : null,
        max_database_reaches: maxSamples ? Number(maxSamples) : null,
      };
      setResult(await api.post<Accepted>('/api/submissions', payload));
      quota.reload();
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 422
          && caught.body && typeof caught.body === 'object'
          && 'validator_log' in caught.body) {
        const body = caught.body as { validator_log: string };
        setRejection({ log: body.validator_log, message: 'Validator rejected the submission.' });
      } else {
        setRejection({
          log: '', message: caught instanceof ApiError ? caught.message : String(caught),
        });
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <div className="page-head">
        <h1>Optimizer submission</h1>
        <p>
          Submitted code first runs in an isolated network-disabled container with
          a memory limit, one CPU core, and a hard
          time limit. You receive the validation result immediately, before the task consumes
          cluster time.
        </p>
      </div>

      {quota.data && (
        <div className="note section">
          Remaining today <strong className="mono">{quota.data.remaining}</strong> of{' '}
          <span className="mono">{quota.data.limit}</span> submissions.
          <span className="small muted"> Each seed counts as a separate task.</span>
        </div>
      )}

      <form onSubmit={submit}>
        <div className="panel section">
          <div className="panel-head"><h3>Optimizer</h3></div>
          <div className="panel-body">
            <div className="segmented" style={{ marginBottom: 'var(--space-3)' }}>
              <button type="button" className={kind === 'builtin' ? 'active' : ''}
                      onClick={() => setKind('builtin')}>Built-in</button>
              <button type="button" className={kind === 'uploaded' ? 'active' : ''}
                      onClick={() => setKind('uploaded')}>Custom code</button>
            </div>

            {kind === 'builtin' ? (
              <div className="field">
                <label htmlFor="s-builtin">Name</label>
                <select id="s-builtin" value={builtin} onChange={(e) => setBuiltin(e.target.value)}>
                  {BUILTINS.map((b) => (
                    <option key={b.name} value={b.name}>{b.name} ({b.family})</option>
                  ))}
                </select>
              </div>
            ) : (
              <>
                <div className="field">
                  <label htmlFor="s-name">Submission name</label>
                  <input id="s-name" style={{ width: '100%' }} value={displayName}
                         placeholder="np. sign-sgd"
                         onChange={(e) => setDisplayName(e.target.value)} required />
                </div>
                <div className="field">
                  <label htmlFor="s-source">
                    Optimizer code (.py):{' '}
                    <Link to="/docs">contract and template</Link>
                  </label>
                  <textarea id="s-source" rows={16} value={source} required
                            placeholder="class MyOptimizer(NumpyBenchmarkOptimizer): ..."
                            onChange={(e) => setSource(e.target.value)} />
                </div>
                <div className="row">
                  <label htmlFor="s-file" className="button" style={{ marginBottom: 0 }}>
                    Load from file
                  </label>
                  <input id="s-file" type="file" accept=".py" style={{ display: 'none' }}
                         onChange={async (e) => {
                           const file = e.target.files?.[0];
                           if (file) setSource(await file.text());
                         }} />
                  <a className="small" href={withBase('/api/protocol/template')} download>
                    Download template optimizer_template.py
                  </a>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="panel section">
          <div className="panel-head"><h3>Task</h3></div>
          <div className="panel-body">
            <div className="controls">
              <div>
                <label htmlFor="s-dataset">Dataset</label>
                <select id="s-dataset" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                  {datasets.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="s-model">Model</label>
                <select id="s-model" value={model} onChange={(e) => setModel(e.target.value)}>
                  {models.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>

        <div className="panel section">
          <div className="panel-head">
            <h3>
              <button type="button" className="disclosure" aria-expanded={advanced}
                      aria-controls="s-advanced" onClick={() => setAdvanced(!advanced)}>
                <span className="disclosure-marker" aria-hidden="true">{advanced ? '▾' : '▸'}</span>
                Advanced
              </button>
            </h3>
            <span className="small muted">{advancedSummary}</span>
          </div>
          <div className="panel-body" id="s-advanced" hidden={!advanced}>
            <div className="params">
              <div>
                <label htmlFor="s-suite">Suite</label>
                <select id="s-suite" value={suite} onChange={(e) => setSuite(e.target.value)}>
                  <option value="test">test</option>
                  <option value="final">final</option>
                </select>
                <p className="field-note">
                  Run label. It separates results in the leaderboard and run list
                  and does not change how the run is scored.
                </p>
              </div>
              <div>
                <label htmlFor="s-seeds">Seeds (comma-separated)</label>
                <input id="s-seeds" value={seeds} onChange={(e) => setSeeds(e.target.value)} />
                <p className="field-note">
                  Each seed is a separate run of the same optimizer
                  with a different random draw. Each seed is a separate task from the daily quota.
                </p>
              </div>
              {kind === 'builtin' && (
                <div>
                  <label htmlFor="s-display-name">Submission name</label>
                  <input id="s-display-name" value={displayName} placeholder={builtin}
                         onChange={(e) => setDisplayName(e.target.value)} />
                  <p className="field-note">
                    The run appears under this name in the leaderboard. Leave the field blank
                    to use the built-in optimizer name.
                  </p>
                </div>
              )}
              <div>
                <label htmlFor="s-epochs">Epoch budget</label>
                <input id="s-epochs" type="number" min={1} value={maxEpochs}
                       onChange={(e) => setMaxEpochs(e.target.value)} />
                <p className="field-note">
                  How many times the optimizer may pass through the full training
                  dataset before the run is stopped.
                </p>
              </div>
              <div>
                <label htmlFor="s-grads">Gradient evaluation budget</label>
                <input id="s-grads" type="number" min={1} value={maxGradients}
                       placeholder="unlimited"
                       onChange={(e) => setMaxGradients(e.target.value)} />
                <p className="field-note">
                  How many gradient evaluations the optimizer may perform. The counter increases
                  by 1 for each evaluate_with_grad() and grad().
                </p>
              </div>
              <div>
                <label htmlFor="s-samples">Processed sample budget</label>
                <input id="s-samples" type="number" min={1} value={maxSamples}
                       placeholder="unlimited"
                       onChange={(e) => setMaxSamples(e.target.value)} />
                <p className="field-note">
                  How many samples the optimizer may read from the dataset. The counter
                  increases by batch_size on each forward pass.
                </p>
              </div>
            </div>
            <p className="small muted" style={{ marginTop: 'var(--space-4)', marginBottom: 0 }}>
              At least one stop condition is required. The stop reason is included
              in the result. “converged” and “budget exhausted” are different outcomes.
            </p>
          </div>
        </div>

        <button className="primary" type="submit"
                disabled={busy || (quota.data?.remaining ?? 1) === 0}>
          {busy ? 'Validating and queueing…' : 'Submit'}
        </button>
      </form>

      {rejection && (
        <div className="note note-error section" style={{ marginTop: 'var(--space-4)' }}>
          <strong>{rejection.message}</strong>
          {rejection.log && <pre className="log" style={{ marginTop: 10 }}>{rejection.log}</pre>}
        </div>
      )}

      {result && (
        <div className="section" style={{ marginTop: 'var(--space-4)' }}>
          <div className="note note-accent">
            <strong>Submission accepted.</strong> Created{' '}
            <span className="mono">{result.task_ids.length}</span> tasks.
            Remaining today <span className="mono">{result.remaining_today}</span>.
          </div>
          <div className="panel" style={{ marginTop: 'var(--space-3)' }}>
            <div className="panel-head"><h3>Validator log</h3></div>
            <div className="panel-body"><pre className="log">{result.validator_log}</pre></div>
          </div>
          <div className="row" style={{ marginTop: 'var(--space-3)' }}>
            <button onClick={() => navigate('/runs?mine=1')}>View my runs</button>
            {result.task_ids[0] && (
              <Link className="button" to={`/runs/${result.task_ids[0]}`}>
                First task
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
