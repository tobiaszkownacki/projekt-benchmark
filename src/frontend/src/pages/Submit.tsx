import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api, ApiError } from '../api';
import { useApi } from '../hooks/useApi';
import type { User } from '../types';

const BUILTINS = [
  { name: 'adam', family: 'gradientowa' },
  { name: 'adamw', family: 'gradientowa' },
  { name: 'lion', family: 'gradientowa' },
  { name: 'rmsprop', family: 'gradientowa' },
  { name: 'sgd', family: 'gradientowa' },
  { name: 'sgd_momentum', family: 'gradientowa' },
  { name: 'cma-es', family: 'bezgradientowa' },
  { name: 'de', family: 'bezgradientowa' },
  { name: 'des', family: 'bezgradientowa' },
];

interface Quota { limit: number; remaining: number; used: number }
interface Accepted {
  submission_id: string; status: string; validator_log: string;
  family: string; task_ids: string[]; remaining_today: number;
}

export function Submit({ user }: { user: User | null }) {
  const navigate = useNavigate();
  const quota = useApi<Quota>(user?.is_verified ? '/api/submissions/quota' : null);
  const filters = useApi<{ datasets: string[]; models: string[] }>('/api/runs/filters');

  const [kind, setKind] = useState<'builtin' | 'uploaded'>('builtin');
  const [builtin, setBuiltin] = useState('adam');
  const [source, setSource] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [dataset, setDataset] = useState('wine');
  const [model, setModel] = useState('mlp-1x16');
  const [suite, setSuite] = useState('test');
  const [seeds, setSeeds] = useState('11,23,42');
  const [maxEpochs, setMaxEpochs] = useState('12');
  const [maxGradients, setMaxGradients] = useState('');
  const [maxSamples, setMaxSamples] = useState('');

  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Accepted | null>(null);
  const [rejection, setRejection] = useState<{ log: string; message: string } | null>(null);

  if (!user) {
    return (
      <div className="page">
        <div className="page-head"><h1>Zgłoszenie optymalizatora</h1></div>
        <div className="note note-accent">
          Żeby zgłosić optymalizator, trzeba mieć zatwierdzone konto.{' '}
          <Link to="/login">Zaloguj się lub załóż konto</Link>.
        </div>
      </div>
    );
  }
  if (!user.is_verified) {
    return (
      <div className="page">
        <div className="page-head"><h1>Zgłoszenie optymalizatora</h1></div>
        <div className="note note-warning">
          Twoje konto czeka na zatwierdzenie przez administratora. Do tego czasu
          możesz przeglądać ranking i wyniki, ale nie zgłaszać zadań.
        </div>
      </div>
    );
  }

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
        seeds: seeds.split(',').map((s) => Number(s.trim())).filter((n) => Number.isFinite(n)),
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
        setRejection({ log: body.validator_log, message: 'Walidator odrzucił zgłoszenie.' });
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
        <h1>Zgłoszenie optymalizatora</h1>
        <p>
          Zgłoszony kod jest najpierw uruchamiany w izolowanym kontenerze bez
          dostępu do sieci, z limitem pamięci, jednym rdzeniem i twardym limitem
          czasu. Wynik walidacji dostajesz od razu — zanim zadanie zajmie czas
          klastra.
        </p>
      </div>

      {quota.data && (
        <div className="note section">
          Pozostało dziś <strong className="mono">{quota.data.remaining}</strong> z{' '}
          <span className="mono">{quota.data.limit}</span> zgłoszeń.
          <span className="small muted"> Każde ziarno liczy się jako osobne zadanie.</span>
        </div>
      )}

      <form onSubmit={submit}>
        <div className="panel section">
          <div className="panel-head"><h3>Optymalizator</h3></div>
          <div className="panel-body">
            <div className="segmented" style={{ marginBottom: 'var(--space-3)' }}>
              <button type="button" className={kind === 'builtin' ? 'active' : ''}
                      onClick={() => setKind('builtin')}>Wbudowany</button>
              <button type="button" className={kind === 'uploaded' ? 'active' : ''}
                      onClick={() => setKind('uploaded')}>Własny kod</button>
            </div>

            {kind === 'builtin' ? (
              <div className="field">
                <label htmlFor="s-builtin">Nazwa</label>
                <select id="s-builtin" value={builtin} onChange={(e) => setBuiltin(e.target.value)}>
                  {BUILTINS.map((b) => (
                    <option key={b.name} value={b.name}>{b.name} — {b.family}</option>
                  ))}
                </select>
              </div>
            ) : (
              <>
                <div className="field">
                  <label htmlFor="s-name">Nazwa zgłoszenia</label>
                  <input id="s-name" style={{ width: '100%' }} value={displayName}
                         placeholder="np. sign-sgd"
                         onChange={(e) => setDisplayName(e.target.value)} required />
                </div>
                <div className="field">
                  <label htmlFor="s-source">
                    Kod optymalizatora (.py) —{' '}
                    <Link to="/docs">kontrakt i szablon</Link>
                  </label>
                  <textarea id="s-source" rows={16} value={source} required
                            placeholder="class MyOptimizer(NumpyBenchmarkOptimizer): ..."
                            onChange={(e) => setSource(e.target.value)} />
                </div>
                <div className="row">
                  <label htmlFor="s-file" className="button" style={{ marginBottom: 0 }}>
                    Wczytaj z pliku
                  </label>
                  <input id="s-file" type="file" accept=".py" style={{ display: 'none' }}
                         onChange={async (e) => {
                           const file = e.target.files?.[0];
                           if (file) setSource(await file.text());
                         }} />
                  <a className="small" href="/api/protocol/template" download>
                    Pobierz szablon optimizer_template.py
                  </a>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="panel section">
          <div className="panel-head"><h3>Zadanie i budżet</h3></div>
          <div className="panel-body">
            <div className="controls">
              <div>
                <label htmlFor="s-dataset">Zbiór danych</label>
                <select id="s-dataset" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                  {(filters.data?.datasets ?? ['wine']).map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="s-model">Model</label>
                <select id="s-model" value={model} onChange={(e) => setModel(e.target.value)}>
                  {(filters.data?.models ?? ['mlp-1x16']).map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="s-suite">Zestaw</label>
                <select id="s-suite" value={suite} onChange={(e) => setSuite(e.target.value)}>
                  <option value="test">testowy</option>
                  <option value="final">finałowy</option>
                </select>
              </div>
              <div>
                <label htmlFor="s-seeds">Ziarna (po przecinku)</label>
                <input id="s-seeds" value={seeds} onChange={(e) => setSeeds(e.target.value)} />
              </div>
            </div>
            <div className="controls" style={{ marginTop: 'var(--space-3)' }}>
              <div>
                <label htmlFor="s-epochs">Limit epok</label>
                <input id="s-epochs" type="number" min={1} value={maxEpochs}
                       onChange={(e) => setMaxEpochs(e.target.value)} style={{ width: 120 }} />
              </div>
              <div>
                <label htmlFor="s-grads">Limit gradientów</label>
                <input id="s-grads" type="number" min={1} value={maxGradients}
                       placeholder="bez limitu" style={{ width: 140 }}
                       onChange={(e) => setMaxGradients(e.target.value)} />
              </div>
              <div>
                <label htmlFor="s-samples">Limit próbek</label>
                <input id="s-samples" type="number" min={1} value={maxSamples}
                       placeholder="bez limitu" style={{ width: 140 }}
                       onChange={(e) => setMaxSamples(e.target.value)} />
              </div>
            </div>
            <p className="small muted" style={{ marginTop: 'var(--space-3)', marginBottom: 0 }}>
              Wymagany co najmniej jeden warunek stopu. Powód zatrzymania trafia
              do wyniku — „zbiegłem” i „wyczerpałem budżet” to zupełnie różne rezultaty.
            </p>
          </div>
        </div>

        <button className="primary" type="submit"
                disabled={busy || (quota.data?.remaining ?? 1) === 0}>
          {busy ? 'Walidacja i kolejkowanie…' : 'Zgłoś'}
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
            <strong>Zgłoszenie przyjęte.</strong> Utworzono{' '}
            <span className="mono">{result.task_ids.length}</span> zadań.
            Pozostało dziś <span className="mono">{result.remaining_today}</span>.
          </div>
          <div className="panel" style={{ marginTop: 'var(--space-3)' }}>
            <div className="panel-head"><h3>Log walidatora</h3></div>
            <div className="panel-body"><pre className="log">{result.validator_log}</pre></div>
          </div>
          <div className="row" style={{ marginTop: 'var(--space-3)' }}>
            <button onClick={() => navigate('/runs?mine=1')}>Zobacz moje uruchomienia</button>
            {result.task_ids[0] && (
              <Link className="button" to={`/runs/${result.task_ids[0]}`}>
                Pierwsze zadanie
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
