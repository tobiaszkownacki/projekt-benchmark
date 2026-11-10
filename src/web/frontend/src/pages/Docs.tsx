import { useEffect, useState } from 'react';
import { useApi } from '../hooks/useApi';
import { Loading } from '../components/Empty';
import { withBase } from '../api';

interface ApiRow { method: string; returns: string; effect: string; note: string }
interface Protocol {
  evaluator_api: ApiRow[];
  examples: Record<string, { filename: string; present: boolean; repo_path: string | null }>;
  stop_reasons: Record<string, { label: string; note: string; converged: boolean }>;
}

function CodeBlock({ name, label }: { name: string; label: string }) {
  const [code, setCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch(withBase(`/api/protocol/example/${name}`))
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then(setCode)
      .catch(() => setCode(null));
  }, [name]);

  if (code === null) {
    return (
      <div className="note note-warning">
        Could not load the example <span className="mono">{label}</span> from the repository.
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="mono small">{label}</span>
        <button className="subtle tiny" onClick={() => {
          navigator.clipboard?.writeText(code);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      {/* Source arrives as text and is rendered as a text node. */}
      <pre className="log" style={{ border: 'none', maxHeight: 520 }}>{code}</pre>
    </div>
  );
}

export function Docs() {
  const protocol = useApi<Protocol>('/api/protocol');
  if (protocol.loading) return <div className="page"><Loading what="documentation" /></div>;
  const data = protocol.data;

  return (
    <div className="page">
      <div className="page-head">
        <h1>Optimizer contract</h1>
        <p>
          You only need to write a simple Python class that implements the{' '}
          <span className="mono">step()</span> method. 
          Only NumPy/CuPy library knowledge is required. Use NumPy for CPU execution or CuPy for GPU acceleration.
        </p>
      </div>

      <div className="section">
        <div className="section-head"><h2>The method you implement</h2></div>
        <pre className="log">{`class MyOptimizer(NumpyBenchmarkOptimizer):

    def __init__(self, initial_params, **config):
        # The first positional argument MUST be named initial_params.
        ...

    def step(self, evaluator: ModelEvaluator) -> bool:
        ...              # any logic
        return False     # True = converged, stop`}</pre>
        <p className="small muted" style={{ marginTop: 'var(--space-3)' }}>
          Inheritance is optional. <span className="mono">BenchmarkableOptimizer</span>{' '} is a <span className="mono">Protocol</span> with{' '}
          <span className="mono">@runtime_checkable</span>, so a compatible method needs <span className="mono">step()</span> and{' '}
          <span className="mono">get_output_type()</span>. The runner supplies the model's initialized weights as the flat{' '}
          <span className="mono">initial_params</span> vector. The optimizer owns its parameter vectors from then on.
        </p>
      </div>

      <div className="section">
        <div className="section-head"><h2>Evaluator API</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Method</th><th>Returns</th>
                <th>Budget counter effect</th><th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {data?.evaluator_api.map((row) => (
                <tr key={row.method}>
                  <td className="mono">{row.method}</td>
                  <td className="mono small">{row.returns}</td>
                  <td className="mono small">{row.effect}</td>
                  <td className="small muted">{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note" style={{ marginTop: 'var(--space-3)' }}>
          Pass the parameter vector you want to test directly to each evaluator method.
          No separate parameter synchronization call is needed.
        </div>
      </div>

      <div className="section">
        <div className="section-head"><h2>Stop conditions and reasons</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr><th>StopReason</th><th>In the interface</th><th>Meaning</th></tr>
            </thead>
            <tbody>
              {Object.entries(data?.stop_reasons ?? {}).map(([key, value]) => (
                <tr key={key}>
                  <td className="mono small">{key}</td>
                  <td>{value.label}</td>
                  <td className="small muted">{value.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <div className="section-head">
          <h2>Complete examples</h2>
          <a className="small" href={withBase('/api/protocol/template')} download>
            Download template
          </a>
        </div>
        <p className="small muted">
          These examples demonstrate explicit candidate evaluation for gradient and non-gradient optimizers.
        </p>
        <div className="stack">
          <CodeBlock name="gradient" label="optimizer_protocols/example_gradient_optimizer.py" />
          <CodeBlock name="evolutionary" label="optimizer_protocols/example_evolutionary_optimizer.py" />
        </div>
      </div>
    </div>
  );
}