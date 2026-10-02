import { useEffect, useState } from 'react';
import { useApi } from '../hooks/useApi';
import { Loading } from '../components/Empty';

interface ApiRow { method: string; returns: string; effect: string; note: string }
interface Limitation { title: string; detail: string }
interface Protocol {
  evaluator_api: ApiRow[];
  examples: Record<string, { filename: string; present: boolean; repo_path: string | null }>;
  known_limitations: Limitation[];
  sandbox: Record<string, string>;
  stop_reasons: Record<string, { label: string; note: string; converged: boolean }>;
}

function CodeBlock({ name, label }: { name: string; label: string }) {
  const [code, setCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch(`/api/protocol/example/${name}`)
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
          You write only the mathematical definition of an optimizer. You do not see
          the model or dataset — the network is reduced to a flat
          parameter vector and objective function. This lets gradient-based and evolutionary
          methods use the same interface, and budget use cannot be
          reported by the optimizer: the evaluator counts it.
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
          Inheritance is optional — <span className="mono">BenchmarkableOptimizer</span>{' '}
          is a <span className="mono">Protocol</span> with{' '}
          <span className="mono">@runtime_checkable</span>, so a compatible
          method needs <span className="mono">step()</span> and{' '}
          <span className="mono">get_output_type()</span>.
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
          The counter-effect column is the most important part. It is the only place
          where you can see the cost of each call — the counter is maintained by the system,
          not your code.
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
          <a className="small" href="/api/protocol/template" download>
            Download template
          </a>
        </div>
        <p className="small muted">
          Loaded directly from files in the repository rather than copied into this
          page. If an example changes in the code, it changes here too.
        </p>
        <div className="stack">
          <CodeBlock name="gradient" label="optimizer_protocols/example_gradient_optimizer.py" />
          <CodeBlock name="evolutionary" label="optimizer_protocols/example_evolutionary_optimizer.py" />
        </div>
      </div>

      <div className="section">
        <div className="section-head"><h2>Validator sandbox</h2></div>
        <p>
          Submitted code runs in an isolated container before it enters the
          queue. The reason is practical: a task that fails on the cluster
          consumes grant time, which is a limited resource.
        </p>
        <dl className="kv">
          {Object.entries(data?.sandbox ?? {}).map(([key, value]) => (
            <div key={key} style={{ display: 'contents' }}>
              <dt>{key}</dt><dd>{value}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="section">
        <div className="section-head"><h2>Known contract limitations</h2></div>
        <p className="small muted">
          Listed explicitly because discovering them by trial and error during the competition
          would be worse for everyone than reading them here.
        </p>
        <div className="stack">
          {data?.known_limitations.map((item) => (
            <div key={item.title} className="note note-warning">
              <strong>{item.title}</strong>
              <p className="small" style={{ marginTop: 4, marginBottom: 0 }}>{item.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
