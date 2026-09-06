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
        Nie udało się wczytać przykładu <span className="mono">{label}</span> z repozytorium.
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
          {copied ? 'Skopiowano' : 'Kopiuj'}
        </button>
      </div>
      {/* Source arrives as text and is rendered as a text node. */}
      <pre className="log" style={{ border: 'none', maxHeight: 520 }}>{code}</pre>
    </div>
  );
}

export function Docs() {
  const protocol = useApi<Protocol>('/api/protocol');
  if (protocol.loading) return <div className="page"><Loading what="dokumentację" /></div>;
  const data = protocol.data;

  return (
    <div className="page">
      <div className="page-head">
        <h1>Kontrakt optymalizatora</h1>
        <p>
          Piszesz wyłącznie matematyczną definicję optymalizatora. Nie widzisz
          modelu ani zbioru danych — sieć jest sprowadzona do płaskiego wektora
          parametrów i funkcji celu. Dzięki temu metoda gradientowa i ewolucyjna
          podłączają się do tego samego interfejsu, a zużycia budżetu nie da się
          zaraportować samemu: liczy je ewaluator.
        </p>
      </div>

      <div className="section">
        <div className="section-head"><h2>Metoda, którą implementujesz</h2></div>
        <pre className="log">{`class MyOptimizer(NumpyBenchmarkOptimizer):

    def __init__(self, initial_params, **config):
        # Pierwszy argument pozycyjny MUSI nazywać się initial_params.
        ...

    def step(self, evaluator: ModelEvaluator) -> bool:
        ...              # dowolna logika
        return False     # True = zbiegłem, kończ`}</pre>
        <p className="small muted" style={{ marginTop: 'var(--space-3)' }}>
          Dziedziczenie jest opcjonalne — <span className="mono">BenchmarkableOptimizer</span>{' '}
          to <span className="mono">Protocol</span> z{' '}
          <span className="mono">@runtime_checkable</span>, więc wystarczy zgodna
          metoda <span className="mono">step()</span> i klasowa{' '}
          <span className="mono">get_output_type()</span>.
        </p>
      </div>

      <div className="section">
        <div className="section-head"><h2>API ewaluatora</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Metoda</th><th>Zwraca</th>
                <th>Efekt na licznikach budżetu</th><th>Uwagi</th>
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
          Kolumna z efektem na licznikach jest tu najważniejsza. To jedyne miejsce,
          w którym widać, ile kosztuje każde wywołanie — a licznik prowadzi system,
          nie Twój kod.
        </div>
      </div>

      <div className="section">
        <div className="section-head"><h2>Warunki i powody zatrzymania</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr><th>StopReason</th><th>W interfejsie</th><th>Znaczenie</th></tr>
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
          <h2>Kompletne przykłady</h2>
          <a className="small" href="/api/protocol/template" download>
            Pobierz szablon
          </a>
        </div>
        <p className="small muted">
          Wczytywane bezpośrednio z plików w repozytorium, a nie skopiowane do tej
          strony. Jeśli przykład w kodzie się zmieni, zmieni się też tutaj.
        </p>
        <div className="stack">
          <CodeBlock name="gradient" label="optimizer_protocols/example_gradient_optimizer.py" />
          <CodeBlock name="evolutionary" label="optimizer_protocols/example_evolutionary_optimizer.py" />
        </div>
      </div>

      <div className="section">
        <div className="section-head"><h2>Sandbox walidatora</h2></div>
        <p>
          Zgłoszony kod jest uruchamiany w izolowanym kontenerze zanim trafi do
          kolejki. Powód jest praktyczny: zadanie, które wywali się na klastrze,
          zużywa czas grantu, a ten jest zasobem ograniczonym.
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
        <div className="section-head"><h2>Znane ograniczenia kontraktu</h2></div>
        <p className="small muted">
          Wypisane wprost, bo znalezienie ich metodą prób i błędów w trakcie konkursu
          byłoby gorsze dla wszystkich niż przeczytanie ich tutaj.
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
