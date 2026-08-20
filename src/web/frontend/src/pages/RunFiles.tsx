import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { ErrorNote, Loading } from '../components/Empty';
import { bytes, timestamp } from '../components/format';
import type { FileEntry, Run } from '../types';

interface TreePayload {
  entries: FileEntry[];
  file_count: number;
  total_bytes: number;
  artifact_status: string | null;
  available: boolean;
  preview_limit_bytes?: number;
}
interface MetaPayload {
  path: string; name: string; size: number; modified: number;
  content_type: string; inline: boolean; preview: string | null;
  too_large: boolean; preview_limit_bytes: number;
}

interface Node {
  entry: FileEntry;
  children: Node[];
}

function buildTree(entries: FileEntry[]): Node[] {
  const nodes = new Map<string, Node>();
  for (const entry of entries) nodes.set(entry.path, { entry, children: [] });

  const roots: Node[] = [];
  for (const entry of entries) {
    const node = nodes.get(entry.path)!;
    const parentPath = entry.path.includes('/')
      ? entry.path.slice(0, entry.path.lastIndexOf('/')) : '';
    const parent = parentPath ? nodes.get(parentPath) : undefined;
    if (parent) parent.children.push(node); else roots.push(node);
  }

  const order = (list: Node[]) => {
    list.sort((a, b) => {
      if (a.entry.is_dir !== b.entry.is_dir) return a.entry.is_dir ? -1 : 1;
      return a.entry.name.localeCompare(b.entry.name);
    });
    list.forEach((node) => order(node.children));
  };
  order(roots);
  return roots;
}

function flatten(nodes: Node[], expanded: Set<string>, depth = 0): Array<Node & { depth: number }> {
  const out: Array<Node & { depth: number }> = [];
  for (const node of nodes) {
    out.push({ ...node, depth });
    if (node.entry.is_dir && expanded.has(node.entry.path)) {
      out.push(...flatten(node.children, expanded, depth + 1));
    }
  }
  return out;
}

const EMPTY_STATES: Record<string, { title: string; detail: string }> = {
  absent: {
    title: 'Artefakty jeszcze nie istnieją',
    detail: 'Pojawią się po zakończeniu obliczeń i pobraniu wyników z klastra.',
  },
  downloading: {
    title: 'Wyniki są pobierane z klastra',
    detail: 'Obliczenia się zakończyły, downloader jeszcze kopiuje pliki.',
  },
  empty: {
    title: 'Downloader nie znalazł żadnych plików',
    detail: 'Zadanie zakończyło się, ale katalog raportów na Atenie był pusty.',
  },
  error: {
    title: 'Pobieranie artefaktów zakończyło się błędem',
    detail: 'Sprawdź komunikat błędu na stronie uruchomienia.',
  },
};

export function RunFiles() {
  const { taskId } = useParams();
  // The selected file lives in the query string, so the URL of a specific
  // artifact is a link somebody can paste into an email. §11.2 requires this,
  // and it is the requirement Streamlit could not meet.
  const [params, setParams] = useSearchParams();
  const selected = params.get('path') ?? '';

  const run = useApi<Run>(`/api/runs/${taskId}`);
  const tree = useApi<TreePayload>(`/api/runs/${taskId}/files`);
  const meta = useApi<MetaPayload>(
    selected ? `/api/runs/${taskId}/files/meta?path=${encodeURIComponent(selected)}` : null,
  );

  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [content, setContent] = useState<string | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  const filterRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<HTMLPreElement>(null);

  const storageKey = `filetree:${taskId}`;

  useEffect(() => {
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      try { setExpanded(new Set(JSON.parse(stored))); } catch { /* ignore */ }
    } else if (tree.data) {
      setExpanded(new Set(tree.data.entries.filter((e) => e.is_dir).map((e) => e.path)));
    }
  }, [storageKey, tree.data]);

  const persist = useCallback((next: Set<string>) => {
    setExpanded(next);
    localStorage.setItem(storageKey, JSON.stringify([...next]));
  }, [storageKey]);

  const roots = useMemo(() => buildTree(tree.data?.entries ?? []), [tree.data]);
  const rows = useMemo(() => {
    const all = flatten(roots, expanded);
    if (!filter) return all;
    const needle = filter.toLowerCase();
    return all.filter((row) => row.entry.path.toLowerCase().includes(needle));
  }, [roots, expanded, filter]);

  const selectableRows = rows.filter((row) => !row.entry.is_dir);
  const currentIndex = rows.findIndex((row) => row.entry.path === selected);

  const select = useCallback((path: string) => {
    const next = new URLSearchParams(params);
    next.set('path', path);
    setParams(next, { replace: true });
  }, [params, setParams]);

  // Load the file body. Rendering is decided by `preview`, which the server
  // sets from the extension whitelist; the response is always data.
  useEffect(() => {
    setContent(null); setContentError(null);
    if (!selected || !meta.data || meta.data.too_large) return;
    if (meta.data.preview === 'image' || meta.data.preview === null) return;
    fetch(`/api/runs/${taskId}/files/raw?path=${encodeURIComponent(selected)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`Błąd ${response.status}`);
        return response.text();
      })
      .then(setContent)
      .catch((error) => setContentError(String(error.message ?? error)));
  }, [selected, meta.data, taskId]);

  const onKeyDown = useCallback((event: KeyboardEvent) => {
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
      if (event.key === 'Escape') { (target as HTMLInputElement).blur(); }
      return;
    }
    if (event.key === '?') { setShowHelp((v) => !v); event.preventDefault(); return; }
    if (event.key === '/') { filterRef.current?.focus(); event.preventDefault(); return; }
    if (event.key === 'y' && selected) {
      navigator.clipboard?.writeText(selected); event.preventDefault(); return;
    }
    if (event.key === 'g') { previewRef.current?.scrollTo({ top: 0 }); return; }
    if (event.key === 'G') {
      previewRef.current?.scrollTo({ top: previewRef.current.scrollHeight }); return;
    }
    if (['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight', 'Enter'].includes(event.key)) {
      event.preventDefault();
      const index = currentIndex >= 0 ? currentIndex : 0;
      const row = rows[index];
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        const step = event.key === 'ArrowDown' ? 1 : -1;
        for (let i = index + step; i >= 0 && i < rows.length; i += step) {
          if (!rows[i].entry.is_dir) { select(rows[i].entry.path); return; }
        }
        return;
      }
      if (!row) return;
      if (event.key === 'ArrowRight' && row.entry.is_dir) {
        const next = new Set(expanded); next.add(row.entry.path); persist(next);
      }
      if (event.key === 'ArrowLeft' && row.entry.is_dir) {
        const next = new Set(expanded); next.delete(row.entry.path); persist(next);
      }
      if (event.key === 'Enter' && !row.entry.is_dir) select(row.entry.path);
    }
  }, [rows, currentIndex, expanded, persist, select, selected]);

  useEffect(() => {
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onKeyDown]);

  if (tree.loading || run.loading) return <div className="page"><Loading what="pliki" /></div>;
  if (run.error) return <div className="page"><ErrorNote error={run.error} /></div>;

  const status = tree.data?.artifact_status ?? 'absent';
  const emptyState = EMPTY_STATES[status] ?? EMPTY_STATES.absent;

  return (
    <div className="page page-wide">
      <div className="breadcrumb" style={{ marginBottom: 'var(--space-3)' }}>
        <Link to="/runs">Uruchomienia</Link>
        <span className="sep">›</span>
        <Link className="mono" to={`/runs/${taskId}`}>
          {run.data?.run_name ?? taskId}
        </Link>
        <span className="sep">›</span>
        <span>Pliki</span>
      </div>

      <div className="spread" style={{ marginBottom: 'var(--space-4)' }}>
        <div>
          <h1>Artefakty uruchomienia</h1>
          <p className="small muted" style={{ marginTop: 4, marginBottom: 0 }}>
            Katalog <span className="mono">/downloads/{taskId}</span> — wyłącznie
            pliki tego przebiegu.
          </p>
        </div>
        <div className="row">
          <button className="subtle" onClick={() => setShowHelp((v) => !v)}>
            Skróty klawiszowe (?)
          </button>
          <a className="button" href={`/api/runs/${taskId}/archive.zip`}>Pobierz .zip</a>
        </div>
      </div>

      {showHelp && (
        <div className="note section">
          <strong>Skróty</strong>
          <div className="small mono" style={{ marginTop: 6, lineHeight: 1.8 }}>
            ↑ ↓ — poprzedni / następny plik · ← → — zwiń / rozwiń katalog ·
            Enter — otwórz · / — szukaj · y — kopiuj ścieżkę ·
            g / G — początek / koniec podglądu · ? — ta ściąga
          </div>
        </div>
      )}

      {!tree.data?.available || tree.data.entries.length === 0 ? (
        <div className="note note-warning">
          <strong>{emptyState.title}</strong>
          <p className="small" style={{ marginTop: 6, marginBottom: 0 }}>{emptyState.detail}</p>
          {run.data?.error_message && (
            <pre className="log" style={{ marginTop: 10, maxHeight: 160 }}>
              {run.data.error_message}
            </pre>
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'flex-start',
                      flexWrap: 'wrap' }}>
          <div className="panel" style={{ width: 300, flex: '0 0 300px', minWidth: 260 }}>
            <div className="panel-head" style={{ padding: 'var(--space-2) var(--space-3)' }}>
              <input ref={filterRef} placeholder="Filtruj (/)" value={filter}
                     onChange={(e) => setFilter(e.target.value)}
                     style={{ width: '100%', fontSize: 12 }} />
            </div>
            <div style={{ maxHeight: 560, overflow: 'auto' }}>
              {rows.map((row) => {
                const isSelected = row.entry.path === selected;
                return (
                  <div
                    key={row.entry.path}
                    role="button"
                    tabIndex={0}
                    onClick={() => {
                      if (row.entry.is_dir) {
                        const next = new Set(expanded);
                        if (next.has(row.entry.path)) next.delete(row.entry.path);
                        else next.add(row.entry.path);
                        persist(next);
                      } else select(row.entry.path);
                    }}
                    onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.click(); }}
                    className="mono"
                    style={{
                      padding: '3px var(--space-3)',
                      paddingLeft: 12 + row.depth * 14,
                      fontSize: 12,
                      cursor: 'pointer',
                      background: isSelected ? 'var(--accent-soft)' : undefined,
                      color: isSelected ? 'var(--accent)' : undefined,
                      borderLeft: isSelected ? '2px solid var(--accent)' : '2px solid transparent',
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}
                    title={row.entry.path}
                  >
                    <span className="dim" style={{ marginRight: 5 }}>
                      {row.entry.is_dir ? (expanded.has(row.entry.path) ? '▾' : '▸') : ' '}
                    </span>
                    {row.entry.name}
                    {!row.entry.is_dir && (
                      <span className="dim tiny" style={{ marginLeft: 6 }}>
                        {bytes(row.entry.size)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
            <div className="panel-head small mono" style={{
              borderTop: '1px solid var(--border)', borderBottom: 'none',
              justifyContent: 'flex-start',
            }}>
              {tree.data.file_count} plików · {bytes(tree.data.total_bytes)}
              {selectableRows.length !== tree.data.file_count && ` · ${selectableRows.length} po filtrze`}
            </div>
          </div>

          <div className="panel" style={{ flex: 1, minWidth: 340 }}>
            {!selected ? (
              <div className="panel-body">
                <div className="empty" style={{ border: 'none' }}>
                  Wybierz plik z drzewa po lewej.
                </div>
              </div>
            ) : (
              <>
                <div className="panel-head">
                  <div>
                    <div className="mono small">{selected}</div>
                    {meta.data && (
                      <div className="tiny muted mono" style={{ marginTop: 2 }}>
                        {meta.data.content_type} · {bytes(meta.data.size)} ·{' '}
                        {timestamp(new Date(meta.data.modified * 1000).toISOString())}
                      </div>
                    )}
                  </div>
                  <div className="row" style={{ gap: 'var(--space-2)' }}>
                    <button className="subtle tiny"
                            onClick={() => navigator.clipboard?.writeText(selected)}>
                      Kopiuj ścieżkę
                    </button>
                    <a className="button tiny"
                       href={`/api/runs/${taskId}/files/raw?path=${encodeURIComponent(selected)}`}
                       download>
                      Pobierz
                    </a>
                  </div>
                </div>
                <div className="panel-body">
                  {meta.error && <ErrorNote error={meta.error} />}
                  {meta.data?.too_large && (
                    <div className="note note-warning">
                      Plik przekracza limit podglądu ({bytes(meta.data.preview_limit_bytes)}).
                      Dostępny tylko do pobrania.
                    </div>
                  )}
                  {contentError && <div className="note note-error">{contentError}</div>}

                  {meta.data?.preview === 'image' && !meta.data.too_large && (
                    <img
                      src={`/api/runs/${taskId}/files/raw?path=${encodeURIComponent(selected)}`}
                      alt={selected}
                      style={{ maxWidth: '100%', border: '1px solid var(--border)' }}
                    />
                  )}

                  {content !== null && meta.data?.preview === 'table' && (
                    <CsvTable text={content} />
                  )}
                  {content !== null && meta.data?.preview === 'json' && (
                    <pre className="log">{formatJson(content)}</pre>
                  )}
                  {content !== null
                    && ['code', 'log', 'text'].includes(meta.data?.preview ?? '') && (
                    /* Rendered as a text node. React escapes children, so a file
                       whose contents are markup stays visibly markup and never
                       becomes markup. dangerouslySetInnerHTML is banned by the
                       ESLint config precisely so this cannot regress. */
                    <pre className="log" ref={previewRef}>{content}</pre>
                  )}
                  {meta.data && meta.data.preview === null && !meta.data.too_large && (
                    <div className="note">
                      Ten typ pliku nie ma podglądu. Serwowany wyłącznie jako pobranie,
                      z nagłówkiem <span className="mono">Content-Disposition: attachment</span>.
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function formatJson(text: string): string {
  try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; }
}

function CsvTable({ text }: { text: string }) {
  const lines = text.split('\n').filter((line) => line.trim().length > 0);
  const limit = 500;
  const rows = lines.slice(0, limit + 1).map((line) => line.split(','));
  if (rows.length === 0) return <div className="empty">Pusty plik.</div>;
  const [head, ...body] = rows;

  return (
    <>
      <div className="table-scroll" style={{ maxHeight: 460, overflowY: 'auto' }}>
        <table>
          <thead>
            <tr>{head.map((cell, i) => <th key={i} className="num">{cell}</th>)}</tr>
          </thead>
          <tbody>
            {body.map((row, i) => (
              <tr key={i}>{row.map((cell, j) => <td key={j} className="num">{cell}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      {lines.length > limit + 1 && (
        <div className="tiny dim" style={{ marginTop: 6 }}>
          Pokazano pierwsze {limit} z {lines.length - 1} wierszy. Pełny plik przez „Pobierz”.
        </div>
      )}
    </>
  );
}
