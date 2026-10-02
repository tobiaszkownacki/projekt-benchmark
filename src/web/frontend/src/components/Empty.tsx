import type { ReactNode } from 'react';

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div style={{ fontWeight: 600, color: 'var(--text)' }}>{title}</div>
      {children ? <div className="small" style={{ marginTop: 6 }}>{children}</div> : null}
    </div>
  );
}

export function Loading({ what = 'data' }: { what?: string }) {
  return <div className="empty">Loading: {what}…</div>;
}

export function ErrorNote({ error }: { error: Error | null }) {
  if (!error) return null;
  return (
    <div className="note note-error">
      <strong>Could not load data.</strong>
      <div className="small mono" style={{ marginTop: 4 }}>{error.message}</div>
    </div>
  );
}
