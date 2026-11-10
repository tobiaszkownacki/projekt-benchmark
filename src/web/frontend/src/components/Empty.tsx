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

export function RunNotFound() {
  return (
    <div className="page">
      <div className="note note-error">
        <strong>Run not found or you do not have access.</strong>
        <p className="small" style={{ marginTop: 6, marginBottom: 0 }}>
          Results are visible to the owner and administrators. We return 404 instead of 403
          so we do not confirm that someone else’s resource exists.
        </p>
      </div>
    </div>
  );
}

// The archive endpoint answers 404 until the run's files are on disk, so a
// plain link would open a bare JSON error for queued, failed or running runs.
export function ArchiveLink({ href, status }: { href: string; status: string | null | undefined }) {
  if (status === 'ready') {
    return <a className="button" href={href}>Download .zip</a>;
  }
  const reason = status === 'downloading' ? 'Files are still downloading' : 'No files for this run';
  return <span className="small muted">{reason}</span>;
}
