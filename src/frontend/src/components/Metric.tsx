import type { ReactNode } from 'react';

/** One number, its label, and the context that makes it mean something.
 *
 * The context line is not decoration: "12 480 gradientów" answers nothing on
 * its own, and "12 480 / limit 100 000" answers the question the reader
 * actually has. */
export function Metric({
  label, value, context, hint,
}: {
  label: string; value: ReactNode; context?: ReactNode; hint?: string;
}) {
  return (
    <div className="metric">
      <div className="metric-label">
        {hint ? <span className="hint" title={hint}>{label}</span> : label}
      </div>
      <div className="metric-value">{value}</div>
      {context ? <div className="metric-context">{context}</div> : null}
    </div>
  );
}
