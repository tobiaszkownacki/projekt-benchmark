import { useMemo, useRef, useState } from 'react';
import type { AggregatedSeries } from '../types';
import { decimal, integer } from './format';

/* A convergence chart drawn directly as SVG.
 *
 * Written rather than pulled in, for three reasons that all point the same way.
 * The plan's own constraint is to add no runtime dependency that is not
 * load-bearing, because whoever maintains this in three years is not in the team
 * today. The band, the step-wise median and the "stop drawing a full band where
 * the runs ran out" rule are specific enough that a general chart library has to
 * be bent into them anyway. And §9.3 wants SVG export of exactly what is on
 * screen -- which is free when the thing on screen is already SVG, and a second
 * rendering path otherwise.
 *
 * Series are downsampled server-side to about a thousand points, and each one
 * becomes a single <path>, so the element count stays flat regardless of the
 * data. */

const PALETTE = [
  'var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)',
  'var(--series-5)', 'var(--series-6)', 'var(--series-7)', 'var(--series-8)',
];
// Series are distinguished twice: by colour, and by dash pattern. Colour is
// what disappears when these figures are printed in black and white.
const DASHES = ['', '6 3', '2 3', '8 3 2 3', '4 2', '1 3', '10 4', '3 3 1 3'];

const PADDING = { top: 16, right: 18, bottom: 42, left: 66 };

interface Props {
  series: AggregatedSeries[];
  xLabel: string;
  yLabel: string;
  logY?: boolean;
  logX?: boolean;
  height?: number;
  showBand?: boolean;
}

type Scale = (value: number) => number;

function makeScale(min: number, max: number, from: number, to: number, log: boolean): Scale {
  if (log) {
    const lo = Math.log10(Math.max(min, 1e-12));
    const hi = Math.log10(Math.max(max, 1e-11));
    const span = hi - lo || 1;
    return (value) => from + ((Math.log10(Math.max(value, 1e-12)) - lo) / span) * (to - from);
  }
  const span = max - min || 1;
  return (value) => from + ((value - min) / span) * (to - from);
}

function ticks(min: number, max: number, count: number, log: boolean): number[] {
  if (log) {
    const lo = Math.floor(Math.log10(Math.max(min, 1e-12)));
    const hi = Math.ceil(Math.log10(Math.max(max, 1e-11)));
    const out: number[] = [];
    for (let power = lo; power <= hi; power += 1) out.push(10 ** power);
    return out.filter((v) => v >= min * 0.99 && v <= max * 1.01);
  }
  const span = max - min || 1;
  const rough = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ?? magnitude * 10;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let value = start; value <= max + step * 0.001; value += step) out.push(value);
  return out;
}

function tickLabel(value: number, log: boolean): string {
  if (log) {
    const power = Math.round(Math.log10(value));
    if (power >= -2 && power <= 4) return String(10 ** power);
    return `1e${power}`;
  }
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(abs >= 10_000 ? 0 : 1)}k`;
  if (abs >= 1) return value.toFixed(abs < 10 ? 1 : 0);
  return value.toPrecision(2);
}

export function Chart({
  series, xLabel, yLabel, logY = false, logX = false, height = 380, showBand = true,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<{ x: number; y: number; index: number } | null>(null);
  const [zoom, setZoom] = useState<[number, number] | null>(null);
  const [drag, setDrag] = useState<{ from: number; to: number } | null>(null);

  const width = 900;
  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = height - PADDING.top - PADDING.bottom;

  const visible = useMemo(
    () => series.filter((s) => !hidden.has(s.label) && s.x.length > 0),
    [series, hidden],
  );

  const domain = useMemo(() => {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const entry of visible) {
      for (let i = 0; i < entry.x.length; i += 1) {
        const x = entry.x[i];
        if (zoom && (x < zoom[0] || x > zoom[1])) continue;
        if (logX && x <= 0) continue;
        xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
        for (const track of [entry.median, showBand ? entry.q1 : [], showBand ? entry.q3 : []]) {
          const value = track[i];
          if (value === null || value === undefined) continue;
          if (logY && value <= 0) continue;
          yMin = Math.min(yMin, value); yMax = Math.max(yMax, value);
        }
      }
    }
    if (!Number.isFinite(xMin)) return null;
    if (yMin === yMax) { yMin *= 0.95; yMax *= 1.05; }
    if (!logY) {
      const pad = (yMax - yMin) * 0.06;
      yMin -= pad; yMax += pad;
      if (yMin < 0 && series.every((s) => s.median.every((v) => v === null || v >= 0))) yMin = 0;
    }
    return { xMin, xMax, yMin, yMax };
  }, [visible, zoom, logX, logY, showBand, series]);

  if (!domain) {
    return <div className="empty">Brak danych do narysowania.</div>;
  }

  const scaleX = makeScale(domain.xMin, domain.xMax, PADDING.left, PADDING.left + plotWidth, logX);
  const scaleY = makeScale(domain.yMin, domain.yMax, PADDING.top + plotHeight, PADDING.top, logY);

  const inView = (x: number) => !zoom || (x >= zoom[0] && x <= zoom[1]);

  function linePath(entry: AggregatedSeries, from: number, to: number): string {
    const parts: string[] = [];
    let open = false;
    for (let i = from; i <= to && i < entry.x.length; i += 1) {
      const value = entry.median[i];
      if (value === null || !inView(entry.x[i]) || (logY && value <= 0)) { open = false; continue; }
      parts.push(`${open ? 'L' : 'M'}${scaleX(entry.x[i]).toFixed(2)} ${scaleY(value).toFixed(2)}`);
      open = true;
    }
    return parts.join(' ');
  }

  function bandPath(entry: AggregatedSeries, from: number, to: number): string {
    const upper: string[] = [];
    const lower: string[] = [];
    for (let i = from; i <= to && i < entry.x.length; i += 1) {
      const high = entry.q3[i]; const low = entry.q1[i];
      if (high === null || low === null || !inView(entry.x[i])) continue;
      if (logY && (high <= 0 || low <= 0)) continue;
      upper.push(`${upper.length ? 'L' : 'M'}${scaleX(entry.x[i]).toFixed(2)} ${scaleY(high).toFixed(2)}`);
      lower.push(`L${scaleX(entry.x[i]).toFixed(2)} ${scaleY(low).toFixed(2)}`);
    }
    if (!upper.length) return '';
    return `${upper.join(' ')} ${lower.reverse().join(' ')} Z`;
  }

  const xTicks = ticks(domain.xMin, domain.xMax, 7, logX);
  const yTicks = ticks(domain.yMin, domain.yMax, 6, logY);

  function pointerIndex(event: React.MouseEvent<SVGSVGElement>): number | null {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * width;
    if (px < PADDING.left || px > PADDING.left + plotWidth) return null;
    const reference = visible[0];
    if (!reference) return null;
    let best = 0; let bestDistance = Infinity;
    for (let i = 0; i < reference.x.length; i += 1) {
      if (!inView(reference.x[i])) continue;
      const distance = Math.abs(scaleX(reference.x[i]) - px);
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    }
    return best;
  }

  function exportSvg() {
    const node = svgRef.current;
    if (!node) return;
    // Tokens resolve to nothing outside the document, so they are frozen into
    // the exported copy. The alternative is a file that looks blank when opened.
    const clone = node.cloneNode(true) as SVGSVGElement;
    const computed = getComputedStyle(document.documentElement);
    let markup = new XMLSerializer().serializeToString(clone);
    markup = markup.replace(/var\((--[a-z0-9-]+)\)/g, (_m, name) =>
      computed.getPropertyValue(name).trim() || '#000000');
    const blob = new Blob([markup], { type: 'image/svg+xml' });
    download(URL.createObjectURL(blob), 'wykres.svg');
  }

  function exportPng() {
    const node = svgRef.current;
    if (!node) return;
    const computed = getComputedStyle(document.documentElement);
    let markup = new XMLSerializer().serializeToString(node.cloneNode(true) as SVGSVGElement);
    markup = markup.replace(/var\((--[a-z0-9-]+)\)/g, (_m, name) =>
      computed.getPropertyValue(name).trim() || '#000000');
    const image = new Image();
    const blobUrl = URL.createObjectURL(new Blob([markup], { type: 'image/svg+xml' }));
    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = width * 2; canvas.height = height * 2;
      const context = canvas.getContext('2d');
      if (!context) return;
      context.fillStyle = computed.getPropertyValue('--bg').trim() || '#ffffff';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(blobUrl);
      download(canvas.toDataURL('image/png'), 'wykres.png');
    };
    image.src = blobUrl;
  }

  function download(href: string, filename: string) {
    const anchor = document.createElement('a');
    anchor.href = href; anchor.download = filename;
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
  }

  return (
    <div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 'auto', display: 'block', touchAction: 'none' }}
        role="img"
        aria-label={`${yLabel} w funkcji: ${xLabel}`}
        onMouseMove={(event) => {
          const index = pointerIndex(event);
          if (index === null) { setHover(null); return; }
          const rect = event.currentTarget.getBoundingClientRect();
          setHover({
            index,
            x: ((event.clientX - rect.left) / rect.width) * width,
            y: ((event.clientY - rect.top) / rect.height) * height,
          });
          if (drag) setDrag({ ...drag, to: ((event.clientX - rect.left) / rect.width) * width });
        }}
        onMouseLeave={() => { setHover(null); setDrag(null); }}
        onMouseDown={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const px = ((event.clientX - rect.left) / rect.width) * width;
          if (px >= PADDING.left && px <= PADDING.left + plotWidth) setDrag({ from: px, to: px });
        }}
        onMouseUp={() => {
          if (drag && Math.abs(drag.to - drag.from) > 12) {
            const invert = (px: number) => {
              const ratio = (px - PADDING.left) / plotWidth;
              if (logX) {
                const lo = Math.log10(Math.max(domain.xMin, 1e-12));
                const hi = Math.log10(Math.max(domain.xMax, 1e-11));
                return 10 ** (lo + ratio * (hi - lo));
              }
              return domain.xMin + ratio * (domain.xMax - domain.xMin);
            };
            const a = invert(Math.min(drag.from, drag.to));
            const b = invert(Math.max(drag.from, drag.to));
            setZoom([a, b]);
          }
          setDrag(null);
        }}
        onDoubleClick={() => setZoom(null)}
      >
        <defs>
          {/* Hatching marks the part of the band computed from fewer runs. */}
          <pattern id="thin-band" width="5" height="5" patternUnits="userSpaceOnUse"
                   patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" stroke="currentColor" strokeWidth="1.4" />
          </pattern>
        </defs>

        {yTicks.map((tick) => (
          <g key={`y${tick}`}>
            <line x1={PADDING.left} x2={PADDING.left + plotWidth}
                  y1={scaleY(tick)} y2={scaleY(tick)}
                  stroke="var(--border)" strokeWidth="1" />
            <text x={PADDING.left - 8} y={scaleY(tick) + 4} textAnchor="end"
                  fontSize="11" fill="var(--text-2)" fontFamily="var(--font-mono)">
              {tickLabel(tick, logY)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <g key={`x${tick}`}>
            <line x1={scaleX(tick)} x2={scaleX(tick)} y1={PADDING.top}
                  y2={PADDING.top + plotHeight} stroke="var(--border)" strokeWidth="1" />
            <text x={scaleX(tick)} y={PADDING.top + plotHeight + 16} textAnchor="middle"
                  fontSize="11" fill="var(--text-2)" fontFamily="var(--font-mono)">
              {tickLabel(tick, logX)}
            </text>
          </g>
        ))}

        <text x={PADDING.left + plotWidth / 2} y={height - 6} textAnchor="middle"
              fontSize="12" fill="var(--text-2)">{xLabel}</text>
        <text x={14} y={PADDING.top + plotHeight / 2} textAnchor="middle" fontSize="12"
              fill="var(--text-2)" transform={`rotate(-90 14 ${PADDING.top + plotHeight / 2})`}>
          {yLabel}
        </text>

        {visible.map((entry) => {
          const index = series.findIndex((s) => s.label === entry.label);
          const colour = PALETTE[index % PALETTE.length];
          const dash = DASHES[index % DASHES.length];
          const boundary = Math.min(entry.full_until_index, entry.x.length - 1);
          return (
            <g key={entry.label} color={colour}>
              {showBand && (
                <>
                  <path d={bandPath(entry, 0, boundary)} fill={colour} opacity="0.14" stroke="none" />
                  {/* Past the boundary at least one run has ended, so the band
                      is narrowing because there is less data, not because the
                      runs agree. Hatched so it cannot be misread as the same
                      thing. */}
                  <path d={bandPath(entry, boundary, entry.x.length - 1)}
                        fill="url(#thin-band)" opacity="0.30" stroke="none" />
                </>
              )}
              <path d={linePath(entry, 0, boundary)} fill="none" stroke={colour}
                    strokeWidth="1.9" strokeDasharray={dash} strokeLinejoin="round" />
              <path d={linePath(entry, boundary, entry.x.length - 1)} fill="none"
                    stroke={colour} strokeWidth="1.4" strokeDasharray="3 3" opacity="0.75" />
            </g>
          );
        })}

        {drag && Math.abs(drag.to - drag.from) > 4 && (
          <rect x={Math.min(drag.from, drag.to)} y={PADDING.top}
                width={Math.abs(drag.to - drag.from)} height={plotHeight}
                fill="var(--accent)" opacity="0.12" />
        )}

        {hover && visible.length > 0 && (
          <line x1={hover.x} x2={hover.x} y1={PADDING.top} y2={PADDING.top + plotHeight}
                stroke="var(--text-3)" strokeWidth="1" strokeDasharray="3 3" />
        )}

        <rect x={PADDING.left} y={PADDING.top} width={plotWidth} height={plotHeight}
              fill="none" stroke="var(--border-strong)" strokeWidth="1" />
      </svg>

      {hover && visible.length > 0 && (
        <div className="panel" style={{ padding: 'var(--space-2) var(--space-3)', marginTop: 8 }}>
          <div className="tiny muted mono" style={{ marginBottom: 4 }}>
            {xLabel} ≈ {integer(visible[0].x[hover.index])}
          </div>
          <table style={{ fontSize: 12 }}>
            <tbody>
              {visible.map((entry) => {
                const index = series.findIndex((s) => s.label === entry.label);
                const median = entry.median[hover.index];
                const n = entry.n_at_x[hover.index];
                return (
                  <tr key={entry.label}>
                    <td style={{ borderBottom: 'none', padding: '1px 8px 1px 0' }}>
                      <span style={{
                        display: 'inline-block', width: 14, height: 2,
                        background: PALETTE[index % PALETTE.length], verticalAlign: 'middle',
                      }} />
                    </td>
                    <td style={{ borderBottom: 'none', padding: '1px 8px 1px 0' }}>{entry.label}</td>
                    <td className="num" style={{ borderBottom: 'none', padding: '1px 8px 1px 0' }}>
                      {median === null ? '—' : decimal(median, 4)}
                    </td>
                    <td className="num tiny muted" style={{ borderBottom: 'none' }}>
                      {/* n at this budget, because it changes along the axis and
                          the reader needs to know when it has dropped. */}
                      n={n ?? 0}/{entry.n_runs}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="row" style={{ marginTop: 10, justifyContent: 'space-between' }}>
        <div className="row" style={{ gap: 'var(--space-3)' }}>
          {series.map((entry, index) => {
            const off = hidden.has(entry.label);
            // A gradient-free optimizer never increments the gradient counter,
            // so on the gradient axis its whole series sits at x=0 and there is
            // no curve to draw. Saying so is better than a legend entry
            // pointing at nothing, which reads as a bug.
            const flat = new Set(entry.x).size < 2;
            return (
              <button
                key={entry.label}
                className="subtle"
                onClick={() => setHidden((previous) => {
                  const next = new Set(previous);
                  if (next.has(entry.label)) next.delete(entry.label); else next.add(entry.label);
                  return next;
                })}
                style={{ opacity: off ? 0.4 : 1, fontSize: 12, padding: '2px 6px' }}
                title={flat
                  ? 'Ta metoda nie zużywa tej waluty budżetu — brak krzywej na tej osi'
                  : (off ? 'Pokaż serię' : 'Ukryj serię')}
              >
                <svg width="20" height="8" style={{ verticalAlign: 'middle', marginRight: 5 }}>
                  <line x1="0" y1="4" x2="20" y2="4" strokeWidth="2"
                        stroke={PALETTE[index % PALETTE.length]}
                        strokeDasharray={DASHES[index % DASHES.length]} />
                </svg>
                {entry.label}
                <span className="tiny muted mono" style={{ marginLeft: 5 }}>n={entry.n_runs}</span>
                {flat && (
                  <span className="tiny" style={{ marginLeft: 5, color: 'var(--warning)' }}>
                    brak na tej osi
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <div className="row" style={{ gap: 'var(--space-2)' }}>
          {zoom && <button className="subtle" onClick={() => setZoom(null)}>Pełny zakres</button>}
          <button className="subtle" onClick={exportSvg}>SVG</button>
          <button className="subtle" onClick={exportPng}>PNG</button>
        </div>
      </div>
      <div className="tiny dim" style={{ marginTop: 4 }}>
        Przeciągnij po wykresie, żeby przybliżyć. Podwójne kliknięcie wraca do pełnego zakresu.
        Kreskowana część wstęgi oznacza budżet, na którym część przebiegów już się zakończyła.
      </div>
    </div>
  );
}
