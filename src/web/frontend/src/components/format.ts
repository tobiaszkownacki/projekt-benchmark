/** Formatting helpers.
 *
 * §14.5 asks for numbers with a unit and a context rather than bare digits, so
 * these return the surrounding text too where there is any. Polish digit
 * grouping uses a narrow no-break space. */

// U+202F narrow no-break space: the Polish digit group separator. Written as
// an escape so it is visible in review and does not trip the linter.
const GROUP = '\u202f';

export function integer(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return Math.round(value).toLocaleString('pl-PL').replace(/[\u00a0\u2007\u202f ]/g, GROUP);
}

export function decimal(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return '∞';
  return value.toFixed(digits);
}

export function percent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) return '—';
  return `${value.toFixed(digits)}%`;
}

export function bytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  const units = ['B', 'kB', 'MB', 'GB'];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size < 10 && unit > 0 ? size.toFixed(1) : Math.round(size)} ${units[unit]}`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${Math.round(seconds % 60)} s`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${minutes % 60} min`;
}

export function timestamp(value: string | null | undefined): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString('pl-PL', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

/** Elapsed time in words. Used for queue waits, which is a number people care
 *  about far more than the absolute timestamp it is derived from. */
export function since(value: string | null | undefined, until?: string | null): string {
  if (!value) return '—';
  const start = new Date(value).getTime();
  const end = until ? new Date(until).getTime() : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end)) return '—';
  return duration(Math.max(0, (end - start) / 1000));
}

export function shortId(id: string): string {
  return id.slice(0, 8);
}

export function familyLabel(family: string | null): string {
  if (family === 'gradient') return 'gradientowa';
  if (family === 'gradient_free') return 'bezgradientowa';
  return '—';
}

export function suiteLabel(suite: string | null): string {
  if (suite === 'final') return 'finałowy';
  if (suite === 'test') return 'testowy';
  return '—';
}
