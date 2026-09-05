import type { Tone } from '../types';

/** State is carried by the words, with colour as reinforcement: colour alone
 * fails for a colour-blind reader and disappears when a page is printed. */
export function StatusPill({ label, tone, title }: { label: string; tone: Tone; title?: string }) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      {label}
    </span>
  );
}
