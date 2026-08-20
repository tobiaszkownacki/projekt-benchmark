import type { Tone } from '../types';

/** State is carried by the words, with colour as reinforcement.
 *
 * §14.5 asks for this explicitly, and the reason is practical rather than
 * decorative: colour alone fails for a colour-blind reader and disappears
 * entirely when a page is printed for a meeting. */
export function StatusPill({ label, tone, title }: { label: string; tone: Tone; title?: string }) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      {label}
    </span>
  );
}
