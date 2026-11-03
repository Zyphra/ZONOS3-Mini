/** Zyphra brand marks: the chevron mark and the wordmark (path data in `lib/marks.ts`), drawn in currentColor. */
import { MARK, MARK_VIEWBOX, WORD, WORD_VIEWBOX } from '../lib/marks';

export function ZyphraMark({ size = 22 }: { size?: number }) {
  return (
    <svg class="z-mark" width={size} height={(size * 164) / 224} viewBox={MARK_VIEWBOX} aria-hidden="true">
      {MARK.map((d) => (
        <path d={d} fill="currentColor" />
      ))}
    </svg>
  );
}

export function ZyphraWord({ height = 12 }: { height?: number }) {
  return (
    <svg class="z-word" height={height} viewBox={WORD_VIEWBOX} role="img" aria-label="Zyphra">
      {WORD.map((d) => (
        <path d={d} fill="currentColor" />
      ))}
    </svg>
  );
}
