/** Dark (default) or light Zyphra theme: `data-theme` on <html>; canvases redraw when it changes. */
import { effect, signal } from '@preact/signals';
import { load, persist } from './persist';

export type Theme = 'dark' | 'light';
export const theme = signal<Theme>(load<Theme>('theme', 'dark'));
persist('theme', theme);

effect(() => {
  if (theme.value === 'light') document.documentElement.dataset.theme = 'light';
  else delete document.documentElement.dataset.theme;
});

export const toggleTheme = () => (theme.value = theme.value === 'dark' ? 'light' : 'dark');

/** A design token's colour (`--o`, `--text-2` …) as a canvas fill, optionally at `alpha` (tokens are hex). */
export function token(name: string, alpha = 1): string {
  void theme.value;
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(raw);
  if (!m) return raw || 'currentColor';
  const n = parseInt(m[1], 16);
  const a = (m[2] ? parseInt(m[2], 16) / 255 : 1) * alpha;
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${+a.toFixed(3)})`;
}
